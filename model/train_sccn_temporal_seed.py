#!/usr/bin/env python3
"""Train / smoke-test ``SCCNTemporalClassifier`` on SEED-VII trial sequences.

Example (smoke)::

    cd /Users/nikitamalik/Downloads/toponetx_project
    source environment/topo_simplicial_env/bin/activate
    PYTHONPATH=models/temporal-simplicial-net:scripts/seed_vii \\
      python scripts/seed_vii/train_sccn_temporal_seed.py --smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
_MODEL_ROOT = _ROOT / "models" / "temporal-simplicial-net"
for p in (str(_SCRIPTS), str(_MODEL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from channel_positions import (  # noqa: E402
    DEFAULT_LOCS,
    channel_position_features,
)
from sccn_temporal import SCCNTemporalClassifier  # noqa: E402
from seed_sccn_dataset import (  # noqa: E402
    DEFAULT_WEIGHTED,
    EMOTION_ORDER,
    SeedTrialSCCNDataset,
    auto_subject_split,
    list_available_subjects,
    loso_splits,
    parse_subjects,
    resolve_method_dir,
    subject_cv_folds,
    trial_cv_folds,
    trial_split_indices,
)


def move_windows(windows: list[dict], device: torch.device) -> list[dict]:
    if not windows:
        return windows
    sample = next(iter(windows[0]["features"].values()))
    if sample.device == device:
        return windows
    out = []
    for w in windows:
        moved = {
            "features": {k: v.to(device, non_blocking=True) for k, v in w["features"].items()},
            "adjacencies": {k: v.to(device, non_blocking=True) for k, v in w["adjacencies"].items()},
            "incidences": {k: v.to(device, non_blocking=True) for k, v in w["incidences"].items()},
            "keys": w["keys"],
        }
        # cell_nodes indexes the channel-identity/position tables, so it has to follow
        # the features onto the GPU; t_frac/window_idx are plain scalars.
        if "cell_nodes" in w:
            moved["cell_nodes"] = {
                k: v.to(device, non_blocking=True) for k, v in w["cell_nodes"].items()
            }
        # local_ids index the dense recurrent state, so they must sit on the same
        # device as it; n_cells are plain ints describing its shape.
        if "local_ids" in w:
            moved["local_ids"] = {
                k: v.to(device, non_blocking=True) for k, v in w["local_ids"].items()
            }
        if "n_cells" in w:
            moved["n_cells"] = w["n_cells"]
        # the eye vector is consumed on-device by the eye encoder
        if "eye" in w:
            moved["eye"] = w["eye"].to(device, non_blocking=True)
        if "t_frac" in w:
            moved["t_frac"] = w["t_frac"]
        if "window_idx" in w:
            moved["window_idx"] = w["window_idx"]
        out.append(moved)
    return out


def preload_dataset_to_device(dataset, device: torch.device) -> None:
    """Move cached CPU window tensors to GPU once (not every epoch)."""
    if device.type != "cuda":
        return
    if not hasattr(dataset, "_windows"):
        return
    n = len(dataset)
    print(f"Preloading {n} trials onto {device} (one-time)...", flush=True)
    t0 = time.time()
    for i in range(n):
        dataset._windows[i] = move_windows(dataset._windows[i], device)
        dataset._labels[i] = dataset._labels[i].to(device, non_blocking=True)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  preload {i + 1}/{n}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Preload done in {time.time() - t0:.1f}s", flush=True)


class ConfusionCounts:
    """Per-class TP/FP/FN tallies for multi-class precision / recall / F1.

    Plain Python ints rather than tensors: predictions are already pulled to the host
    via ``.item()``, and the training loop is dominated by small-op overhead, so this
    adds no GPU work at all.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.tp = [0] * n_classes
        self.fp = [0] * n_classes
        self.fn = [0] * n_classes

    def update(self, pred: int, target: int) -> None:
        if pred == target:
            self.tp[target] += 1
        else:
            self.fp[pred] += 1
            self.fn[target] += 1

    def per_class_precision(self) -> list[float]:
        # A class that is never predicted (or never present) scores 0 rather than
        # being dropped -- same convention as sklearn's zero_division=0.
        return [
            self.tp[c] / (self.tp[c] + self.fp[c]) if (self.tp[c] + self.fp[c]) else 0.0
            for c in range(self.n_classes)
        ]

    def per_class_recall(self) -> list[float]:
        return [
            self.tp[c] / (self.tp[c] + self.fn[c]) if (self.tp[c] + self.fn[c]) else 0.0
            for c in range(self.n_classes)
        ]

    def per_class_f1(self) -> list[float]:
        out = []
        for p, r in zip(self.per_class_precision(), self.per_class_recall()):
            out.append(2 * p * r / (p + r) if (p + r) else 0.0)
        return out

    @staticmethod
    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def macro_precision(self) -> float:
        return self._mean(self.per_class_precision())

    def macro_recall(self) -> float:
        return self._mean(self.per_class_recall())

    def macro_f1(self) -> float:
        """Unweighted mean of per-class F1.

        Macro (not weighted) on purpose: it punishes collapsing onto one class, which
        weighted F1 and plain accuracy both hide. A model that always predicts the
        majority class scores ~0.15 accuracy here but only ~0.04 macro F1.
        """
        return self._mean(self.per_class_f1())

    def accuracy(self) -> float:
        total = sum(self.tp) + sum(self.fp)
        return sum(self.tp) / total if total else float("nan")

    def metrics(self) -> dict[str, float]:
        """All four reported metrics from one pass over the tallies."""
        return {
            "accuracy": self.accuracy(),
            "precision": self.macro_precision(),
            "recall": self.macro_recall(),
            "f1": self.macro_f1(),
        }


def format_per_class_f1(counts: ConfusionCounts, labels) -> str:
    parts = [
        f"{name}={score:.3f}" for name, score in zip(labels, counts.per_class_f1())
    ]
    return "  ".join(parts)


@torch.no_grad()
def evaluate(
    model, dataset, device, batch_size: int = 1, batched_forward: bool = False,
    micro_batch: int = 0,
) -> tuple[float, float, float, ConfusionCounts]:
    """Evaluate on a split.

    Validation and test are scored every epoch, so once training is batched this loop
    becomes a large share of each epoch. It can use the same block-diagonal batched
    forward -- with no optimizer involved there is nothing to change semantically, and
    ``forward_batch`` is verified equivalent to per-sample ``forward``.
    """
    model.eval()
    counts = ConfusionCounts(len(EMOTION_ORDER))
    total_loss = 0.0
    correct = 0
    n = 0
    order = list(range(len(dataset)))
    step = batch_size if (batched_forward and batch_size > 1) else 1
    if micro_batch > 0:
        step = min(step, micro_batch)
    for start in range(0, len(order), step):
        chunk = order[start : start + step]
        if len(chunk) > 1:
            seqs, ys = [], []
            for i in chunk:
                windows, y, _meta = dataset[i]
                seqs.append(move_windows(windows, device))
                ys.append(y.to(device))
            y_batch = torch.cat(ys, dim=0)
            logits = model.forward_batch(seqs)
            total_loss += float(
                F.cross_entropy(logits, y_batch, reduction="sum").item()
            )
            for pred, target in zip(logits.argmax(dim=-1).tolist(), y_batch.tolist()):
                counts.update(int(pred), int(target))
                correct += int(pred == target)
            n += len(chunk)
        else:
            windows, y, _meta = dataset[chunk[0]]
            windows = move_windows(windows, device)
            y = y.to(device)
            logits = model(windows)
            total_loss += float(F.cross_entropy(logits, y).item())
            pred = int(logits.argmax(dim=-1).item())
            target = int(y.item())
            counts.update(pred, target)
            correct += int(pred == target)
            n += 1
    if n == 0:
        return float("nan"), float("nan"), float("nan"), counts
    return total_loss / n, correct / n, counts.macro_f1(), counts


def train_one_epoch(
    model, dataset, opt, device, grad_clip: float = 0.0, batch_size: int = 1,
    batched_forward: bool = False, micro_batch: int = 0,
) -> tuple[float, float, float]:
    """One epoch of training with a minibatch size of ``batch_size`` trials.

    Each sample is a variable-length sequence of variable-size complexes, so trials are
    still run through the model one at a time; the batching is in the *optimizer*.
    Per-sample losses are divided by the batch's size and backpropagated as they go,
    so the accumulated gradient is exactly the gradient of the batch's mean loss --
    identical to a true minibatch step, without needing block-diagonal graph batching.

    Gradient clipping applies to the accumulated batch gradient (the right analogue of
    clipping a minibatch step), not per sample.

    ``batch_size=1`` reproduces the original per-trial update exactly.
    """
    model.train()
    counts = ConfusionCounts(len(EMOTION_ORDER))
    total_loss = 0.0
    correct = 0
    n = 0
    order = torch.randperm(len(dataset)).tolist()
    batches = [order[s : s + batch_size] for s in range(0, len(order), batch_size)]
    t_epoch = time.time()
    for step, chunk in enumerate(batches, start=1):
        opt.zero_grad(set_to_none=True)
        # divide by the *actual* chunk length so a short final batch is still the
        # mean over the samples it contains
        scale = 1.0 / len(chunk)
        if batched_forward and len(chunk) > 1:
            # Block-diagonal passes over sub-chunks of the batch. The recurrent state
            # tensor spans every trial in a pass and autograd keeps one copy per
            # timestep per rank per block, so peak memory grows linearly with the pass
            # size -- a full 256-trial pass needs hundreds of GB. micro_batch caps the
            # pass size; scaling each sub-chunk's loss by its share of the batch keeps
            # the accumulated gradient exactly the batch mean, so the optimizer still
            # sees the requested batch size.
            micro = micro_batch if micro_batch > 0 else len(chunk)
            for ms in range(0, len(chunk), micro):
                sub = chunk[ms : ms + micro]
                seqs, ys = [], []
                for i in sub:
                    windows, y, _meta = dataset[i]
                    seqs.append(move_windows(windows, device))
                    ys.append(y.to(device))
                y_sub = torch.cat(ys, dim=0)
                logits = model.forward_batch(seqs)
                # mean over the sub-chunk, reweighted to the sub-chunk's share
                loss = F.cross_entropy(logits, y_sub)
                (loss * (len(sub) / len(chunk))).backward()
                total_loss += float(loss.item()) * len(sub)
                for pred, target in zip(logits.argmax(dim=-1).tolist(), y_sub.tolist()):
                    counts.update(int(pred), int(target))
                    correct += int(pred == target)
                n += len(sub)
        else:
            for i in chunk:
                windows, y, _meta = dataset[i]
                windows = move_windows(windows, device)
                y = y.to(device)
                logits = model(windows)
                loss = F.cross_entropy(logits, y)
                (loss * scale).backward()
                total_loss += float(loss.item())
                pred = int(logits.argmax(dim=-1).item())
                target = int(y.item())
                counts.update(pred, target)
                correct += int(pred == target)
                n += 1
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if step == 1 or step % 5 == 0 or step == len(batches):
            elapsed = time.time() - t_epoch
            rate = n / max(elapsed, 1e-6)
            eta = (len(order) - n) / max(rate, 1e-6)
            print(
                f"  train batch {step}/{len(batches)} ({n}/{len(order)} trials)  "
                f"loss={total_loss / n:.4f} acc={correct / n:.3f}  "
                f"{rate:.2f} trials/s  eta={eta / 60:.1f}min",
                flush=True,
            )
    return total_loss / max(n, 1), correct / max(n, 1), counts.macro_f1()


def build_model(args, device, eye_dim: int = 0, eye_stats=None) -> SCCNTemporalClassifier:
    position_features = None
    if args.use_positions:
        position_features = channel_position_features(
            args.locs_file, n_fourier_bands=args.pos_fourier_bands
        )
        print(
            f"[model] electrode positions: {tuple(position_features.shape)} "
            f"from {args.locs_file or DEFAULT_LOCS}",
            flush=True,
        )
    model = SCCNTemporalClassifier(
        in_channels=args.feat_dim,
        channels=args.channels,
        max_rank=args.max_rank,
        num_classes=len(EMOTION_ORDER),
        n_layers=args.n_layers,
        variant=args.variant,
        aggr_func=args.aggr_func,
        update_func=(None if args.update_func == "none" else args.update_func),
        dropout=args.dropout,
        cell_learner=args.cell_learner,
        position_features=position_features,
        temporal_phase=args.temporal_phase,
        cache_augmented=not args.no_cache_augmented,
        eye_dim=eye_dim,
        eye_fusion=args.eye_fusion,
        eye_stats=eye_stats,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[model] variant={args.variant} channels={args.channels} "
        f"batch_size={args.batch_size} micro_batch={args.micro_batch} "
        f"eye_fusion={args.eye_fusion} eye_dim={eye_dim} "
        f"batched_forward={args.batched_forward} "
        f"cell_learner={args.cell_learner} positions={args.use_positions} "
        f"temporal_phase={args.temporal_phase} dropout={args.dropout} "
        f"trainable_params={n_params:,}",
        flush=True,
    )
    return model


def eye_feature_stats(dataset, train_idx: list[int]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-dimension mean/std of the eye features over the TRAINING trials only.

    The raw features span ~7 orders of magnitude across dimensions, so they have to be
    standardized before a linear layer can use them. Computing the statistics over the
    whole dataset would leak validation and test information into training, so this
    walks only the fold's training indices.
    """
    total = None
    total_sq = None
    n = 0
    for i in train_idx:
        windows, _y, _meta = dataset[i]
        for w in windows:
            e = w.get("eye")
            if e is None or e.numel() == 0:
                continue
            e = e.double()
            if total is None:
                total = torch.zeros_like(e)
                total_sq = torch.zeros_like(e)
            total += e
            total_sq += e * e
            n += 1
    if total is None or n == 0:
        return None
    mean = total / n
    var = (total_sq / n) - mean * mean
    return mean.float().cpu(), var.clamp(min=0).sqrt().float().cpu()


def eye_dim_of(dataset) -> int:
    """Width of the eye vector in this dataset, or 0 when the extract has none."""
    if len(dataset) == 0:
        return 0
    windows, _y, _meta = dataset[0]
    for w in windows:
        e = w.get("eye")
        if e is not None and e.numel() > 0:
            return int(e.numel())
    return 0


def indices_for_subjects(trials, subjects) -> list[int]:
    """Dataset indices of every trial belonging to one of ``subjects``."""
    wanted = set(subjects)
    return [i for i, t in enumerate(trials) if t.subject in wanted]


def _is_better(score: float, best: float, metric: str) -> bool:
    if score != score:  # nan
        return False
    if best != best:
        return True
    return score < best if metric.endswith("loss") else score > best


def run_fold(args, device, full_ds, split_idx, fold_name: str) -> dict | None:
    """Train one fold from scratch; return the test metrics at its best val epoch.

    Model selection uses ``args.select_metric`` on the validation split, and the
    returned numbers are the *test* metrics recorded at that epoch -- so the test set
    never influences which model is chosen.
    """
    train_idx, val_idx, test_idx = split_idx
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx) if val_idx else None
    test_ds = Subset(full_ds, test_idx)

    print(
        f"\n===== {fold_name} : train={len(train_ds)} val={len(val_ds) if val_ds else 0} "
        f"test={len(test_ds)} trials =====",
        flush=True,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        print(f"[warn] {fold_name}: empty train or test split, skipping", file=sys.stderr, flush=True)
        return None
    if val_ds is None or len(val_ds) == 0:
        print(
            f"[warn] {fold_name}: empty validation split -> falling back to the final "
            "epoch instead of best-val selection",
            file=sys.stderr,
            flush=True,
        )

    # Fresh model + optimizer per fold: carrying either across folds would leak
    # information from a fold's test subjects into the next fold's training.
    eye_dim, eye_stats = 0, None
    if args.eye_fusion != "none":
        eye_dim = eye_dim_of(train_ds)
        if eye_dim == 0:
            print(
                f"[warn] {fold_name}: --eye-fusion {args.eye_fusion} requested but this "
                "extract has no eye features; continuing EEG-only",
                file=sys.stderr, flush=True,
            )
        else:
            # statistics from this fold's TRAIN split only
            eye_stats = eye_feature_stats(full_ds, train_idx)
            if eye_stats is not None:
                print(
                    f"[{fold_name}] eye stats from {len(train_idx)} train trials: "
                    f"mean|range|=[{eye_stats[0].min():.3g},{eye_stats[0].max():.3g}] "
                    f"std|range|=[{eye_stats[1].min():.3g},{eye_stats[1].max():.3g}]",
                    flush=True,
                )
    model = build_model(args, device, eye_dim, eye_stats)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    nan = float("nan")
    best_score = nan
    best: dict | None = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_ds, opt, device, args.grad_clip, args.batch_size,
            args.batched_forward, args.micro_batch,
        )
        if val_ds is not None and len(val_ds) > 0:
            va_loss, va_acc, va_f1, va_counts = evaluate(
                model, val_ds, device, args.batch_size, args.batched_forward,
                args.micro_batch,
            )
        else:
            va_loss, va_acc, va_f1, va_counts = nan, nan, nan, None
        te_loss, te_acc, te_f1, te_counts = evaluate(
            model, test_ds, device, args.batch_size, args.batched_forward,
            args.micro_batch,
        )
        print(
            f"[{fold_name}] epoch {epoch}/{args.epochs}  "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} train_f1={tr_f1:.3f}  "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} val_f1={va_f1:.3f}  "
            f"test_loss={te_loss:.4f} test_acc={te_acc:.3f} test_f1={te_f1:.3f}",
            flush=True,
        )

        score = {"val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1}[args.select_metric]
        no_val = va_counts is None
        if no_val or _is_better(score, best_score, args.select_metric):
            best_score = score
            best = {"fold": fold_name, "epoch": epoch, **te_counts.metrics()}
            best["per_class_f1"] = te_counts.per_class_f1()
            if args.save:
                best_state = copy.deepcopy(model.state_dict())

    if best is not None:
        print(
            f"[{fold_name}] best {args.select_metric} at epoch {best['epoch']}  ->  "
            f"test acc={best['accuracy']:.4f} precision={best['precision']:.4f} "
            f"recall={best['recall']:.4f} f1={best['f1']:.4f}",
            flush=True,
        )
        # Near-zero F1 on most classes next to a plausible accuracy is the signature
        # of collapsing onto one or two classes; the aggregates alone hide it.
        parts = [f"{n}={s:.3f}" for n, s in zip(EMOTION_ORDER, best["per_class_f1"])]
        print(f"[{fold_name}] test per-class f1: {'  '.join(parts)}", flush=True)
    if args.save and best_state is not None:
        out = args.save.with_name(f"{args.save.stem}_{fold_name}{args.save.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": best_state,
                "args": vars(args),
                "emotion_order": list(EMOTION_ORDER),
                "fold": fold_name,
                "best_epoch": best["epoch"],
            },
            out,
        )
        print(f"[{fold_name}] saved {out}", flush=True)
    return best


def report_cv(results: list[dict], select_metric: str) -> None:
    """Print per-fold metrics and their mean +/- std across folds."""
    keys = ("accuracy", "precision", "recall", "f1")
    print("\n" + "=" * 72, flush=True)
    print(f"CROSS-VALIDATION SUMMARY  ({len(results)} folds, model selected on {select_metric})", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'fold':<26} {'epoch':>5}  " + "  ".join(f"{k:>9}" for k in keys), flush=True)
    for r in results:
        print(
            f"  {r['fold']:<26} {r['epoch']:>5}  " + "  ".join(f"{r[k]:>9.4f}" for k in keys),
            flush=True,
        )
    print("  " + "-" * 70, flush=True)
    line_mean, line_std = [], []
    for k in keys:
        vals = [r[k] for r in results]
        mean = statistics.mean(vals)
        # sample std (n-1): these folds are a sample of possible partitions
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        line_mean.append(f"{mean:>9.4f}")
        line_std.append(f"{std:>9.4f}")
    print(f"  {'MEAN':<26} {'':>5}  " + "  ".join(line_mean), flush=True)
    print(f"  {'STD':<26} {'':>5}  " + "  ".join(line_std), flush=True)
    print("=" * 72, flush=True)
    for k in keys:
        vals = [r[k] for r in results]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {k:<10} = {mean:.4f} +/- {std:.4f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_WEIGHTED,
        help="Root with {clique,neighbor,vietoris_rips}/subject_XX_windows.pkl",
    )
    p.add_argument("--method", default="clique", choices=("clique", "neighbor", "vietoris_rips"))
    p.add_argument("--subjects", default=None, help="e.g. 1 or 1-5; default=all available")
    p.add_argument(
        "--train-subjects",
        default=None,
        help="Train subject ids, e.g. 1-13 (overrides LOSO from --subjects/--test-subject)",
    )
    p.add_argument(
        "--test-subjects",
        default=None,
        help="Test subject ids, e.g. 14-20 (overrides single --test-subject)",
    )
    p.add_argument("--test-subject", type=int, default=None, help="LOSO held-out subject")
    p.add_argument(
        "--split-mode",
        default="subject",
        choices=("subject", "trial"),
        help=(
            "'subject' = disjoint subjects per split, tests cross-subject generalization "
            "(default; uses --train-subjects/--val-subjects/--test-subjects, or an "
            "automatic train/val/test-frac split of --subjects if none are given). "
            "'trial' = pool every requested subject's trials and split by trial "
            "(stratified by emotion), via --train-frac/--val-frac."
        ),
    )
    p.add_argument(
        "--val-subjects",
        default=None,
        help="Val subject ids, e.g. 5,10,15,20 (subject split mode; optional)",
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Train fraction: trial split mode, or automatic subject split when no "
        "--train-subjects/--test-subjects given",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Val fraction (test fraction/count is whatever remains)",
    )
    p.add_argument("--split-seed", type=int, default=42)
    # ── Cross-validation ────────────────────────────────────────────────
    p.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="K-fold cross-validation. 1 (default) keeps the original single "
        "train/val/test split. With K>1 every subject (subject mode) or trial "
        "(trial mode, stratified by emotion) is held out for test exactly once, "
        "and mean/std over folds is reported.",
    )
    p.add_argument(
        "--loso",
        action="store_true",
        help="Leave-one-subject-out CV (subject mode only): one fold per subject, "
        "so with 20 subjects you get 20 folds each testing on a single held-out "
        "subject. Overrides --n-folds.",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Run only this fold (1-based) instead of all K -- lets each fold go to a "
        "separate job in parallel rather than K x runtime in one job. "
        "Combine the resulting --metrics-out JSONs with aggregate_folds.py.",
    )
    p.add_argument(
        "--select-metric",
        default="val_f1",
        choices=("val_f1", "val_acc", "val_loss"),
        help="Validation metric picking each fold's reported epoch. Default val_f1: "
        "this model overfits so fast that val_loss bottoms out around epoch 3-8, "
        "while still collapsed onto one class (test_f1 ~0.04), so val_loss selects "
        "a degenerate model here.",
    )
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Write per-fold metrics as JSON (for aggregating parallel fold jobs)",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Trials per optimizer step. 1 (default) reproduces the original "
        "per-trial update. Larger values average the gradient over that many "
        "trials before stepping, which CHANGES the optimization trajectory.",
    )
    p.add_argument(
        "--update-func",
        default="sigmoid",
        choices=("sigmoid", "relu", "tanh", "none"),
        help="Activation applied inside each SCCN layer. The model has always "
        "defaulted to sigmoid; 'relu' avoids the saturation that a sigmoid over an "
        "unnormalized sum aggregation is prone to.",
    )
    p.add_argument(
        "--aggr-func",
        default="sum",
        choices=("sum", "mean"),
        help="How a cell aggregates messages from its neighbours.",
    )
    p.add_argument(
        "--eye-fusion",
        default="none",
        choices=("none", "window"),
        help="Fuse the per-window eye-tracking vector. 'window' concatenates it onto "
        "each window's pooled EEG embedding before the final aggregation; 'none' is "
        "the EEG-only control.",
    )
    p.add_argument(
        "--micro-batch",
        type=int,
        default=0,
        help="Max trials per block-diagonal forward pass (0 = the whole batch). "
        "Caps peak memory without changing the optimizer's batch size: sub-chunk "
        "losses are reweighted so the accumulated gradient is still the batch mean.",
    )
    p.add_argument(
        "--batched-forward",
        action="store_true",
        help="Run each minibatch as one block-diagonal graph instead of sample by "
        "sample. Same gradient, far fewer GPU kernel launches. Verify with "
        "scripts/verify_equivalence.py --check-batched before relying on it.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Max grad norm for clip_grad_norm_ (<=0 disables)",
    )
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-rank", type=int, default=2)
    p.add_argument("--feat-dim", type=int, default=5)
    p.add_argument(
        "--variant",
        default="sist",
        choices=("sist", "temporal_then_spatial", "spatial_then_temporal"),
    )
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay; another lever against the train/test gap",
    )
    # ── Optional priors (all default off so each can be ablated cleanly) ──
    p.add_argument(
        "--cell-learner",
        action="store_true",
        help="Learnable per-channel identity embeddings, pooled to every rank "
        "(the node / edge / triangle learners)",
    )
    p.add_argument(
        "--use-positions",
        action="store_true",
        help="Encode electrode scalp coordinates (centroid + spread) per cell",
    )
    p.add_argument(
        "--locs-file",
        type=Path,
        default=None,
        help=f"Montage .locs file for --use-positions (default: {DEFAULT_LOCS})",
    )
    p.add_argument(
        "--pos-fourier-bands",
        type=int,
        default=4,
        help="Fourier bands lifting raw electrode coordinates (0 = raw x,y,z)",
    )
    p.add_argument(
        "--temporal-phase",
        action="store_true",
        help="Sinusoidal encoding of each window's position within its trial "
        "(early / mid / late)",
    )
    p.add_argument(
        "--no-cache-augmented",
        action="store_true",
        help="Disable memoizing the SiST augmented matrices (saves GPU memory, "
        "costs speed; numerically identical either way)",
    )
    p.add_argument("--max-trials-per-subject", type=int, default=None)
    p.add_argument("--max-windows-per-trial", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run: subject 1, 2 trials, 4 windows, 1 epoch",
    )
    p.add_argument("--save", type=Path, default=None, help="Optional checkpoint path")
    args = p.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.smoke:
        args.subjects = "1"
        args.max_trials_per_subject = 2
        args.max_windows_per_trial = 4
        args.epochs = 1
        args.n_layers = 1
        args.channels = 16
        print("SMOKE mode: subject=1, trials=2, windows<=4, epochs=1", flush=True)

    if args.train_frac + args.val_frac >= 1.0 or args.train_frac <= 0 or args.val_frac < 0:
        print(
            f"--train-frac ({args.train_frac}) + --val-frac ({args.val_frac}) must leave "
            "a positive remainder for test",
            file=sys.stderr,
        )
        return 1

    method_dir = resolve_method_dir(args.data_root, args.method)
    available = list_available_subjects(method_dir)
    if not available:
        print(f"No subject pickles in {method_dir}", file=sys.stderr)
        return 1

    common = dict(
        data_root=args.data_root,
        method=args.method,
        max_rank=args.max_rank,
        feat_dim=args.feat_dim,
        max_trials_per_subject=args.max_trials_per_subject,
        max_windows_per_trial=args.max_windows_per_trial,
        dtype=torch.float32,
    )

    # ---- resolve the subject pool ------------------------------------------------
    if args.subjects is None:
        subject_ids = available
    else:
        subject_ids = [s for s in parse_subjects(args.subjects) if s in available]
        if not subject_ids:
            print(f"Requested subjects not available. Have: {available}", file=sys.stderr)
            return 1

    explicit_subject_split = args.split_mode == "subject" and (
        args.train_subjects is not None or args.test_subjects is not None
    )
    if explicit_subject_split:
        if args.train_subjects is None or args.test_subjects is None:
            print(
                "Provide both --train-subjects and --test-subjects (e.g. 1-13 and 14-20)",
                file=sys.stderr,
            )
            return 1
        ex_train = [s for s in parse_subjects(args.train_subjects) if s in available]
        ex_test = [s for s in parse_subjects(args.test_subjects) if s in available]
        ex_val = (
            [s for s in parse_subjects(args.val_subjects) if s in available]
            if args.val_subjects is not None
            else []
        )
        if not ex_train or not ex_test:
            print(f"Empty train/test after filtering. Have: {available}", file=sys.stderr)
            return 1
        subject_ids = sorted(set(ex_train) | set(ex_val) | set(ex_test))

    # ---- build the dataset ONCE ---------------------------------------------------
    # Every split is an index list into this one dataset, so the ~90s build and the
    # GPU preload happen once rather than per split (or per fold).
    print(
        f"method={args.method} device={device} split_mode={args.split_mode} "
        f"cv={'LOSO' if args.loso else args.n_folds} select={args.select_metric} "
        f"subjects={subject_ids} ({len(subject_ids)} subj) seed={args.split_seed}",
        flush=True,
    )
    full_ds = SeedTrialSCCNDataset(subjects=subject_ids, **common)
    if len(full_ds) == 0:
        print("Empty dataset", file=sys.stderr)
        return 1
    preload_dataset_to_device(full_ds, device)
    print(f"dataset: {len(full_ds)} trials  classes={list(EMOTION_ORDER)}", flush=True)

    # ---- build the fold list ------------------------------------------------------
    # LOSO is subject-only; in trial mode there are no subjects to leave out.
    if args.loso and args.split_mode != "subject":
        print("--loso requires --split-mode subject", file=sys.stderr)
        return 1
    # LOSO is just K-fold with one subject per fold.
    n_folds = len(subject_ids) if args.loso else args.n_folds

    folds: list[tuple[str, tuple[list[int], list[int], list[int]]]] = []
    if n_folds > 1:
        if args.split_mode == "trial":
            for k, (tr, va, te) in enumerate(
                trial_cv_folds(full_ds.trials, n_folds, args.val_frac, args.split_seed), 1
            ):
                folds.append((f"fold{k}", (tr, va, te)))
        else:
            for k, (trs, vas, tes) in enumerate(
                subject_cv_folds(subject_ids, n_folds, args.val_frac, args.split_seed), 1
            ):
                # Name LOSO folds by the held-out subject -- far more useful than an
                # index when reading logs or chasing a subject-specific outlier.
                name = f"loso_s{tes[0]:02d}" if args.loso else f"fold{k}"
                print(f"  {name}: test subjects {tes}  val {vas} ({len(trs)} train subj)", flush=True)
                folds.append(
                    (
                        name,
                        (
                            indices_for_subjects(full_ds.trials, trs),
                            indices_for_subjects(full_ds.trials, vas),
                            indices_for_subjects(full_ds.trials, tes),
                        ),
                    )
                )
    elif args.split_mode == "trial":
        # Single split, trial mode: pool all trials, split stratified by emotion.
        tr, va, te = trial_split_indices(
            full_ds.trials, args.train_frac, args.val_frac, args.split_seed
        )
        folds.append(("single", (tr, va, te)))
    else:
        # Single split, subject mode.
        if explicit_subject_split:
            train_subjects, val_subjects, test_subjects = ex_train, ex_val, ex_test
        elif len(subject_ids) == 1:
            train_subjects, val_subjects, test_subjects = subject_ids, [], subject_ids
            print(f"Single-subject mode: train+eval on subject {subject_ids[0]}", flush=True)
        elif args.test_subject is not None:
            # explicit LOSO held-out subject: no automatic validation split
            train_subjects, test_subjects = loso_splits(subject_ids, args.test_subject)
            val_subjects = []
        else:
            train_subjects, val_subjects, test_subjects = auto_subject_split(
                subject_ids, args.train_frac, args.val_frac, args.split_seed
            )
        print(
            f"train_subjects={train_subjects} ({len(train_subjects)}) "
            f"val_subjects={val_subjects} ({len(val_subjects)}) "
            f"test_subjects={test_subjects} ({len(test_subjects)})",
            flush=True,
        )
        folds.append(
            (
                "single",
                (
                    indices_for_subjects(full_ds.trials, train_subjects),
                    indices_for_subjects(full_ds.trials, val_subjects),
                    indices_for_subjects(full_ds.trials, test_subjects),
                ),
            )
        )

    # ---- optionally run just one fold (for parallel per-fold jobs) -------
    if args.fold is not None:
        if not 1 <= args.fold <= len(folds):
            print(f"--fold {args.fold} out of range 1..{len(folds)}", file=sys.stderr)
            return 1
        folds = [folds[args.fold - 1]]
        print(f"Running only {folds[0][0]}", flush=True)

    # ---- train ------------------------------------------------------------------
    results: list[dict] = []
    t_all = time.time()
    for name, split_idx in folds:
        res = run_fold(args, device, full_ds, split_idx, name)
        if res is not None:
            results.append(res)

    if not results:
        print("No fold produced a result", file=sys.stderr)
        return 1

    if len(results) > 1:
        report_cv(results, args.select_metric)
    else:
        r = results[0]
        print(
            f"\nRESULT ({r['fold']}, best {args.select_metric} @ epoch {r['epoch']}): "
            f"accuracy={r['accuracy']:.4f} precision={r['precision']:.4f} "
            f"recall={r['recall']:.4f} f1={r['f1']:.4f}",
            flush=True,
        )
    print(f"total time {(time.time() - t_all) / 3600:.2f}h", flush=True)

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "select_metric": args.select_metric,
            "folds": results,
        }
        if len(results) > 1:
            payload["summary"] = {
                k: {
                    "mean": statistics.mean([r[k] for r in results]),
                    "std": statistics.stdev([r[k] for r in results]) if len(results) > 1 else 0.0,
                }
                for k in ("accuracy", "precision", "recall", "f1")
            }
        args.metrics_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.metrics_out}", flush=True)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
