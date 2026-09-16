"""Per-trial NAVIS trainer for SEED-VII."""

from __future__ import annotations

import timeit

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loaders.seed_vii.bundle import SeedVIIBundle
from data_loaders.seed_vii.constants import NUM_EEG_NODES
from data_loaders.seed_vii.trials import TrialSequence, window_edge_tensors
from seed_models.base import SeedTrainConfig
from TGNN_models.navis import NAVISModel


class SeedNAVISModel(NAVISModel):
    """NAVIS for EEG: accumulate edge activity onto channel nodes."""

    def update_semi_labels(self, src: torch.Tensor, dst: torch.Tensor, msg: torch.Tensor):
        vals = msg.abs().mean(dim=1) if msg.ndim > 1 else msg.abs().reshape(-1)
        for node_id, value in zip(src.tolist(), vals.tolist()):
            self.semi_labels.data[node_id] += float(value)
        for node_id, value in zip(dst.tolist(), vals.tolist()):
            self.semi_labels.data[node_id] += float(value)


def run_seed_navis(
    bundle: SeedVIIBundle,
    config: SeedTrainConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model = SeedNAVISModel(
        num_nodes=NUM_EEG_NODES,
        num_classes=bundle.num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    evaluator = bundle.evaluator
    all_nodes = torch.arange(NUM_EEG_NODES, dtype=torch.long, device=device)

    def _epoch(trials: List[TrialSequence], train: bool) -> Dict[str, float]:
        model.train() if train else model.eval()
        trial_preds, trial_labels = [], []
        window_preds, window_labels = [], []
        total_loss = 0.0
        n_loss = 0

        max_trials = config.max_batches
        iterable = trials if max_trials is None else trials[:max_trials]
        if train:
            # See dyglib_trainer: fixed, emotion-blocked trial order otherwise.
            iterable = [iterable[i] for i in torch.randperm(len(iterable)).tolist()]
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for trial in tqdm(iterable, desc="train" if train else "eval", leave=False):
                model.reset()
                emotion = torch.tensor([trial.emotion_idx], dtype=torch.long, device=device)
                window_logits: List[torch.Tensor] = []
                n_windows = int(sum(1 for r in trial.windows if r.edges))
                if train:
                    optimizer.zero_grad()

                for t, rec in enumerate(trial.windows):
                    if not rec.edges:
                        continue
                    src, dst, _, msg = window_edge_tensors(rec, t, device=device)
                    model.update_semi_labels(src, dst, msg)
                    # Soft target: uniform prior + model state (no GT teacher forcing).
                    soft = torch.full(
                        (NUM_EEG_NODES, bundle.num_classes),
                        1.0 / bundle.num_classes,
                        device=device,
                    )
                    out = model(all_nodes, soft)
                    logits = out.mean(dim=0, keepdim=True)
                    window_logits.append(logits)
                    window_preds.append(logits.detach().cpu().numpy()[0])
                    window_labels.append(trial.emotion_idx)

                    if train:
                        # One step per TRIAL (see dyglib_trainer); scaled so the
                        # accumulated gradient is the trial mean.
                        loss = criterion(logits, emotion) / max(n_windows, 1)
                        # No retain_graph: NAVISModel writes its recurrent state with
                        # .data[...], which strips autograd, so each window's graph is
                        # independent and retaining them would only waste memory.
                        loss.backward()
                        total_loss += float(loss.detach())

                if train and n_windows:
                    optimizer.step()
                    n_loss += 1

                if not window_logits:
                    continue
                trial_logit = torch.stack(window_logits, dim=0).mean(dim=0)
                trial_preds.append(trial_logit.detach().cpu().numpy()[0])
                trial_labels.append(trial.emotion_idx)

        if not trial_preds:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "window_accuracy": 0.0, "window_f1": 0.0}
        trial_m = evaluator.eval({
            "y_true": np.asarray(trial_labels),
            "y_pred": np.stack(trial_preds, axis=0),
            "eval_metric": ["accuracy", "precision", "recall", "f1"],
        })
        window_m = evaluator.eval({
            "y_true": np.asarray(window_labels),
            "y_pred": np.stack(window_preds, axis=0),
            "eval_metric": ["accuracy", "precision", "recall", "f1"],
        })
        out = {
            "accuracy": trial_m["accuracy"],
            "precision": trial_m["precision"],
            "recall": trial_m["recall"],
            "f1": trial_m["f1"],
            "window_accuracy": window_m["accuracy"],
            "window_f1": window_m["f1"],
        }
        if train and n_loss:
            out["loss"] = total_loss / n_loss
        return out

    train_curve: List[Dict[str, float]] = []
    val_curve: List[Dict[str, float]] = []
    test_curve: List[Dict[str, float]] = []
    best_val = -1.0
    best_idx = 0

    for epoch in range(1, config.epochs + 1):
        _t0 = timeit.default_timer()
        train_m = _epoch(bundle.train_trials, train=True)
        _t1 = timeit.default_timer()
        val_m = _epoch(bundle.val_trials, train=False)
        _t2 = timeit.default_timer()
        test_m = _epoch(bundle.test_trials, train=False)
        _t3 = timeit.default_timer()
        train_curve.append(train_m)
        val_curve.append(val_m)
        test_curve.append(test_m)
        if val_m["accuracy"] > best_val:
            best_val = val_m["accuracy"]
            best_idx = epoch - 1
        print(
            f"Epoch {epoch:02d} train={train_m['accuracy']:.4f} "
            f"val={val_m['accuracy']:.4f} test={test_m['accuracy']:.4f} "
            f"(train {_t1-_t0:.1f}s val {_t2-_t1:.1f}s test {_t3-_t2:.1f}s)"
        )
        print(f"train_loss: {train_m.get('loss', float('nan')):.4f}")
        print(f"epoch_total: {_t3-_t0:.1f}s")

    best_test = test_curve[best_idx]
    return {
        "best_val_accuracy": best_val,
        "best_val_epoch": best_idx + 1,
        "best_test_accuracy": best_test["accuracy"],
        "best_test_precision": best_test["precision"],
            "best_test_recall": best_test["recall"],
            "best_test_f1": best_test["f1"],
        "train_curve": train_curve,
        "val_curve": val_curve,
        "test_curve": test_curve,
        "final_train_accuracy": train_curve[-1]["accuracy"],
        "final_train_f1": train_curve[-1]["f1"],
        "final_test_accuracy": test_curve[-1]["accuracy"],
        "final_test_precision": test_curve[-1]["precision"],
            "final_test_recall": test_curve[-1]["recall"],
            "final_test_f1": test_curve[-1]["f1"],
    }
