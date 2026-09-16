"""Per-trial DyGLib trainer for SEED-VII (TGAT / JODIE / GraphMixer / DyGFormer).

Each trial is its own temporal graph. Neighbor sampling never crosses trials.
Emotion is predicted from mean-pooled node embeddings after each window.
"""

from __future__ import annotations

import timeit

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loaders.seed_vii.bundle import SeedVIIBundle, dyg_trial_to_data
from dyglib.models.DyGFormer import DyGFormer
from dyglib.models.GraphMixer import GraphMixer
from dyglib.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from dyglib.models.TGAT import TGAT
from dyglib.models.modules import MLPClassifier
from dyglib.utils.utils import convert_to_gpu, create_optimizer, get_neighbor_sampler
from seed_models.base import SeedTrainConfig
from TGNN_models.dyglib_runner import MODEL_DEFAULTS, _compute_embeddings


def build_seed_dyglib_model(
    model_name: str,
    bundle: SeedVIIBundle,
    config: SeedTrainConfig,
    device: torch.device,
) -> tuple[nn.Sequential, str]:
    defaults = MODEL_DEFAULTS.get(model_name.lower(), {})
    num_neighbors = config.num_neighbors or defaults.get("num_neighbors", 20)
    num_layers = defaults.get("num_layers", 2)
    num_heads = defaults.get("num_heads", 2)
    dropout = defaults.get("dropout", 0.1)

    # Placeholder sampler; replaced per-trial during training.
    if not bundle.train_dyg_trials:
        raise ValueError("No training trials in bundle")
    placeholder = dyg_trial_to_data(bundle.train_dyg_trials[0])
    neighbor_sampler = get_neighbor_sampler(placeholder, sample_neighbor_strategy="recent")

    node_raw_features = bundle.node_raw_features
    edge_raw_features = bundle.edge_raw_features
    name = model_name.lower()

    if name == "tgat":
        backbone = TGAT(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=config.time_dim,
            output_dim=config.embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
        )
        dyglib_name = "TGAT"
    elif name == "graphmixer":
        backbone = GraphMixer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=config.time_dim,
            output_dim=config.embedding_dim,
            num_tokens=num_neighbors,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
        )
        dyglib_name = "GraphMixer"
    elif name == "dygformer":
        backbone = DyGFormer(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=config.time_dim,
            channel_embedding_dim=defaults.get("channel_embedding_dim", 50),
            output_dim=config.embedding_dim,
            patch_size=defaults.get("patch_size", 8),
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_input_sequence_length=defaults.get("max_input_sequence_length", 64),
            device=device,
        )
        dyglib_name = "DyGFormer"
    elif name == "jodie":
        # Time-shift stats from first few train trials (stable enough).
        srcs, dsts, times = [], [], []
        for packed in bundle.train_dyg_trials[: min(64, len(bundle.train_dyg_trials))]:
            srcs.append(packed["src_node_ids"])
            dsts.append(packed["dst_node_ids"])
            times.append(packed["node_interact_times"])
        shifts = compute_src_dst_node_time_shifts(
            np.concatenate(srcs), np.concatenate(dsts), np.concatenate(times)
        )
        backbone = MemoryModel(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=config.time_dim,
            output_dim=config.embedding_dim,
            model_name="JODIE",
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            src_node_mean_time_shift=shifts[0],
            src_node_std_time_shift=shifts[1],
            dst_node_mean_time_shift_dst=shifts[2],
            dst_node_std_time_shift=shifts[3],
            device=device,
        )
        dyglib_name = "JODIE"
    else:
        raise ValueError(f"Unknown DyGLib model: {model_name}")

    classifier = MLPClassifier(
        input_dim=config.embedding_dim,
        output_dim=bundle.num_classes,
        dropout=dropout,
    )
    return convert_to_gpu(nn.Sequential(backbone, classifier), device=device), dyglib_name


def run_seed_dyglib_model(
    model_name: str,
    bundle: SeedVIIBundle,
    config: SeedTrainConfig,
    device: torch.device,
) -> Dict[str, Any]:
    model, dyglib_name = build_seed_dyglib_model(model_name, bundle, config, device)
    optimizer = create_optimizer(model=model, optimizer_name="Adam", learning_rate=config.lr)
    loss_func = nn.CrossEntropyLoss()
    defaults = MODEL_DEFAULTS.get(model_name.lower(), {})
    num_neighbors = config.num_neighbors or defaults.get("num_neighbors", 20)
    time_gap = defaults.get("time_gap", 2000)

    def _epoch(packed_trials: List[dict], train: bool) -> Dict[str, float]:
        if train:
            model.train()
        else:
            model.eval()

        trial_preds: List[np.ndarray] = []
        trial_labels: List[int] = []
        window_preds: List[np.ndarray] = []
        window_labels: List[int] = []
        total_loss = 0.0
        n_loss = 0

        max_trials = config.max_batches
        iterable = packed_trials if max_trials is None else packed_trials[:max_trials]
        if train:
            # Trials are stored sorted by (subject, trial), so without this every
            # epoch walks an identical, emotion-blocked sequence. Combined with a
            # per-trial step that is a strongly correlated gradient stream.
            iterable = [iterable[i] for i in torch.randperm(len(iterable)).tolist()]
        desc = "train" if train else "eval"
        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for packed in tqdm(iterable, desc=desc, leave=False):
                trial = packed["trial"]
                data = dyg_trial_to_data(packed)
                sampler = get_neighbor_sampler(data, sample_neighbor_strategy="recent")
                # JODIE has no graph-attention neighbor sampler; TGN/DyRep/TGAT/etc. do.
                if dyglib_name != "JODIE" and hasattr(model[0], "set_neighbor_sampler"):
                    model[0].set_neighbor_sampler(sampler)
                if dyglib_name in ["JODIE", "DyRep", "TGN"] and hasattr(model[0], "memory_bank"):
                    model[0].memory_bank.__init_memory_bank__()

                bounds = packed["window_bounds"]
                window_logits: List[torch.Tensor] = []
                emotion = torch.tensor([trial.emotion_idx], dtype=torch.long, device=device)
                n_windows = int(sum(1 for a, b in bounds if b > a))
                if train:
                    optimizer.zero_grad()

                for start, end in bounds:
                    if end <= start:
                        continue
                    batch_src = packed["src_node_ids"][start:end]
                    batch_dst = packed["dst_node_ids"][start:end]
                    batch_t = packed["node_interact_times"][start:end]
                    batch_edge_ids = packed["edge_ids"][start:end]

                    src_emb, _ = _compute_embeddings(
                        dyglib_name,
                        model[0],
                        batch_src,
                        batch_dst,
                        batch_t,
                        batch_edge_ids,
                        num_neighbors,
                        time_gap,
                    )
                    graph_emb = src_emb.mean(dim=0, keepdim=True)
                    logits = model[1](x=graph_emb)
                    if logits.ndim > 2:
                        logits = logits.squeeze(1)
                    window_logits.append(logits)
                    window_preds.append(logits.detach().cpu().numpy()[0])
                    window_labels.append(trial.emotion_idx)

                    if train:
                        # Accumulate: every window of a trial carries the SAME label,
                        # so stepping per window gave ~43 identical-label single-sample
                        # updates in a row (43,954 steps/epoch) and the model tracked
                        # the most recent class instead of converging. Scale by the
                        # window count so the accumulated gradient is the trial mean.
                        loss = loss_func(logits, emotion) / max(n_windows, 1)
                        loss.backward()
                        total_loss += float(loss.item())
                        if dyglib_name in ["JODIE", "DyRep", "TGN"]:
                            model[0].memory_bank.detach_memory_bank()

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
        trial_m = bundle.evaluator.eval({
            "y_true": np.asarray(trial_labels),
            "y_pred": np.stack(trial_preds, axis=0),
            "eval_metric": ["accuracy", "precision", "recall", "f1"],
        })
        window_m = bundle.evaluator.eval({
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

    train_curve, val_curve, test_curve = [], [], []
    best_val = -1.0
    best_idx = 0

    for epoch in range(1, config.epochs + 1):
        _t0 = timeit.default_timer()
        train_m = _epoch(bundle.train_dyg_trials, train=True)
        _t1 = timeit.default_timer()
        val_m = _epoch(bundle.val_dyg_trials, train=False)
        _t2 = timeit.default_timer()
        test_m = _epoch(bundle.test_dyg_trials, train=False)
        _t3 = timeit.default_timer()
        train_curve.append(train_m)
        val_curve.append(val_m)
        test_curve.append(test_m)
        if val_m["accuracy"] > best_val:
            best_val = val_m["accuracy"]
            best_idx = epoch - 1
        print("-" * 40)
        print(f"Epoch {epoch:02d}")
        print(
            f"train: trial_acc={train_m['accuracy']:.4f} win_acc={train_m['window_accuracy']:.4f} ({_t1-_t0:.1f}s)"
        )
        print(
            f"val:   trial_acc={val_m['accuracy']:.4f} win_acc={val_m['window_accuracy']:.4f} ({_t2-_t1:.1f}s)"
        )
        print(
            f"test:  trial_acc={test_m['accuracy']:.4f} win_acc={test_m['window_accuracy']:.4f} ({_t3-_t2:.1f}s)"
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
