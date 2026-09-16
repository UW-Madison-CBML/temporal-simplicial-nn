"""Build per-trial SEED-VII training bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from config import SEED_DATA_ROOT
from data_loaders.seed_vii.constants import NUM_EEG_NODES, NUM_EMOTION_CLASSES, SEED_BANDS
from data_loaders.seed_vii.dataset import WindowRecord, load_all_windows
from data_loaders.seed_vii.splits import FoldSplit, SplitStrategy, make_fold_splits
from data_loaders.seed_vii.trials import (
    TrialSequence,
    build_dyglib_trial_arrays,
    build_static_node_features,
    feature_dims,
    message_dim,
    records_to_trial_sequences,
)
from dyglib.utils.DataLoader import Data


@dataclass
class SeedVIIBundle:
    """Per-trial temporal bundle for SEED-VII baselines."""

    band: str
    split_strategy: SplitStrategy
    fold_id: int
    num_classes: int
    num_nodes: int
    node_feat_dim: int
    edge_feat_dim: int
    msg_dim: int
    eval_metric: str
    evaluator: Any
    train_trials: List[TrialSequence]
    val_trials: List[TrialSequence]
    test_trials: List[TrialSequence]
    # DyGLib global feature tables (edge msgs indexed by global edge_id)
    node_raw_features: Optional[np.ndarray] = None
    edge_raw_features: Optional[np.ndarray] = None
    train_dyg_trials: List[dict] = field(default_factory=list)
    val_dyg_trials: List[dict] = field(default_factory=list)
    test_dyg_trials: List[dict] = field(default_factory=list)
    # Kept for logging / backward compatibility
    train_records: Optional[List[WindowRecord]] = None
    val_records: Optional[List[WindowRecord]] = None
    test_records: Optional[List[WindowRecord]] = None


def get_seed_fold_splits(
    band: str,
    split_strategy: SplitStrategy = "trial",
    data_root: str | Path = SEED_DATA_ROOT,
    n_folds: int = 4,
    seed: int = 42,
) -> List[FoldSplit]:
    records = load_all_windows(Path(data_root), band=band)
    return make_fold_splits(records, strategy=split_strategy, n_folds=n_folds, seed=seed)


def _build_dyg_split(
    trials: Sequence[TrialSequence],
    edge_id_offset: int,
) -> tuple[List[dict], np.ndarray, int]:
    packed: List[dict] = []
    feat_rows: List[np.ndarray] = []
    next_id = edge_id_offset
    for trial in trials:
        fields, edge_feats, next_id = build_dyglib_trial_arrays(trial, edge_id_offset=next_id)
        packed.append({"trial": trial, **fields})
        feat_rows.append(edge_feats)
    if feat_rows:
        edge_mat = np.concatenate(feat_rows, axis=0)
    else:
        edge_mat = np.zeros((0, 1), dtype=np.float64)
    return packed, edge_mat, next_id


def build_seed_bundle(
    fold: FoldSplit,
    band: str,
    split_strategy: SplitStrategy,
    batch_size: int = 200,
    device: torch.device | None = None,
) -> SeedVIIBundle:
    """Build trial-scoped bundle. ``batch_size`` is unused (kept for CLI compat)."""
    del batch_size, device  # per-trial loop; device handled in trainers

    train_trials = records_to_trial_sequences(fold.train_records)
    val_trials = records_to_trial_sequences(fold.val_records)
    test_trials = records_to_trial_sequences(fold.test_records)

    node_feat_dim, edge_feat_dim = feature_dims(fold.train_records)
    msg_d = message_dim(node_feat_dim, edge_feat_dim)

    node_raw = build_static_node_features(train_trials, node_feat_dim)

    train_dyg, train_edge_feats, next_id = _build_dyg_split(train_trials, edge_id_offset=1)
    val_dyg, val_edge_feats, next_id = _build_dyg_split(val_trials, edge_id_offset=next_id)
    test_dyg, test_edge_feats, _ = _build_dyg_split(test_trials, edge_id_offset=next_id)

    # Pad row 0 for DyGLib edge feature table.
    parts = [np.zeros((1, msg_d), dtype=np.float64)]
    for mat in (train_edge_feats, val_edge_feats, test_edge_feats):
        if mat.size:
            if mat.shape[1] != msg_d:
                raise ValueError(f"Edge feat dim mismatch: {mat.shape[1]} vs {msg_d}")
            parts.append(mat)
    edge_raw = np.vstack(parts)

    from seed_models.evaluator import SeedEvaluator

    return SeedVIIBundle(
        band=band,
        split_strategy=split_strategy,
        fold_id=fold.fold_id,
        num_classes=NUM_EMOTION_CLASSES,
        num_nodes=NUM_EEG_NODES,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        msg_dim=msg_d,
        eval_metric="accuracy",
        evaluator=SeedEvaluator(),
        train_trials=train_trials,
        val_trials=val_trials,
        test_trials=test_trials,
        node_raw_features=node_raw,
        edge_raw_features=edge_raw,
        train_dyg_trials=train_dyg,
        val_dyg_trials=val_dyg,
        test_dyg_trials=test_dyg,
        train_records=fold.train_records,
        val_records=fold.val_records,
        test_records=fold.test_records,
    )


def list_seed_bands() -> List[str]:
    return list(SEED_BANDS)


def dyg_trial_to_data(packed: dict) -> Data:
    """Convert a packed trial dict to DyGLib Data."""
    return Data(
        src_node_ids=packed["src_node_ids"],
        dst_node_ids=packed["dst_node_ids"],
        node_interact_times=packed["node_interact_times"],
        edge_ids=packed["edge_ids"],
        labels=packed["labels"],
        interact_types=packed["interact_types"],
        node_label_times=packed["node_label_times"],
    )
