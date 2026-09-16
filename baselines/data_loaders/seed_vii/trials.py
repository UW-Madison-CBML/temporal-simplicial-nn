"""Per-trial temporal sequences for SEED-VII.

Each trial is one sample:
  window_0 graph → window_1 graph → ... → window_T graph
  (62 nodes fixed, edges/features change every window)
  → one emotion label for the whole trial

Temporal baselines see ONLY this trial's edge dynamics, then classify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import TemporalData

from data_loaders.seed_vii.constants import NUM_EEG_NODES, NUM_EMOTION_CLASSES
from data_loaders.seed_vii.dataset import TrialKey, WindowRecord, group_records_by_trial


@dataclass
class TrialSequence:
    """One subject-trial as an ordered sequence of window graphs."""

    subject: int
    trial: int
    emotion_idx: int
    windows: List[WindowRecord]

    @property
    def trial_key(self) -> TrialKey:
        return TrialKey(self.subject, self.trial)

    @property
    def num_windows(self) -> int:
        return len(self.windows)


def records_to_trial_sequences(records: Sequence[WindowRecord]) -> List[TrialSequence]:
    """Group window records into trial sequences (sorted by window_idx)."""
    grouped = group_records_by_trial(records)
    sequences: List[TrialSequence] = []
    for key in sorted(grouped.keys(), key=lambda k: (k.subject, k.trial)):
        wins = grouped[key]
        if not wins:
            continue
        emotions = {w.emotion_idx for w in wins}
        if len(emotions) != 1:
            raise ValueError(f"Trial {key} has mixed emotions: {emotions}")
        sequences.append(
            TrialSequence(
                subject=key.subject,
                trial=key.trial,
                emotion_idx=int(next(iter(emotions))),
                windows=wins,
            )
        )
    return sequences


def feature_dims(records: Sequence[WindowRecord]) -> Tuple[int, int]:
    """Return (node_feat_dim, edge_feat_dim)."""
    if not records:
        return 1, 1
    return int(records[0].node_features.shape[1]), int(records[0].edge_features.shape[1])


def message_dim(node_feat_dim: int, edge_feat_dim: int) -> int:
    """Enriched edge message: [edge_feat || src_node_feat || dst_node_feat]."""
    return edge_feat_dim + 2 * node_feat_dim


def build_window_message(rec: WindowRecord) -> np.ndarray:
    """Build per-edge messages that carry correlation + endpoint EEG features."""
    edges = rec.edges
    if not edges:
        n_dim = int(rec.node_features.shape[1])
        e_dim = int(rec.edge_features.shape[1]) if rec.edge_features.size else 1
        return np.zeros((0, message_dim(n_dim, e_dim)), dtype=np.float32)

    node_feat = rec.node_features.astype(np.float32)
    edge_feat = rec.edge_features.astype(np.float32)
    msgs = []
    for i, (u, v) in enumerate(edges):
        msgs.append(
            np.concatenate([edge_feat[i], node_feat[u], node_feat[v]], axis=0)
        )
    return np.stack(msgs, axis=0)


def window_edge_tensors(
    rec: WindowRecord,
    time_value: int,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return src, dst, t, msg for one window (0-indexed node ids)."""
    if not rec.edges:
        raise ValueError("Window has no edges")
    src = torch.tensor([u for u, _ in rec.edges], dtype=torch.long)
    dst = torch.tensor([v for _, v in rec.edges], dtype=torch.long)
    t = torch.full((len(rec.edges),), time_value, dtype=torch.long)
    msg = torch.from_numpy(build_window_message(rec))
    if device is not None:
        src, dst, t, msg = src.to(device), dst.to(device), t.to(device), msg.to(device)
    return src, dst, t, msg


def trial_to_temporal_data(trial: TrialSequence) -> TemporalData:
    """Flatten one trial into a TemporalData stream (times = window indices)."""
    src_list, dst_list, t_list, msg_list = [], [], [], []
    for t, rec in enumerate(trial.windows):
        if not rec.edges:
            continue
        for i, (u, v) in enumerate(rec.edges):
            src_list.append(u)
            dst_list.append(v)
            t_list.append(t)
        msg_list.append(build_window_message(rec))
    if not src_list:
        raise ValueError(f"Trial ({trial.subject},{trial.trial}) has no edges")
    msg = np.concatenate(msg_list, axis=0)
    return TemporalData(
        src=torch.tensor(src_list, dtype=torch.long),
        dst=torch.tensor(dst_list, dtype=torch.long),
        t=torch.tensor(t_list, dtype=torch.long),
        msg=torch.from_numpy(msg.astype(np.float32)),
        y=torch.ones(len(src_list), dtype=torch.float32),
    )


def build_dyglib_trial_arrays(
    trial: TrialSequence,
    edge_id_offset: int = 1,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int]:
    """Build DyGLib arrays for one trial (1-indexed node ids).

    Returns (fields_dict, edge_features, next_edge_id).
    """
    src, dst, times, edge_ids = [], [], [], []
    labels, interact_types, node_label_times = [], [], []
    edge_feats: List[np.ndarray] = []
    eid = edge_id_offset
    emotion = np.zeros(NUM_EMOTION_CLASSES, dtype=np.float64)
    emotion[trial.emotion_idx] = 1.0

    for t, rec in enumerate(trial.windows):
        msgs = build_window_message(rec)
        for i, (u, v) in enumerate(rec.edges):
            src.append(u + 1)
            dst.append(v + 1)
            times.append(float(t))
            edge_ids.append(eid)
            labels.append(emotion.copy())
            node_label_times.append(float(t))
            # Supervise every edge in the window (same trial emotion).
            interact_types.append("train")
            edge_feats.append(msgs[i].astype(np.float64))
            eid += 1

    if not src:
        raise ValueError(f"Empty trial ({trial.subject},{trial.trial})")

    fields = {
        "src_node_ids": np.asarray(src, dtype=np.int64),
        "dst_node_ids": np.asarray(dst, dtype=np.int64),
        "node_interact_times": np.asarray(times, dtype=np.float64),
        "edge_ids": np.asarray(edge_ids, dtype=np.int64),
        "labels": np.stack(labels, axis=0),
        "interact_types": np.asarray(interact_types, dtype=object),
        "node_label_times": np.asarray(node_label_times, dtype=np.float64),
        "window_bounds": _window_edge_bounds(trial),
    }
    return fields, np.stack(edge_feats, axis=0), eid


def _window_edge_bounds(trial: TrialSequence) -> np.ndarray:
    """Inclusive-exclusive edge index ranges per window within the trial."""
    bounds = []
    start = 0
    for rec in trial.windows:
        end = start + len(rec.edges)
        bounds.append((start, end))
        start = end
    return np.asarray(bounds, dtype=np.int64)


def build_static_node_features(
    trials: Sequence[TrialSequence],
    node_feat_dim: int,
) -> np.ndarray:
    """(NUM_EEG_NODES+1, node_feat_dim) with row 0 padding.

    Uses mean node features over provided trials as a weak static prior.
    Per-window EEG variation is carried in enriched edge messages.
    """
    sums = np.zeros((NUM_EEG_NODES, node_feat_dim), dtype=np.float64)
    counts = np.zeros(NUM_EEG_NODES, dtype=np.float64)
    for trial in trials:
        for rec in trial.windows:
            feats = rec.node_features.astype(np.float64)
            if feats.shape[1] != node_feat_dim:
                raise ValueError("Inconsistent node feature dim")
            sums += feats
            counts += 1.0
    counts = np.maximum(counts, 1.0)
    static = (sums / counts[:, None]).astype(np.float64)
    return np.vstack([np.zeros((1, node_feat_dim), dtype=np.float64), static])
