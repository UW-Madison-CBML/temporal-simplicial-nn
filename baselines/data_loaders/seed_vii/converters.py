"""Convert SEED-VII window records to PyG / DyGLib temporal formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from data_loaders.seed_vii.constants import NUM_EEG_NODES, NUM_EMOTION_CLASSES
from data_loaders.seed_vii.dataset import WindowRecord, edge_feature_dim, node_feature_dim
from dyglib.utils.DataLoader import Data, get_idx_data_loader
from dyglib.utils.utils import get_neighbor_sampler


@dataclass
class SeedLabelDataset:
    """TGB-compatible label interface for PyG temporal trainers."""

    label_dict: Dict[int, Dict[int, np.ndarray]]
    label_ts: np.ndarray
    label_ts_idx: int = 0
    eval_metric: str = "accuracy"
    num_classes: int = NUM_EMOTION_CLASSES

    def reset_label_time(self) -> None:
        self.label_ts_idx = 0

    def get_label_time(self) -> int:
        if self.label_ts_idx >= len(self.label_ts):
            return int(self.label_ts[-1]) if len(self.label_ts) else 0
        return int(self.label_ts[self.label_ts_idx])

    def find_next_labels_batch(self, cur_t: int):
        if self.label_ts_idx >= len(self.label_ts):
            return None
        ts = int(self.label_ts[self.label_ts_idx])
        if cur_t < ts:
            return None
        self.label_ts_idx += 1
        node_ids = np.arange(NUM_EEG_NODES, dtype=np.int64)
        labels = np.stack(
            [self.label_dict[ts][int(nid)] for nid in node_ids],
            axis=0,
        )
        label_ts = np.full(node_ids.shape[0], ts, dtype=np.int64)
        return label_ts, node_ids, labels


def _one_hot_label(class_idx: int, num_classes: int = NUM_EMOTION_CLASSES) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[class_idx] = 1.0
    return vec


def _compute_static_node_features(records: Sequence[WindowRecord]) -> np.ndarray:
    n_feat = node_feature_dim(records)
    sums = np.zeros((NUM_EEG_NODES, n_feat), dtype=np.float64)
    counts = np.zeros(NUM_EEG_NODES, dtype=np.float64)
    for rec in records:
        for node_id in range(rec.node_features.shape[0]):
            sums[node_id] += rec.node_features[node_id]
            counts[node_id] += 1.0
    counts = np.maximum(counts, 1.0)
    return (sums / counts[:, None]).astype(np.float64)


def build_temporal_arrays(
    records: Sequence[WindowRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[int, np.ndarray]]]:
    src_list: List[int] = []
    dst_list: List[int] = []
    t_list: List[int] = []
    msg_list: List[np.ndarray] = []
    edge_label_list: List[float] = []
    label_dict: Dict[int, Dict[int, np.ndarray]] = {}

    global_t = 0
    for rec in records:
        one_hot = _one_hot_label(rec.emotion_idx)
        label_dict[global_t] = {node_id: one_hot.copy() for node_id in range(NUM_EEG_NODES)}
        for edge_idx, (u, v) in enumerate(rec.edges):
            src_list.append(u)
            dst_list.append(v)
            t_list.append(global_t)
            msg_list.append(rec.edge_features[edge_idx].astype(np.float32))
            edge_label_list.append(1.0)
        global_t += 1

    if not src_list:
        raise ValueError("No edges found while building temporal arrays")

    return (
        np.asarray(src_list, dtype=np.int64),
        np.asarray(dst_list, dtype=np.int64),
        np.asarray(t_list, dtype=np.int64),
        np.stack(msg_list, axis=0),
        np.asarray(edge_label_list, dtype=np.float32),
        label_dict,
    )


def records_to_temporal_data(records: Sequence[WindowRecord]) -> TemporalData:
    src, dst, t, msg, edge_label, _ = build_temporal_arrays(records)
    return TemporalData(
        src=torch.from_numpy(src),
        dst=torch.from_numpy(dst),
        t=torch.from_numpy(t),
        msg=torch.from_numpy(msg),
        y=torch.from_numpy(edge_label),
    )


def build_label_dataset(records: Sequence[WindowRecord]) -> SeedLabelDataset:
    _, _, _, _, _, label_dict = build_temporal_arrays(records)
    label_ts = np.sort(np.array(list(label_dict.keys()), dtype=np.int64))
    return SeedLabelDataset(label_dict=label_dict, label_ts=label_ts)


def _build_dyglib_data(
    records: Sequence[WindowRecord],
    interact_type: str,
    edge_id_offset: int,
) -> Tuple[Data, np.ndarray, int]:
    src_list: List[int] = []
    dst_list: List[int] = []
    t_list: List[float] = []
    edge_ids: List[int] = []
    labels_list: List[np.ndarray] = []
    interact_types: List[str] = []
    node_label_times: List[float] = []
    edge_feats: List[np.ndarray] = []

    edge_counter = edge_id_offset
    global_t = 0
    for rec in records:
        one_hot = _one_hot_label(rec.emotion_idx).astype(np.float64)
        for edge_idx, (u, v) in enumerate(rec.edges):
            src_list.append(u + 1)
            dst_list.append(v + 1)
            t_list.append(float(global_t))
            edge_ids.append(edge_counter)
            labels_list.append(one_hot)
            node_label_times.append(float(global_t))
            edge_feats.append(rec.edge_features[edge_idx].astype(np.float64))
            interact_types.append("just_update")
            edge_counter += 1
        if interact_types:
            interact_types[-1] = interact_type
        global_t += 1

    if not src_list:
        empty = Data(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.zeros((0, NUM_EMOTION_CLASSES)),
            np.array([], dtype=object),
            np.array([], dtype=np.float64),
        )
        return empty, np.zeros((0, edge_feature_dim(records) if records else 1)), edge_counter

    return (
        Data(
            np.asarray(src_list, dtype=np.int64),
            np.asarray(dst_list, dtype=np.int64),
            np.asarray(t_list, dtype=np.float64),
            np.asarray(edge_ids, dtype=np.int64),
            np.stack(labels_list, axis=0),
            np.asarray(interact_types, dtype=object),
            np.asarray(node_label_times, dtype=np.float64),
        ),
        np.stack(edge_feats, axis=0),
        edge_counter,
    )


def build_dyglib_split_data(
    train_records: Sequence[WindowRecord],
    val_records: Sequence[WindowRecord],
    test_records: Sequence[WindowRecord],
) -> Tuple[np.ndarray, np.ndarray, Data, Data, Data, Data]:
    static_nodes = _compute_static_node_features(train_records)
    node_raw_features = np.vstack(
        [np.zeros(static_nodes.shape[1]), static_nodes]
    ).astype(np.float64)

    train_data, train_edge_feats, next_id = _build_dyglib_data(train_records, "train", edge_id_offset=1)
    val_data, val_edge_feats, next_id = _build_dyglib_data(val_records, "validate", edge_id_offset=next_id)
    test_data, test_edge_feats, _ = _build_dyglib_data(test_records, "test", edge_id_offset=next_id)

    edge_raw_features = np.vstack(
        [np.zeros((1, train_edge_feats.shape[1] if train_edge_feats.size else 1)), train_edge_feats, val_edge_feats, test_edge_feats]
    ).astype(np.float64)

    full_data = Data(
        np.concatenate([train_data.src_node_ids, val_data.src_node_ids, test_data.src_node_ids]),
        np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids, test_data.dst_node_ids]),
        np.concatenate([train_data.node_interact_times, val_data.node_interact_times, test_data.node_interact_times]),
        np.concatenate([train_data.edge_ids, val_data.edge_ids, test_data.edge_ids]),
        np.vstack([train_data.labels, val_data.labels, test_data.labels]) if len(train_data.labels) + len(val_data.labels) + len(test_data.labels) else np.zeros((0, NUM_EMOTION_CLASSES)),
        np.concatenate([train_data.interact_types, val_data.interact_types, test_data.interact_types]),
        np.concatenate([train_data.node_label_times, val_data.node_label_times, test_data.node_label_times]),
    )
    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data


def build_temporal_loaders(
    train_data: TemporalData,
    val_data: TemporalData,
    test_data: TemporalData,
    batch_size: int,
) -> Tuple[TemporalDataLoader, TemporalDataLoader, TemporalDataLoader]:
    return (
        TemporalDataLoader(train_data, batch_size=batch_size, shuffle=False),
        TemporalDataLoader(val_data, batch_size=batch_size, shuffle=False),
        TemporalDataLoader(test_data, batch_size=batch_size, shuffle=False),
    )


def build_dyglib_loaders(
    train_data: Data,
    val_data: Data,
    test_data: Data,
    batch_size: int,
) -> Tuple[Any, Any, Any, Any, Any]:
    full_data = Data(
        np.concatenate([train_data.src_node_ids, val_data.src_node_ids, test_data.src_node_ids]),
        np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids, test_data.dst_node_ids]),
        np.concatenate([train_data.node_interact_times, val_data.node_interact_times, test_data.node_interact_times]),
        np.concatenate([train_data.edge_ids, val_data.edge_ids, test_data.edge_ids]),
        np.vstack([train_data.labels, val_data.labels, test_data.labels]),
        np.concatenate([train_data.interact_types, val_data.interact_types, test_data.interact_types]),
        np.concatenate([train_data.node_label_times, val_data.node_label_times, test_data.node_label_times]),
    )
    train_neighbor_sampler = get_neighbor_sampler(train_data, sample_neighbor_strategy="recent")
    full_neighbor_sampler = get_neighbor_sampler(full_data, sample_neighbor_strategy="recent")
    train_idx_loader = get_idx_data_loader(
        list(range(len(train_data.src_node_ids))), batch_size=batch_size, shuffle=False
    )
    val_idx_loader = get_idx_data_loader(
        list(range(len(val_data.src_node_ids))), batch_size=batch_size, shuffle=False
    )
    test_idx_loader = get_idx_data_loader(
        list(range(len(test_data.src_node_ids))), batch_size=batch_size, shuffle=False
    )
    return train_neighbor_sampler, full_neighbor_sampler, train_idx_loader, val_idx_loader, test_idx_loader
