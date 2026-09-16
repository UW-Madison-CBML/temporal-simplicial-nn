"""Load SEED-VII window graph records from CSV + PKL files."""

from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data_loaders.seed_vii.constants import (
    DEFAULT_METHOD,
    EMOTION_TO_IDX,
    NUM_EEG_NODES,
    NUM_SUBJECTS,
    TRIANGLE_FIELDS,
)


@dataclass(frozen=True)
class TrialKey:
    subject: int
    trial: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.subject, self.trial)


@dataclass
class WindowRecord:
    """One EEG graph window (edge-only fields used for baselines)."""

    subject: int
    trial: int
    emotion: str
    emotion_idx: int
    window_idx: int
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int
    method: str
    threshold: float
    edges: List[Tuple[int, int]]
    edge_weights: np.ndarray
    edge_corr: np.ndarray
    edge_features: np.ndarray
    node_features: np.ndarray
    global_idx: int = -1

    @property
    def trial_key(self) -> TrialKey:
        return TrialKey(self.subject, self.trial)


def _subject_tag(subject_id: int) -> str:
    return f"{subject_id:02d}"


def seed_band_dir(data_root: Path, band: str, method: str = DEFAULT_METHOD) -> Path:
    return Path(data_root) / band / method


def list_subject_ids(data_root: Path, band: str, method: str = DEFAULT_METHOD) -> List[int]:
    method_dir = seed_band_dir(data_root, band, method)
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Band directory not found: {method_dir}")
    subjects = []
    for csv_path in sorted(method_dir.glob("subject_*_index.csv")):
        subjects.append(int(csv_path.stem.split("_")[1]))
    if not subjects:
        raise FileNotFoundError(f"No subject index CSV files in {method_dir}")
    return subjects


def load_meta(data_root: Path, band: str) -> Dict[str, Any]:
    meta_path = Path(data_root) / band / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_edge_array(edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected edges Nx2, got shape {arr.shape}")
    return arr


def _as_feature_matrix(values: Sequence, name: str) -> np.ndarray:
    rows = [np.asarray(v, dtype=np.float32).reshape(-1) for v in values]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    feat_dim = rows[0].shape[0]
    for row in rows:
        if row.shape[0] != feat_dim:
            raise ValueError(f"Inconsistent {name} feature dims")
    return np.stack(rows, axis=0)


def _record_from_pkl_dict(raw: Dict[str, Any], global_idx: int) -> WindowRecord:
    emotion = str(raw["emotion"])
    if emotion not in EMOTION_TO_IDX:
        raise ValueError(f"Unknown emotion label: {emotion!r}")
    edges = [(int(u), int(v)) for u, v in raw["edges"]]
    node_features = _as_feature_matrix(raw["node_features"], "node")
    edge_features = _as_feature_matrix(raw["edge_features"], "edge")
    edge_weights = np.asarray(raw["edge_weights"], dtype=np.float32)
    edge_corr = np.asarray(raw["edge_corr"], dtype=np.float32)
    if len(edges) != len(edge_features):
        raise ValueError("edges and edge_features length mismatch")
    max_node = max(max(u, v) for u, v in edges) if edges else -1
    if max_node >= NUM_EEG_NODES:
        raise ValueError(f"Node id {max_node} exceeds NUM_EEG_NODES={NUM_EEG_NODES}")
    return WindowRecord(
        subject=int(raw["subject"]),
        trial=int(raw["trial"]),
        emotion=emotion,
        emotion_idx=EMOTION_TO_IDX[emotion],
        window_idx=int(raw["window_idx"]),
        start_sec=float(raw["start_sec"]),
        end_sec=float(raw["end_sec"]),
        start_sample=int(raw["start_sample"]),
        end_sample=int(raw["end_sample"]),
        method=str(raw["method"]),
        threshold=float(raw["threshold"]),
        edges=edges,
        edge_weights=edge_weights,
        edge_corr=edge_corr,
        edge_features=edge_features,
        node_features=node_features,
        global_idx=global_idx,
    )


def load_subject_windows(
    data_root: Path,
    band: str,
    subject_id: int,
    method: str = DEFAULT_METHOD,
    global_idx_offset: int = 0,
) -> List[WindowRecord]:
    """Load aligned CSV + PKL windows for one subject."""
    method_dir = seed_band_dir(data_root, band, method)
    tag = _subject_tag(subject_id)
    csv_path = method_dir / f"subject_{tag}_index.csv"
    pkl_path = method_dir / f"subject_{tag}_windows.pkl"
    if not csv_path.exists() or not pkl_path.exists():
        raise FileNotFoundError(f"Missing files for subject {tag} in {method_dir}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"{pkl_path}: expected dict with 'records'")
    pkl_records = payload["records"]

    if len(csv_rows) != len(pkl_records):
        raise ValueError(
            f"subject {tag}: CSV rows ({len(csv_rows)}) != PKL records ({len(pkl_records)})"
        )

    records: List[WindowRecord] = []
    for i, (row, raw) in enumerate(zip(csv_rows, pkl_records)):
        rec = _record_from_pkl_dict(raw, global_idx=global_idx_offset + i)
        ck = (rec.subject, rec.trial, rec.window_idx)
        rk = (int(row["subject"]), int(row["trial"]), int(row["window_idx"]))
        if ck != rk:
            raise ValueError(f"subject {tag} row {i}: CSV/PKL key mismatch {rk} vs {ck}")
        if row["emotion"] != rec.emotion:
            raise ValueError(f"subject {tag} row {i}: emotion mismatch")
        if int(row["n_edges"]) != len(rec.edges):
            raise ValueError(f"subject {tag} row {i}: n_edges mismatch")
        records.append(rec)
    return records


def load_all_windows(
    data_root: Path,
    band: str,
    method: str = DEFAULT_METHOD,
    subjects: Optional[Sequence[int]] = None,
) -> List[WindowRecord]:
    """Load all window records for a band (optionally subset of subjects)."""
    if subjects is None:
        subjects = list_subject_ids(data_root, band, method)
    all_records: List[WindowRecord] = []
    offset = 0
    for subject_id in sorted(subjects):
        subject_records = load_subject_windows(
            data_root=data_root,
            band=band,
            subject_id=subject_id,
            method=method,
            global_idx_offset=offset,
        )
        all_records.extend(subject_records)
        offset += len(subject_records)
    return all_records


def group_records_by_trial(records: Iterable[WindowRecord]) -> Dict[TrialKey, List[WindowRecord]]:
    grouped: Dict[TrialKey, List[WindowRecord]] = {}
    for rec in records:
        key = rec.trial_key
        grouped.setdefault(key, []).append(rec)
    for key in grouped:
        grouped[key].sort(key=lambda r: r.window_idx)
    return grouped


def unique_subjects(records: Sequence[WindowRecord]) -> List[int]:
    return sorted({r.subject for r in records})


def node_feature_dim(records: Sequence[WindowRecord]) -> int:
    if not records:
        return 1
    return int(records[0].node_features.shape[1])


def edge_feature_dim(records: Sequence[WindowRecord]) -> int:
    if not records:
        return 1
    return int(records[0].edge_features.shape[1])
