"""Minimal record loader for SEED per-window pickles (no TopoNetX extraction deps)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WindowComplexRecord:
    subject: int
    trial: int
    emotion: str
    window_idx: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    method: str
    threshold: float
    edges: tuple[tuple[int, int], ...]
    edge_weights: tuple[float, ...]
    edge_corr: tuple[float, ...]
    triangles: tuple[tuple[int, int, int], ...]
    triangle_weights: tuple[float, ...]
    node_features: tuple[tuple[float, ...], ...]
    edge_features: tuple[tuple[float, ...], ...]
    triangle_features: tuple[tuple[float, ...], ...]
    # Present only in the multimodal extract: one eye-tracking vector per window
    # (window-level, not per cell). Defaults to empty so the EEG-only extracts,
    # whose pickles have no such key, keep loading unchanged.
    eye_features: tuple[float, ...] = ()


def _empty_weight_fields() -> dict:
    return {
        "edge_weights": (),
        "edge_corr": (),
        "triangle_weights": (),
        "node_features": (),
        "edge_features": (),
        "triangle_features": (),
    }


_KNOWN_FIELDS = set(WindowComplexRecord.__dataclass_fields__)


def _record_from_dict(row: dict) -> WindowComplexRecord:
    if "edge_weights" not in row:
        row = {**row, **_empty_weight_fields()}
    # Drop keys this schema does not model rather than raising: extracts gain fields
    # over time and an unexpected one should not break every existing pipeline.
    unknown = set(row) - _KNOWN_FIELDS
    if unknown:
        row = {k: v for k, v in row.items() if k in _KNOWN_FIELDS}
    return WindowComplexRecord(**row)


def load_records(path: Path) -> list[WindowComplexRecord]:
    with Path(path).open("rb") as fp:
        obj = pickle.load(fp)
    if isinstance(obj, dict) and "records" in obj:
        return [_record_from_dict(row) for row in obj["records"]]
    if isinstance(obj, list):
        return [
            _record_from_dict(r.__dict__) if hasattr(r, "__dict__") else r for r in obj
        ]
    raise TypeError(f"Unsupported pickle payload in {path}")
