"""Append training metrics to CSV result files."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import EPOCH_RESULTS_CSV, RUN_SUMMARY_CSV, ensure_results_dir


EPOCH_FIELDNAMES = [
    "timestamp",
    "run_id",
    "model_name",
    "dataset_name",
    "epoch",
    "total_epochs",
    "eval_metric",
    "train_score",
    "val_score",
    "test_score",
    "is_best_val_epoch",
    "lr",
    "batch_size",
    "embedding_dim",
    "memory_dim",
    "time_dim",
    "num_neighbors",
    "seed",
    "device",
    "dry_run",
    "max_batches",
    "num_classes",
    "data_root",
]

SUMMARY_FIELDNAMES = [
    "timestamp",
    "run_id",
    "model_name",
    "dataset_name",
    "total_epochs",
    "eval_metric",
    "best_val_epoch",
    "best_val_score",
    "best_test_score",
    "final_train_score",
    "final_val_score",
    "final_test_score",
    "lr",
    "batch_size",
    "embedding_dim",
    "memory_dim",
    "time_dim",
    "num_neighbors",
    "seed",
    "device",
    "dry_run",
    "max_batches",
    "num_classes",
    "data_root",
    "epoch_results_csv",
    "run_summary_csv",
]


@dataclass
class RunContext:
    model_name: str
    dataset_name: str
    eval_metric: str
    num_classes: int
    total_epochs: int
    lr: float
    batch_size: int
    embedding_dim: int
    memory_dim: int
    time_dim: int
    num_neighbors: int
    seed: int
    device: str
    dry_run: bool
    max_batches: Optional[int]
    data_root: str
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self.run_id = f"{ts}_{self.model_name}_{self.dataset_name}_{uuid4().hex[:8]}"


def _append_rows(csv_path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    ensure_results_dir()
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _base_row(ctx: RunContext) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": ctx.run_id,
        "model_name": ctx.model_name,
        "dataset_name": ctx.dataset_name,
        "total_epochs": ctx.total_epochs,
        "eval_metric": ctx.eval_metric,
        "lr": ctx.lr,
        "batch_size": ctx.batch_size,
        "embedding_dim": ctx.embedding_dim,
        "memory_dim": ctx.memory_dim,
        "time_dim": ctx.time_dim,
        "num_neighbors": ctx.num_neighbors,
        "seed": ctx.seed,
        "device": ctx.device,
        "dry_run": ctx.dry_run,
        "max_batches": ctx.max_batches if ctx.max_batches is not None else "",
        "num_classes": ctx.num_classes,
        "data_root": ctx.data_root,
    }


def save_epoch_results(
    ctx: RunContext,
    summary: Dict[str, Any],
    csv_path: str = EPOCH_RESULTS_CSV,
) -> str:
    """Write one CSV row per epoch from training curves in `summary`."""
    train_curve = summary.get("train_curve", [])
    val_curve = summary.get("val_curve", [])
    test_curve = summary.get("test_curve", [])
    best_val_epoch = summary.get("best_val_epoch", 0)

    rows: List[Dict[str, Any]] = []
    n_epochs = max(len(train_curve), len(val_curve), len(test_curve))
    for i in range(n_epochs):
        row = _base_row(ctx)
        row.update({
            "epoch": i + 1,
            "train_score": train_curve[i] if i < len(train_curve) else "",
            "val_score": val_curve[i] if i < len(val_curve) else "",
            "test_score": test_curve[i] if i < len(test_curve) else "",
            "is_best_val_epoch": (i + 1) == best_val_epoch,
        })
        rows.append(row)

    _append_rows(csv_path, EPOCH_FIELDNAMES, rows)
    return os.path.abspath(csv_path)


def save_run_summary(
    ctx: RunContext,
    summary: Dict[str, Any],
    csv_path: str = RUN_SUMMARY_CSV,
    epoch_csv_path: str = EPOCH_RESULTS_CSV,
) -> str:
    """Write a single summary row for a completed run."""
    train_curve = summary.get("train_curve", [])
    val_curve = summary.get("val_curve", [])
    test_curve = summary.get("test_curve", [])

    row = _base_row(ctx)
    row.update({
        "best_val_epoch": summary.get("best_val_epoch", ""),
        "best_val_score": summary.get("best_val_score", ""),
        "best_test_score": summary.get("best_test_score", ""),
        "final_train_score": train_curve[-1] if train_curve else "",
        "final_val_score": val_curve[-1] if val_curve else "",
        "final_test_score": test_curve[-1] if test_curve else "",
        "epoch_results_csv": os.path.abspath(epoch_csv_path),
        "run_summary_csv": os.path.abspath(csv_path),
    })
    _append_rows(csv_path, SUMMARY_FIELDNAMES, [row])
    return os.path.abspath(csv_path)


def save_results(ctx: RunContext, summary: Dict[str, Any]) -> Dict[str, str]:
    """Persist per-epoch and summary CSV rows; return output paths."""
    epoch_path = save_epoch_results(ctx, summary)
    summary_path = save_run_summary(ctx, summary, epoch_csv_path=epoch_path)
    return {"epoch_results_csv": epoch_path, "run_summary_csv": summary_path}
