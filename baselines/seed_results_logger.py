"""CSV logging for SEED-VII experiments (Table III format)."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import SEED_RESULTS_DIR, ensure_seed_results_dir


RUN_FIELDNAMES = [
    "timestamp",
    "run_id",
    "model_name",
    "band",
    "split_strategy",
    "fold",
    "epochs",
    "train_accuracy",
    "test_accuracy",
    "train_f1",
    "test_precision",
    "test_recall",
    "test_f1",
    "best_val_accuracy",
    "best_val_epoch",
    "best_test_accuracy",
    "best_test_precision",
    "best_test_recall",
    "best_test_f1",
    "lr",
    "batch_size",
    "embedding_dim",
    "seed",
    "device",
    "dry_run",
    "max_batches",
    "data_root",
]

TABLE_III_FIELDNAMES = [
    "split_strategy",
    "model_name",
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
    "all_band",
    "mean",
]


@dataclass
class SeedRunContext:
    model_name: str
    band: str
    split_strategy: str
    fold: int
    epochs: int
    lr: float
    batch_size: int
    embedding_dim: int
    seed: int
    device: str
    dry_run: bool
    max_batches: Optional[int]
    data_root: str
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self.run_id = f"{ts}_{self.model_name}_{self.band}_f{self.fold}_{uuid4().hex[:6]}"


def _append_rows(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> str:
    ensure_seed_results_dir()
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return os.path.abspath(path)


def save_seed_run_result(
    ctx: SeedRunContext,
    summary: Dict[str, Any],
    csv_path: str | None = None,
) -> str:
    if csv_path is None:
        csv_path = os.path.join(SEED_RESULTS_DIR, "seed_vii_results.csv")
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": ctx.run_id,
        "model_name": ctx.model_name,
        "band": ctx.band,
        "split_strategy": ctx.split_strategy,
        "fold": ctx.fold,
        "epochs": ctx.epochs,
        "train_accuracy": summary.get("final_train_accuracy", ""),
        "test_accuracy": summary.get("final_test_accuracy", ""),
        "train_f1": summary.get("final_train_f1", ""),
        "test_precision": summary.get("final_test_precision", ""),
        "test_recall": summary.get("final_test_recall", ""),
        "test_f1": summary.get("final_test_f1", ""),
        "best_val_accuracy": summary.get("best_val_accuracy", ""),
        "best_val_epoch": summary.get("best_val_epoch", ""),
        "best_test_accuracy": summary.get("best_test_accuracy", ""),
        "best_test_precision": summary.get("best_test_precision", ""),
        "best_test_recall": summary.get("best_test_recall", ""),
        "best_test_f1": summary.get("best_test_f1", ""),
        "lr": ctx.lr,
        "batch_size": ctx.batch_size,
        "embedding_dim": ctx.embedding_dim,
        "seed": ctx.seed,
        "device": ctx.device,
        "dry_run": ctx.dry_run,
        "max_batches": ctx.max_batches if ctx.max_batches is not None else "",
        "data_root": ctx.data_root,
    }
    return _append_rows(csv_path, RUN_FIELDNAMES, [row])


def build_table_iii(
    results_csv: str | None = None,
    output_csv: str | None = None,
    split_strategy: str = "trial",
    metric: str = "test_accuracy",
) -> str:
    """Aggregate per-band test accuracy into Table III layout."""
    if results_csv is None:
        results_csv = os.path.join(SEED_RESULTS_DIR, "seed_vii_results.csv")
    if output_csv is None:
        output_csv = os.path.join(SEED_RESULTS_DIR, f"seed_vii_table_iii_{split_strategy}.csv")
    if not os.path.isfile(results_csv):
        raise FileNotFoundError(f"No results file: {results_csv}")

    # model -> band -> list of metric values across folds
    scores: Dict[str, Dict[str, List[float]]] = {}
    with open(results_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split_strategy") != split_strategy:
                continue
            model = row["model_name"]
            band = row["band"]
            val = row.get(metric, "")
            if val == "":
                continue
            scores.setdefault(model, {}).setdefault(band, []).append(float(val))

    band_order = ["delta", "theta", "alpha", "beta", "gamma", "all_band"]
    rows: List[Dict[str, Any]] = []
    for model in sorted(scores):
        row: Dict[str, Any] = {"split_strategy": split_strategy, "model_name": model}
        band_vals = []
        for band in band_order:
            vals = scores[model].get(band, [])
            mean_val = sum(vals) / len(vals) if vals else ""
            row[band] = f"{mean_val:.4f}" if mean_val != "" else ""
            if mean_val != "":
                band_vals.append(mean_val)
        row["mean"] = f"{sum(band_vals) / len(band_vals):.4f}" if band_vals else ""
        rows.append(row)

    ensure_seed_results_dir()
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_III_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return os.path.abspath(output_csv)
