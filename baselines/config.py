"""Project-wide paths for temporal_HTGNN."""

from __future__ import annotations

import os

# Datasets are read from this directory; override with the DATA_ROOT env var.
DATA_ROOT: str = os.environ.get("DATA_ROOT", "./data")

# Training / evaluation CSV logs.
RESULTS_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)
EPOCH_RESULTS_CSV: str = os.path.join(RESULTS_DIR, "epoch_results.csv")
RUN_SUMMARY_CSV: str = os.path.join(RESULTS_DIR, "run_summary.csv")

# SEED-VII EEG graph data root (contains alpha/, delta/, ..., all_band/).
# Overridable by environment so the same checkout runs on a cluster node whose
# sandbox holds the extracts at a different path.
SEED_DATA_ROOT: str = os.environ.get(
    "SEED_DATA_ROOT", "./data"
)

SEED_RESULTS_DIR: str = os.path.join(RESULTS_DIR, "seed_vii")


def ensure_data_root() -> str:
    """Create the dataset root if needed and return its absolute path."""
    os.makedirs(DATA_ROOT, exist_ok=True)
    return os.path.abspath(DATA_ROOT)


def ensure_results_dir() -> str:
    """Create the results directory if needed."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.abspath(RESULTS_DIR)


def ensure_seed_results_dir() -> str:
    os.makedirs(SEED_RESULTS_DIR, exist_ok=True)
    return os.path.abspath(SEED_RESULTS_DIR)
