"""SEED-VII EEG graph dataset constants."""

from __future__ import annotations

from typing import Dict, List

# Bands available under DATA_ROOT (all_band uses 62 x 5 node features).
SEED_BANDS: List[str] = ["delta", "theta", "alpha", "beta", "gamma", "all_band"]

# Seven discrete emotion classes (SEED-VII).
EMOTION_CLASSES: List[str] = [
    "Happy",
    "Sad",
    "Neutral",
    "Disgust",
    "Fear",
    "Surprise",
    "Anger",
]

EMOTION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(EMOTION_CLASSES)}
IDX_TO_EMOTION: Dict[int, str] = {i: name for name, i in EMOTION_TO_IDX.items()}

NUM_EMOTION_CLASSES: int = len(EMOTION_CLASSES)
NUM_EEG_NODES: int = 62
NUM_SUBJECTS: int = 20
NUM_TRIALS_PER_SUBJECT: int = 80
NUM_TRIALS_TOTAL: int = NUM_SUBJECTS * NUM_TRIALS_PER_SUBJECT

DEFAULT_METHOD: str = "clique"
DEFAULT_N_FOLDS: int = 4
DEFAULT_VAL_RATIO: float = 0.15

# Triangle fields in PKL are ignored for edge-only temporal GNN baselines.
TRIANGLE_FIELDS = ("triangles", "triangle_weights", "triangle_features")
