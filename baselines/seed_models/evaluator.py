"""Metrics for SEED-VII emotion classification."""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class SeedEvaluator:
    """Accuracy + macro precision / recall / F1 evaluator.

    Macro averaging throughout: on this 7-class problem it exposes collapse onto a
    single class, which accuracy alone hides.
    """

    def eval(self, input_dict: Dict) -> Dict[str, float]:
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        metrics = input_dict.get(
            "eval_metric", ["accuracy", "precision", "recall", "f1"]
        )
        y_true_idx = self._to_class_indices(y_true)
        y_pred_idx = self._to_class_indices(y_pred)
        out: Dict[str, float] = {}
        if "accuracy" in metrics:
            out["accuracy"] = float(accuracy_score(y_true_idx, y_pred_idx))
        if "precision" in metrics:
            out["precision"] = float(
                precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
            )
        if "recall" in metrics:
            out["recall"] = float(
                recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
            )
        if "f1" in metrics:
            out["f1"] = float(
                f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
            )
        return out

    @staticmethod
    def _to_class_indices(arr: Union[np.ndarray, Sequence]) -> np.ndarray:
        data = np.asarray(arr)
        if data.ndim == 2:
            return data.argmax(axis=1)
        return data.astype(np.int64)
