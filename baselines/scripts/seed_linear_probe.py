#!/usr/bin/env python3
"""
Window-level linear probe for SEED-VII.

This is a sanity-check baseline (NOT a temporal GNN):
  each EEG window -> flatten node features -> LogisticRegression -> emotion class

It answers: "If we use the raw per-window EEG features correctly,
do we beat chance (~14%)?" If yes, the data is usable and low TGNN
accuracy is likely a pipeline/protocol bug.

Examples:
  python scripts/seed_linear_probe.py --band alpha
  python scripts/seed_linear_probe.py --band all_band --fold 0
  python scripts/seed_linear_probe.py --run_all_bands --split trial
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED_DATA_ROOT
from data_loaders.seed_vii.bundle import get_seed_fold_splits, list_seed_bands
from data_loaders.seed_vii.constants import EMOTION_CLASSES, NUM_EMOTION_CLASSES
from data_loaders.seed_vii.dataset import WindowRecord


def _features(
    records: List[WindowRecord],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) where each row is one window."""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for rec in records:
        if mode == "node_flat":
            # Main probe: all channel features concatenated.
            # single band: (62,) ; all_band: (62*5,)
            xs.append(rec.node_features.reshape(-1).astype(np.float32))
        elif mode == "node_mean":
            xs.append(rec.node_features.mean(axis=0).astype(np.float32))
        elif mode == "edge_mean":
            if len(rec.edge_features) == 0:
                xs.append(np.zeros(1, dtype=np.float32))
            else:
                xs.append(rec.edge_features.mean(axis=0).astype(np.float32))
        else:
            raise ValueError(f"Unknown feature mode: {mode}")
        ys.append(int(rec.emotion_idx))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


def run_probe(
    band: str,
    split: str,
    fold_id: int,
    mode: str,
    max_train: int,
    seed: int,
) -> None:
    folds = get_seed_fold_splits(
        band=band,
        split_strategy=split,
        data_root=SEED_DATA_ROOT,
        seed=seed,
    )
    fold = folds[fold_id]
    rng = np.random.RandomState(seed)

    X_train, y_train = _features(fold.train_records, mode)
    X_val, y_val = _features(fold.val_records, mode)
    X_test, y_test = _features(fold.test_records, mode)

    if max_train > 0 and len(X_train) > max_train:
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_fit, y_fit = X_train[idx], y_train[idx]
    else:
        X_fit, y_fit = X_train, y_train

    # StandardScaler + multinomial logistic regression = linear probe
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )
    clf.fit(X_fit, y_fit)

    def _eval(split_name: str, X: np.ndarray, y: np.ndarray) -> None:
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        print(f"  {split_name:5s}  acc={acc:.4f}  macro_f1={f1:.4f}  n={len(y)}")

    maj = Counter(y_fit.tolist()).most_common(1)[0][0]
    maj_acc = float((y_test == maj).mean())
    chance = 1.0 / NUM_EMOTION_CLASSES

    print("=" * 60)
    print(f"Linear probe | band={band} split={split} fold={fold_id} mode={mode}")
    print(f"Feature dim = {X_fit.shape[1]}  |  fit on {len(X_fit)} / {len(X_train)} train windows")
    print(f"Classes     = {EMOTION_CLASSES}")
    print(f"Chance      = {chance:.4f}  |  majority(test) = {maj_acc:.4f} ({EMOTION_CLASSES[maj]})")
    _eval("train", X_fit, y_fit)
    _eval("val", X_val, y_val)
    _eval("test", X_test, y_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEED-VII window-level linear probe")
    p.add_argument("--band", type=str, default="alpha", choices=list_seed_bands())
    p.add_argument("--split", type=str, default="trial", choices=["trial", "subject"])
    p.add_argument("--fold", type=int, default=0, help="Fold id 0-3")
    p.add_argument(
        "--mode",
        type=str,
        default="node_flat",
        choices=["node_flat", "node_mean", "edge_mean"],
        help="node_flat = main sanity probe (recommended)",
    )
    p.add_argument(
        "--max_train",
        type=int,
        default=15000,
        help="Subsample train windows for speed (0 = use all)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_all_bands", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bands = list_seed_bands() if args.run_all_bands else [args.band]
    for band in bands:
        run_probe(
            band=band,
            split=args.split,
            fold_id=args.fold,
            mode=args.mode,
            max_train=args.max_train,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
