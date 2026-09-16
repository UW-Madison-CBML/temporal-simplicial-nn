#!/usr/bin/env python3
"""Aggregate per-fold SEED-VII baseline CSVs into mean +/- std per model and band.

Each baseline job writes one row to its own CSV
(results/seed_vii/<model>_<band>_<split><n_folds>_f<fold>.csv). This walks those,
groups by (model, band, split) and reports mean and sample std of accuracy,
macro precision, macro recall and macro F1 across folds -- the same summary our
own aggregate_folds.py produces, so the two are directly comparable.

    python aggregate_seed_folds.py results/seed_vii/*_trial4_f*.csv
    python aggregate_seed_folds.py --metric final results/seed_vii/*_subject20_f*.csv
    python aggregate_seed_folds.py --latex results/seed_vii/*_trial4_f*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict

METRICS = ("accuracy", "precision", "recall", "f1")
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma", "all_band"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csvs", nargs="+", help="per-fold result CSVs (globs are expanded)")
    ap.add_argument(
        "--metric", choices=("best", "final"), default="best",
        help="'best' = test scores at the best-validation epoch (default); "
             "'final' = scores after the last epoch.",
    )
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular body")
    args = ap.parse_args()

    paths = [p for pat in args.csvs for p in sorted(glob.glob(pat))] or []
    if not paths:
        print("No CSVs matched", file=sys.stderr)
        return 1

    prefix = "best_test_" if args.metric == "best" else "test_"
    # (model, band, split) -> metric -> [per-fold values]
    acc: dict = defaultdict(lambda: defaultdict(list))
    folds_seen: dict = defaultdict(set)

    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("model_name", "?"), row.get("band", "?"),
                       row.get("split_strategy", "?"))
                fold = row.get("fold", "?")
                if fold in folds_seen[key]:
                    continue          # a re-run of the same fold: keep the first
                folds_seen[key].add(fold)
                for m in METRICS:
                    v = row.get(prefix + m, "")
                    if v not in ("", None):
                        acc[key][m].append(float(v))

    def stats(vals):
        if not vals:
            return None
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean * 100, std * 100

    rows = sorted(acc, key=lambda k: (k[2], k[0],
                                      BAND_ORDER.index(k[1]) if k[1] in BAND_ORDER else 99))
    if args.latex:
        for (model, band, split) in rows:
            cells = []
            for m in METRICS:
                s = stats(acc[(model, band, split)][m])
                cells.append(f"{s[0]:.2f}/{s[1]:.2f}" if s else "--")
            print(f"{model} & {band} & " + " & ".join(cells) + r" \\")
        return 0

    hdr = f"{'model':<12} {'band':<9} {'split':<8} {'n':>3}  " + "".join(
        f"{m:>16}" for m in METRICS
    )
    print(hdr)
    print("-" * len(hdr))
    for (model, band, split) in rows:
        n = len(acc[(model, band, split)]["accuracy"])
        cells = []
        for m in METRICS:
            s = stats(acc[(model, band, split)][m])
            cells.append(f"{s[0]:>8.2f}+/-{s[1]:<5.2f}" if s else f"{'--':>16}")
        print(f"{model:<12} {band:<9} {split:<8} {n:>3}  " + "".join(cells))

    expected = {"trial": 4, "subject": 20}
    for key, seen in sorted(folds_seen.items()):
        want = expected.get(key[2])
        if want and len(seen) != want:
            print(f"[warn] {key[0]}/{key[1]}/{key[2]}: {len(seen)} folds, expected {want}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
