#!/usr/bin/env python3
"""Combine per-fold metric JSONs into one cross-validation summary.

Use when each fold was run as its own job (``--fold K --metrics-out ...``)
rather than all folds sequentially in a single job -- K folds in parallel takes one
fold's wall time instead of K times it.

    python scripts/aggregate_folds.py results/phase_subject_fold*.json

Only the standard library is used, so this runs anywhere (no torch needed).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

METRICS = ("accuracy", "precision", "recall", "f1")


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        try:
            blob = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] skipping {p}: {exc}", file=sys.stderr)
            continue
        for fold in blob.get("folds", []):
            # Remember which file a fold came from, so a duplicate or missing fold is
            # traceable back to its job.
            rows.append({**fold, "_src": Path(p).name})
    return rows


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    paths = [Path(a) for a in argv]
    rows = load(paths)
    if not rows:
        print("No fold results found.", file=sys.stderr)
        return 1

    names = [r.get("fold", "?") for r in rows]
    if len(set(names)) != len(names):
        print(f"[warn] duplicate fold names: {sorted(names)}", file=sys.stderr)

    width = max(len(str(r.get("fold", "?"))) for r in rows) + 2
    print(f"{'fold':<{width}} {'epoch':>5}  " + "  ".join(f"{m:>9}" for m in METRICS))
    print("-" * (width + 8 + 11 * len(METRICS)))
    for r in sorted(rows, key=lambda x: str(x.get("fold"))):
        print(
            f"{str(r.get('fold','?')):<{width}} {r.get('epoch',0):>5}  "
            + "  ".join(f"{r.get(m, float('nan')):>9.4f}" for m in METRICS)
        )
    print("-" * (width + 8 + 11 * len(METRICS)))

    for label, fn in (
        ("MEAN", statistics.mean),
        ("STD", lambda v: statistics.stdev(v) if len(v) > 1 else 0.0),
    ):
        print(
            f"{label:<{width}} {'':>5}  "
            + "  ".join(f"{fn([r[m] for r in rows]):>9.4f}" for m in METRICS)
        )

    print(f"\n{len(rows)} folds")
    for m in METRICS:
        vals = [r[m] for r in rows]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {m:<10} = {statistics.mean(vals):.4f} +/- {std:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
