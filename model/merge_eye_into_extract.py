#!/usr/bin/env python3
"""Attach the multimodal extract's eye features to another extract's EEG windows.

Motivation: the ``multimodal`` extract's EEG node features sit on a very different
scale from ``all_band`` (mean ~19.6 vs ~1.5), which drives the sigmoid-activated SCCN
layers into saturation and leaves the model underfitting (train_acc ~0.37 after 100
epochs, against ~0.99 on all_band). That makes any eye-vs-no-eye comparison on the
multimodal extract unreliable. Pairing the eye vectors with all_band's well-scaled EEG
features isolates the eye contribution on an EEG pathway that demonstrably trains.

Windowing differs between the two extracts -- all_band uses 8 s windows at 4 s stride,
multimodal uses 4 s non-overlapping -- so eye vectors are re-aligned by **averaging the
eye windows that overlap each target window in time**. That is the same rule the
original extraction used to align MAET's 4 s eye windows onto EEG windows
(``eye_alignment`` in the band metadata), applied at the new window length.

    python scripts/merge_eye_into_extract.py \
        --eeg-root data/all_band --eye-root data/multimodal --out-root data/all_band_eye
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    obj = pickle.loads(path.read_bytes())
    rows = obj["records"] if isinstance(obj, dict) and "records" in obj else obj
    return [r if isinstance(r, dict) else dict(r.__dict__) for r in rows]


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the temporal overlap between ``[a0,a1]`` and ``[b0,b1]``."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def merge_subject(eeg_path: Path, eye_path: Path) -> tuple[list[dict], dict]:
    eeg_rows = load_rows(eeg_path)
    eye_rows = load_rows(eye_path)

    # eye windows grouped by trial, so each target window only scans its own trial
    by_trial: dict[int, list[dict]] = defaultdict(list)
    for r in eye_rows:
        if r.get("eye_features"):
            by_trial[r["trial"]].append(r)

    stats = {"windows": 0, "matched": 0, "no_overlap": 0, "sources": 0}
    out = []
    for r in eeg_rows:
        stats["windows"] += 1
        cands = by_trial.get(r["trial"], [])
        hits = [
            e for e in cands
            if overlap(r["start_sec"], r["end_sec"], e["start_sec"], e["end_sec"]) > 0
        ]
        if hits:
            dim = len(hits[0]["eye_features"])
            mean = [sum(e["eye_features"][i] for e in hits) / len(hits) for i in range(dim)]
            stats["matched"] += 1
            stats["sources"] += len(hits)
        else:
            # keep the window but mark it: a silently-zeroed eye vector would look like
            # a real measurement of "no eye activity"
            stats["no_overlap"] += 1
            mean = []
        out.append({**r, "eye_features": tuple(mean)})
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-root", type=Path, default=Path("data/all_band"))
    ap.add_argument("--eye-root", type=Path, default=Path("data/multimodal"))
    ap.add_argument("--out-root", type=Path, default=Path("data/all_band_eye"))
    ap.add_argument("--method", default="clique")
    args = ap.parse_args()

    eeg_dir = args.eeg_root / args.method
    eye_dir = args.eye_root / args.method
    out_dir = args.out_root / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg_files = sorted(eeg_dir.glob("subject_*_windows.pkl"))
    if not eeg_files:
        print(f"No pickles under {eeg_dir}", file=sys.stderr)
        return 1

    totals = {"windows": 0, "matched": 0, "no_overlap": 0, "sources": 0}
    for f in eeg_files:
        eye_f = eye_dir / f.name
        if not eye_f.is_file():
            print(f"  {f.name}: no eye counterpart, SKIPPED", file=sys.stderr)
            continue
        rows, st = merge_subject(f, eye_f)
        (out_dir / f.name).write_bytes(pickle.dumps({"records": rows}))
        # copy the index csv across so the extract looks like any other
        idx = f.with_name(f.name.replace("_windows.pkl", "_index.csv"))
        if idx.is_file():
            (out_dir / idx.name).write_bytes(idx.read_bytes())
        for k in totals:
            totals[k] += st[k]
        print(
            f"  {f.name}: {st['windows']} windows, {st['matched']} with eye "
            f"({st['sources'] / max(st['matched'],1):.2f} eye windows averaged each)"
            + (f", {st['no_overlap']} WITHOUT" if st["no_overlap"] else ""),
            flush=True,
        )

    meta = {
        "derived_from": {"eeg": str(args.eeg_root), "eye": str(args.eye_root)},
        "features": "EEG: same as the eeg-root extract. eye: mean of eye windows "
        "overlapping each EEG window (same rule as the original MAET alignment).",
        "windows_total": totals["windows"],
        "windows_with_eye": totals["matched"],
        "windows_without_eye": totals["no_overlap"],
        "mean_eye_windows_averaged": totals["sources"] / max(totals["matched"], 1),
    }
    (args.out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  total {totals['windows']} windows, {totals['matched']} with eye "
          f"({100*totals['matched']/max(totals['windows'],1):.1f}%), "
          f"{totals['no_overlap']} without")
    print(f"  wrote {out_dir}")
    if totals["no_overlap"]:
        print("  [warn] some windows had no overlapping eye window; those carry an "
              "empty eye vector and the model will treat the extract as EEG-only",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
