#!/usr/bin/env bash
# Table I: per-band results, trial split, 4-fold cross-validation.
#   ./scripts/run_band_ablation.sh                 # all six bands
#   ./scripts/run_band_ablation.sh alpha theta     # selected bands
source "$(dirname "$0")/common.sh"
BANDS=("$@"); [[ ${#BANDS[@]} -eq 0 ]] && BANDS=(delta theta alpha beta gamma all_band)
for band in "${BANDS[@]}"; do
  require_band "$band"
  for fold in 1 2 3 4; do run_fold "$band" trial 4 "$fold" "$BATCH_TRIAL"; done
done
echo; echo "Aggregate with: ./scripts/aggregate.sh"
