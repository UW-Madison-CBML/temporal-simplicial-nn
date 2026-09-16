#!/usr/bin/env bash
# Table II: EEG-only control vs EEG + eye movements, trial split, 4-fold.
# Both arms use identical windows and folds; only the eye branch differs.
source "$(dirname "$0")/common.sh"
require_band all_band_eye
for arm in none window; do
  for fold in 1 2 3 4; do
    TAG_SUFFIX="_eye${arm}" \
      run_fold all_band_eye trial 4 "$fold" "$BATCH_TRIAL" --eye-fusion "$arm"
  done
done
echo; echo "Aggregate with: ./scripts/aggregate.sh"
