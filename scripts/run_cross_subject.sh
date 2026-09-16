#!/usr/bin/env bash
# Table III: leave-one-subject-out, 20 folds, all_band.
#   ./scripts/run_cross_subject.sh          # all 20 folds
#   ./scripts/run_cross_subject.sh 3 7      # only folds 3 and 7
source "$(dirname "$0")/common.sh"
require_band all_band
FOLDS=("$@"); [[ ${#FOLDS[@]} -eq 0 ]] && FOLDS=($(seq 1 20))
for fold in "${FOLDS[@]}"; do
  run_fold all_band subject 1 "$fold" "$BATCH_SUBJECT" --loso
done
echo; echo "Aggregate with: ./scripts/aggregate.sh"
