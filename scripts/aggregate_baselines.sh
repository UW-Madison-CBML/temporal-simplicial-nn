#!/usr/bin/env bash
# Mean and standard deviation over folds for the baselines.
source "$(dirname "$0")/common.sh"
export PYTHONPATH="${REPO_ROOT}/baselines:${PYTHONPATH:-}"
shopt -s nullglob
files=("${REPO_ROOT}"/results/baselines/*.csv)
[[ ${#files[@]} -eq 0 ]] && { echo "no baseline results yet"; exit 0; }
python "${REPO_ROOT}/baselines/aggregate_seed_folds.py" "${files[@]}"
