#!/usr/bin/env bash
# Mean and standard deviation over folds for every completed configuration.
source "$(dirname "$0")/common.sh"
shopt -s nullglob
declare -A groups
for f in "${RESULTS_DIR}"/metrics_*.json; do
  key="$(basename "$f" | sed -E 's/_f[0-9]+\.json$//')"
  groups["$key"]=1
done
[[ ${#groups[@]} -eq 0 ]] && { echo "no results in ${RESULTS_DIR}"; exit 0; }
for key in $(printf '%s\n' "${!groups[@]}" | sort); do
  echo "--- ${key#metrics_}"
  python "${REPO_ROOT}/model/aggregate_folds.py" "${RESULTS_DIR}/${key}"_f*.json
done
