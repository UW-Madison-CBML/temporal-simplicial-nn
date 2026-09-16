# Shared settings for all experiment scripts. Source, do not execute.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/model:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/ours}"
DEVICE="${DEVICE:-auto}"          # auto | cuda | cpu
EPOCHS="${EPOCHS:-100}"
CHANNELS="${CHANNELS:-512}"
LR="${LR:-3e-4}"

if [[ "$DEVICE" == "auto" ]]; then
  # Tolerate a missing/!importable torch: aggregation scripts source this file and
  # do not need it, so fall back to cpu instead of aborting with a traceback.
  DEVICE=$(python -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)
fi

# Trials per optimizer step, and the cap on trials per forward pass. MICRO_BATCH
# bounds activation memory without changing the optimizer step (sub-chunk losses
# are re-weighted to the batch mean). Lower it if you hit out-of-memory.
BATCH_TRIAL="${BATCH_TRIAL:-64}"
BATCH_SUBJECT="${BATCH_SUBJECT:-256}"
MICRO_BATCH="${MICRO_BATCH:-64}"

require_band () {
  local band="$1"
  if [[ ! -d "${DATA_ROOT}/${band}/clique" ]]; then
    echo "error: missing ${DATA_ROOT}/${band}/clique" >&2
    echo "       run ./scripts/download_data.sh first" >&2
    exit 1
  fi
}

# all_band carries 5 band-power values per channel; single bands carry 1.
feat_dim_for () { [[ "$1" == all_band* ]] && echo 5 || echo 1; }

# TAG_SUFFIX distinguishes runs that share band/split/fold but differ in a flag
# (e.g. the two eye-fusion arms); without it the second run overwrites the first.
run_fold () {                     # run_fold <band> <split> <folds> <fold> <batch> [extra args...]
  local band="$1" split="$2" nfolds="$3" fold="$4" batch="$5"; shift 5
  local tag="${band}_${split}${TAG_SUFFIX:-}_f${fold}"
  echo "=== ${tag} ==="
  python -u "${REPO_ROOT}/model/train_sccn_temporal_seed.py" \
    --data-root "${DATA_ROOT}/${band}" --method clique \
    --split-mode "$split" --n-folds "$nfolds" --fold "$fold" \
    --epochs "$EPOCHS" --channels "$CHANNELS" --lr "$LR" \
    --feat-dim "$(feat_dim_for "$band")" \
    --variant sist --temporal-phase \
    --dropout 0.0 --weight-decay 0.0 --grad-clip 5.0 \
    --batch-size "$batch" --micro-batch "$MICRO_BATCH" --batched-forward \
    --select-metric val_f1 --device "$DEVICE" \
    --metrics-out "${RESULTS_DIR}/metrics_${tag}.json" "$@"
}
