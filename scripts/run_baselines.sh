#!/usr/bin/env bash
# Temporal-GNN baselines (Table I), trial split, 4-fold cross-validation.
#
#   ./scripts/run_baselines.sh                        # tgn, jodie, graphmixer x 6 bands
#   ./scripts/run_baselines.sh --models tgn           # one model
#   ./scripts/run_baselines.sh --bands all_band       # one band
#   ./scripts/run_baselines.sh --models tgn --folds 0 --epochs 3   # quick check
source "$(dirname "$0")/common.sh"

MODELS="tgn jodie graphmixer"
BANDS="delta theta alpha beta gamma all_band"
FOLDS="0 1 2 3"
SPLIT="trial"; NFOLDS=4
BASE_LR="${BASE_LR:-1e-4}"
BASE_EPOCHS="${EPOCHS:-100}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="${2//,/ }"; shift 2 ;;
    --bands)  BANDS="${2//,/ }";  shift 2 ;;
    --folds)  FOLDS="${2//,/ }";  shift 2 ;;
    --epochs) BASE_EPOCHS="$2";   shift 2 ;;
    --lr)     BASE_LR="$2";       shift 2 ;;
    --split)  SPLIT="$2"; [[ "$2" == subject ]] && NFOLDS=20; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

BASE_OUT="${REPO_ROOT}/results/baselines"
mkdir -p "$BASE_OUT"
export PYTHONPATH="${REPO_ROOT}/baselines:${PYTHONPATH:-}"
export SEED_DATA_ROOT="$DATA_ROOT"
GPU=-1; [[ "$DEVICE" == "cuda" ]] && GPU=0

for model in $MODELS; do
  for band in $BANDS; do
    require_band "$band"
    for fold in $FOLDS; do
      echo "=== ${model} ${band} fold ${fold} ==="
      ( cd "${REPO_ROOT}/baselines" && python -u main_seed.py \
          --model "$model" --band "$band" --split "$SPLIT" \
          --fold "$fold" --n_folds "$NFOLDS" \
          --epochs "$BASE_EPOCHS" --lr "$BASE_LR" --gpu "$GPU" )
      src="${REPO_ROOT}/baselines/results/seed_vii/seed_vii_results.csv"
      [[ -f "$src" ]] && mv "$src" "${BASE_OUT}/${model}_${band}_${SPLIT}${NFOLDS}_f${fold}.csv"
    done
  done
done
echo; echo "Aggregate with: ./scripts/aggregate_baselines.sh"
