#!/usr/bin/env bash
# Runner for one temporal_HTGNN SEED-VII baseline run
# (one model x one band x one fold).
#
# the job scheduler flattens transferred directories to the sandbox root by basename, so
# the sandbox already looks like a temporal_HTGNN checkout: main_seed.py,
# config.py, data_loaders/, seed_models/, TGNN_models/, modules/, dyglib/, plus
# the band extract directory (e.g. all_band/) and the prebuilt htgnn_env/.
#
# Every knob arrives as an environment variable from the submit file.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-tgn}"
BAND="${BAND:-all_band}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-trial}"
# 4 for the trial split; 20 with SPLIT=subject gives leave-one-subject-out.
NFOLDS="${NFOLDS:-4}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-4}"
# Hidden widths. main_seed.py defaults all three to 100; NAVIS reads none of them
# (NAVISModel takes only num_nodes/num_classes), and memory_dim applies to tgn/dyrep.
MEMORY_DIM="${MEMORY_DIM:-100}"
TIME_DIM="${TIME_DIM:-100}"
EMBEDDING_DIM="${EMBEDDING_DIM:-100}"
NUM_NEIGHBORS="${NUM_NEIGHBORS:-10}"
EXTRA=()
# --dry_run is the only knob: main_seed.py sets max_batches=15 itself when it is
# passed, and rejects --max_batches as an unrecognized argument.
[[ "${DRY_RUN:-0}" == "1" ]] && EXTRA+=(--dry_run)

# config.py reads SEED_DATA_ROOT from the environment; the band dirs sit at the
# sandbox root, so the root itself is the data root.
export SEED_DATA_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/htgnn_env"
export PYTHONUNBUFFERED=1

echo "model=$MODEL band=$BAND fold=$FOLD/$NFOLDS split=$SPLIT epochs=$EPOCHS batch=$BATCH lr=$LR dims mem=$MEMORY_DIM time=$TIME_DIM emb=$EMBEDDING_DIM nbr=$NUM_NEIGHBORS"
echo "data_root=$SEED_DATA_ROOT"
ls -d "$BAND"/clique >/dev/null || { echo "band extract $BAND/clique missing" >&2; exit 1; }

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

GPU=-1
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" && GPU=0

mkdir -p results/seed_vii
python -u main_seed.py \
  --model "$MODEL" \
  --band "$BAND" \
  --split "$SPLIT" \
  --fold "$FOLD" \
  --n_folds "$NFOLDS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --lr "$LR" \
  --memory_dim "$MEMORY_DIM" \
  --time_dim "$TIME_DIM" \
  --embedding_dim "$EMBEDDING_DIM" \
  --num_neighbors "$NUM_NEIGHBORS" \
  --gpu "$GPU" \
  ${EXTRA[@]+"${EXTRA[@]}"}

# Rename so 168 sandboxes do not all return a file called seed_vii_results.csv.
SRC=results/seed_vii/seed_vii_results.csv
if [[ -f "$SRC" ]]; then
  TAG=""
  [[ "$EMBEDDING_DIM" != "100" || "$TIME_DIM" != "100" || "$MEMORY_DIM" != "100" ]] \
    && TAG="_d${EMBEDDING_DIM}"
  mv "$SRC" "results/seed_vii/${MODEL}_${BAND}_${SPLIT}${NFOLDS}_f${FOLD}${TAG}.csv"
fi
ls -la results/seed_vii
