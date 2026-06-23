#!/bin/bash
#SBATCH --qos=normal
#SBATCH --account=f202603931cpcaa0g
#SBATCH --job-name=gnn_full_cmp
#SBATCH --output=logs/gnn_full_cmp_%j.out
#SBATCH --error=logs/gnn_full_cmp_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

NODE_FEATURE_MODE="${1:?usage: sbatch scripts/deucalion_train_full_and_compare.sh NODE_FEATURE_MODE EDGE_FEATURE_MODE RUN_NAME [DATASET_NAME] [DATASET_LABEL]}"
EDGE_FEATURE_MODE="${2:?usage: sbatch scripts/deucalion_train_full_and_compare.sh NODE_FEATURE_MODE EDGE_FEATURE_MODE RUN_NAME [DATASET_NAME] [DATASET_LABEL]}"
RUN_NAME="${3:?usage: sbatch scripts/deucalion_train_full_and_compare.sh NODE_FEATURE_MODE EDGE_FEATURE_MODE RUN_NAME [DATASET_NAME] [DATASET_LABEL]}"
DATASET_NAME="${4:-run5_expD_all}"
DATASET_LABEL="${5:-expD_dataset}"
EXTRA_ARGS=("${@:6}")

REPO="/projects/F202603931CPCAA0/goncalo/tetra4"
DATASET_DIR="synth_dataset/runs/${DATASET_NAME}"
SPLIT_FILE="GNN/split/runs/${DATASET_NAME}/guide_split_seed12345.npz"
RUNS_ROOT="GNN/runs/${DATASET_LABEL}/magnitude_as_is/full_${DATASET_NAME}/Deucalion_runs"
IMAGE_ROOT="imgs_extras/imgs_teste"
GNN_BRIGHTEST_STARS="${GNN_BRIGHTEST_STARS:-8}"

cd "$REPO"
mkdir -p logs "$RUNS_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source .venv-gnn/bin/activate

echo "HOST=$(hostname)"
echo "DATASET_DIR=$DATASET_DIR"
echo "SPLIT_FILE=$SPLIT_FILE"
echo "RUNS_ROOT=$RUNS_ROOT"
echo "RUN_NAME=$RUN_NAME"
echo "NODE_FEATURE_MODE=$NODE_FEATURE_MODE"
echo "EDGE_FEATURE_MODE=$EDGE_FEATURE_MODE"
echo "GNN_BRIGHTEST_STARS=$GNN_BRIGHTEST_STARS"
nvidia-smi || true

python -u -m GNN.GNN \
  --dataset-dir "$DATASET_DIR" \
  --split-file "$SPLIT_FILE" \
  --runs-root "$RUNS_ROOT" \
  --run-name "$RUN_NAME" \
  --epochs 80 \
  --batch-size-scenes 2048 \
  --num-workers 4 \
  --cache-chunks 12 \
  --log-every-batches 20 \
  --worker-timeout-sec 3600 \
  --early-stop-patience 40 \
  --early-stop-min-delta 0.0 \
  --early-stop-monitor val_loss \
  --device cuda \
  --seed 12345 \
  --top-n-choices 4 \
  --top-n-mode max \
  --graph-connectivity fully \
  --node-feature-mode "$NODE_FEATURE_MODE" \
  --edge-feature-mode "$EDGE_FEATURE_MODE" \
  --quad-combinations-top-n 8 \
  --quad-combination-mode balanced_sample \
  --hidden-dim 256 \
  --num-layers 3 \
  --heads 4 \
  --dropout 0.2 \
  "${EXTRA_ARGS[@]}"

python -u scripts/compare_tetra_vs_gnn_batch.py \
  --image-root "$IMAGE_ROOT" \
  --gnn-run "$RUNS_ROOT/$RUN_NAME" \
  --output-dir "$RUNS_ROOT/$RUN_NAME/comparisons" \
  --label "$RUN_NAME" \
  --device cuda \
  --gnn-topk 10 \
  --gnn-brightest-stars "$GNN_BRIGHTEST_STARS" \
  --gnn-catalog-candidates 10
