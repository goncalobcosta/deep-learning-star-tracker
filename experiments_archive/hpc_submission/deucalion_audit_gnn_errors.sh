#!/bin/bash
#SBATCH --qos=normal
#SBATCH --account=f202603931cpcaa0g
#SBATCH --job-name=gnn_audit
#SBATCH --output=logs/gnn_audit_%j.out
#SBATCH --error=logs/gnn_audit_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

RUN_NAME="${1:-balanced_T2_node_magnitude_norm_median}"
DATASET_NAME="${2:-run5_expD_all}"
DATASET_LABEL="${3:-expD_dataset}"
SPLIT="${4:-test}"
MAX_EXAMPLES="${5:-200}"
MAX_SCENES="${6:-0}"

REPO="/projects/F202603931CPCAA0/goncalo/tetra4"
RUN_ROOT="GNN/runs/${DATASET_LABEL}/magnitude_as_is/full_${DATASET_NAME}/Deucalion_runs/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/best_checkpoint.pt"
DATASET_DIR="synth_dataset/runs/${DATASET_NAME}"
SPLIT_FILE="GNN/split/runs/${DATASET_NAME}/guide_split_seed12345.npz"
OUTPUT_DIR="GNN/error_audits/full_${DATASET_NAME}/${RUN_NAME}"

cd "$REPO"
mkdir -p logs "$OUTPUT_DIR"

module purge
module load Python/3.11.5-GCCcore-13.2.0
source .venv-gnn/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "HOST=$(hostname)"
echo "RUN_NAME=$RUN_NAME"
echo "CHECKPOINT=$CHECKPOINT"
echo "DATASET_DIR=$DATASET_DIR"
echo "SPLIT_FILE=$SPLIT_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SPLIT=$SPLIT"
echo "MAX_EXAMPLES=$MAX_EXAMPLES"
echo "MAX_SCENES=$MAX_SCENES"
nvidia-smi || true

python -u scripts/audit_gnn_synthetic_errors.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-dir "$DATASET_DIR" \
  --split-file "$SPLIT_FILE" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --topk 10 \
  --max-scenes "$MAX_SCENES" \
  --max-examples "$MAX_EXAMPLES" \
  --continue-after-max-examples \
  --device cuda
