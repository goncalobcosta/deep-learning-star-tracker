#!/bin/bash
set -euo pipefail

NODE_FEATURE_MODE="${1:?usage: bash scripts/deucalion_submit_full_3datasets.sh NODE_FEATURE_MODE EDGE_FEATURE_MODE RUN_NAME}"
EDGE_FEATURE_MODE="${2:?usage: bash scripts/deucalion_submit_full_3datasets.sh NODE_FEATURE_MODE EDGE_FEATURE_MODE RUN_NAME}"
RUN_NAME="${3:?usage: bash scripts/deucalion_submit_full_3datasets.sh NODE_FEATURE_MODE EDGE_FEATURE_MODE RUN_NAME}"

submit_one() {
  local dataset_name="$1"
  local dataset_label="$2"
  echo "Submitting ${dataset_name} -> ${dataset_label}/${RUN_NAME}"
  sbatch scripts/deucalion_train_full_and_compare.sh \
    "$NODE_FEATURE_MODE" \
    "$EDGE_FEATURE_MODE" \
    "$RUN_NAME" \
    "$dataset_name" \
    "$dataset_label"
}

submit_one run1_baseline_all baseline360_dataset
submit_one run2_expA_all expA_dataset
submit_one run5_expD_all expD_dataset
