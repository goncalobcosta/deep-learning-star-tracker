#!/bin/bash
set -euo pipefail

DATASET_NAME="${1:-run5_expD_all}"
DATASET_LABEL="${2:-expD_dataset}"
SPLIT_FILE="${3:-GNN/split/runs/run5_expD_all/guide_split_seed12345_top8_train_covered.npz}"

REPO="/projects/F202603931CPCAA0/goncalo/tetra4"
cd "$REPO"
mkdir -p logs

COMMON_ARGS=(
  --split-file "$SPLIT_FILE"
  --class-scope train_candidates
  --class-scope-top-n 8
  --epochs 80
  --early-stop-patience 10
)

submit_one() {
  local job_name="$1"
  local node_mode="$2"
  local edge_mode="$3"
  local run_name="$4"
  shift 4

  echo "Submitting ${job_name} -> ${DATASET_LABEL}/${run_name}"
  sbatch \
    --job-name="$job_name" \
    --output="logs/${job_name}_%j.out" \
    --error="logs/${job_name}_%j.err" \
    scripts/deucalion_train_full_and_compare.sh \
    "$node_mode" \
    "$edge_mode" \
    "$run_name" \
    "$DATASET_NAME" \
    "$DATASET_LABEL" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# Restricted-output tests: train classes are exactly top8 candidates covered by the train split.
submit_one "tcov_t2_nmedian" "magnitude_norm_median" "distance_max" "traincovered_T2_node_magnitude_norm_median"
submit_one "tcov_t2_sub" "magnitude_subtracted" "distance_max" "traincovered_T2_node_magnitude_subtracted"
submit_one "tcov_t2_mag" "magnitude" "distance_max" "traincovered_T2_node_magnitude"
submit_one "tcov_t3_dmag" "magnitude_rank" "distance_max_dmag" "traincovered_T3_edge_dmag"
submit_one "tcov_t3_dmag_nm" "magnitude_norm_median" "distance_max_dmag_node" "traincovered_T3_edge_dmag_nmedian"

# Probe the catalog-distance/rank auxiliary loss on the best previous full model shape.
submit_one "tcov_t2_nmed_loss" "magnitude_norm_median" "distance_max" "traincovered_T2_node_magnitude_norm_median_loss_dr" \
  --class-distance-loss-weight 0.2 \
  --class-rank-loss-weight 0.05
