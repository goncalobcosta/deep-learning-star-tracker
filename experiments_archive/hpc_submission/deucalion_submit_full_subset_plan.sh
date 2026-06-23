#!/bin/bash
set -euo pipefail

DATASET_NAME="${1:-run5_expD_all}"
DATASET_LABEL="${2:-expD_dataset}"

REPO="/projects/F202603931CPCAA0/goncalo/tetra4"
cd "$REPO"
mkdir -p logs

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
    "$@"
}

# Mirror the balanced subset battery on the full dataset.
submit_one "full_t0_l3_h128" "none" "distance_max" "balanced_T0_l3_h128" --hidden-dim 128 --num-layers 3
submit_one "full_t0_l3_h256" "none" "distance_max" "balanced_T0_l3_h256" --hidden-dim 256 --num-layers 3
submit_one "full_t0_l5_h128" "none" "distance_max" "balanced_T0_l5_h128" --hidden-dim 128 --num-layers 5
submit_one "full_t0_l5_h256" "none" "distance_max" "balanced_T0_l5_h256" --hidden-dim 256 --num-layers 5

submit_one "full_t1_dist_raw" "none" "distance_raw" "balanced_T1_dist_raw"
submit_one "full_t1_dist_diag" "none" "distance_diagonal" "balanced_T1_dist_diagonal"

submit_one "full_t2_rank" "magnitude_rank" "distance_max" "balanced_T2_node_magnitude_rank"
submit_one "full_t2_mag" "magnitude" "distance_max" "balanced_T2_node_magnitude"
submit_one "full_t2_sub" "magnitude_subtracted" "distance_max" "balanced_T2_node_magnitude_subtracted"
submit_one "full_t2_norm_max" "magnitude_norm_max" "distance_max" "balanced_T2_node_magnitude_norm_max"
submit_one "full_t2_norm_med" "magnitude_norm_median" "distance_max" "balanced_T2_node_magnitude_norm_median"
submit_one "full_t2_sub_norm_max" "magnitude_subtracted_norm_max" "distance_max" "balanced_T2_node_magnitude_subtracted_norm_max"
submit_one "full_t2_rank_loss" "magnitude_rank" "distance_max" "balanced_T2_node_magnitude_rank_improve_loss" \
  --class-distance-loss-weight 0.2 \
  --class-rank-loss-weight 0.2

submit_one "full_t3_edge_dmag" "magnitude_rank" "distance_max_dmag" "balanced_T3_edge_dmag"
submit_one "full_t3_edge_dmag_node" "magnitude_norm_median" "distance_max_dmag_node" "balanced_T3_edge_dmag_nmedian"
submit_one "full_t3_edge_rank" "magnitude_rank" "distance_max_dmag_node" "balanced_T3_edge_rank_drank"
