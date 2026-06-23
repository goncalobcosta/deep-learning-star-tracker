#!/bin/bash
set -euo pipefail

DATASET_NAME="${1:-run5_expD_all}"
DATASET_LABEL="${2:-expD_dataset}"

REPO="/projects/F202603931CPCAA0/goncalo/tetra4"
RUNS_ROOT="GNN/runs/${DATASET_LABEL}/magnitude_as_is/full_${DATASET_NAME}/Deucalion_runs"

cd "$REPO"
mkdir -p logs

COMMON_ALL70_ARGS=(
  --quad-combination-mode all
  --loss-group-by-scene
  --batch-size-scenes 8192
  --epochs 8
  --early-stop-patience 3
  --log-every-batches 250
)

submit_one() {
  local job_name="$1"
  local node_mode="$2"
  local edge_mode="$3"
  local run_name="$4"
  local init_run="$5"
  shift 5

  local extra_args=("${COMMON_ALL70_ARGS[@]}")
  if [[ -n "$init_run" ]]; then
    local init_checkpoint="${RUNS_ROOT}/${init_run}/best_checkpoint.pt"
    if [[ -f "$init_checkpoint" ]]; then
      extra_args+=(--init-checkpoint "$init_checkpoint")
    else
      echo "Warning: init checkpoint not found, training from scratch: $init_checkpoint" >&2
    fi
  fi
  extra_args+=("$@")

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
    "${extra_args[@]}"
}

# Full-dataset all70 fine-tuning. Each scene contributes all C(8,4)=70 quads,
# and loss is averaged by scene so scenes do not become 70x heavier.
submit_one "all70_t0_l3_h128" "none" "distance_max" "all70_T0_l3_h128" "balanced_T0_l3_h128" \
  --hidden-dim 128 --num-layers 3
submit_one "all70_t0_l3_h256" "none" "distance_max" "all70_T0_l3_h256" "balanced_T0_l3_h256" \
  --hidden-dim 256 --num-layers 3
submit_one "all70_t0_l5_h128" "none" "distance_max" "all70_T0_l5_h128" "balanced_T0_l5_h128" \
  --hidden-dim 128 --num-layers 5
submit_one "all70_t0_l5_h256" "none" "distance_max" "all70_T0_l5_h256" "balanced_T0_l5_h256" \
  --hidden-dim 256 --num-layers 5

submit_one "all70_t1_dist_diag" "none" "distance_diagonal" "all70_T1_dist_diagonal" "balanced_T1_dist_diagonal"
submit_one "all70_t1_dist_raw" "none" "distance_raw" "all70_T1_dist_raw" "balanced_T1_dist_raw"

submit_one "all70_t2_rank" "magnitude_rank" "distance_max" "all70_T2_node_magnitude_rank" "balanced_T2_node_magnitude_rank"
submit_one "all70_t2_mag" "magnitude" "distance_max" "all70_T2_node_magnitude" "balanced_T2_node_magnitude"
submit_one "all70_t2_mag_nmax" "magnitude_norm_max" "distance_max" "all70_T2_node_magnitude_norm_max" "balanced_T2_node_magnitude_norm_max"
submit_one "all70_t2_mag_nmedian" "magnitude_norm_median" "distance_max" "all70_T2_node_magnitude_norm_median" "balanced_T2_node_magnitude_norm_median"
submit_one "all70_t2_mag_sub" "magnitude_subtracted" "distance_max" "all70_T2_node_magnitude_subtracted" "balanced_T2_node_magnitude_subtracted"
submit_one "all70_t2_mag_sub_nmed" "magnitude_subtracted_norm_median" "distance_max" "all70_T2_node_magnitude_subtracted_norm_median" "balanced_T2_node_magnitude_subtracted_norm_max"

submit_one "all70_t3_edge_dmag" "magnitude_rank" "distance_max_dmag" "all70_T3_edge_dmag" "balanced_T3_edge_dmag"
submit_one "all70_t3_edge_dmag_nm" "magnitude_norm_median" "distance_max_dmag_node" "all70_T3_edge_dmag_nmedian" "balanced_T3_edge_dmag_nmedian"

# Auxiliary-loss probes on the best balanced real-image performers.
submit_one "all70_t2_nmed_loss" "magnitude_norm_median" "distance_max" "all70_T2_node_magnitude_norm_median_loss_dr" "balanced_T2_node_magnitude_norm_median" \
  --class-distance-loss-weight 0.2 \
  --class-rank-loss-weight 0.05
submit_one "all70_t3_nm_loss" "magnitude_norm_median" "distance_max_dmag_node" "all70_T3_edge_dmag_nmedian_loss_dr" "balanced_T3_edge_dmag_nmedian" \
  --class-distance-loss-weight 0.2 \
  --class-rank-loss-weight 0.05
