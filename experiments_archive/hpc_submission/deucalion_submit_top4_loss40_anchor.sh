#!/bin/bash
set -euo pipefail

REPO="/projects/F202603931CPCAA0/goncalo/tetra4"
cd "$REPO"
mkdir -p logs scripts/generated_jobs

SUBSET_PLAN="top4_loss40_anchor_b512"
FULL_PREFIX="top4_loss40_anchor"
COMMON_TOP4_ARGS=(--quad-combinations-top-n 0 --class-scope-top-n 4)

write_subset_job() {
  local job_tag="$1"
  local dataset_dir="$2"
  local split_file="$3"
  local real_image="$4"
  local runs_root="$5"
  local job_file="scripts/generated_jobs/${job_tag}.sh"

  cat > "$job_file" <<SH
#!/bin/bash
#SBATCH --qos=normal
#SBATCH --account=f202603931cpcaa0g
#SBATCH --job-name=${job_tag}
#SBATCH --output=logs/${job_tag}_%j.out
#SBATCH --error=logs/${job_tag}_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

cd "$REPO"
mkdir -p logs

module purge
module load Python/3.11.5-GCCcore-13.2.0
source .venv-gnn/bin/activate

echo "HOST=\$(hostname)"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

PLAN_ROOT="$runs_root/$SUBSET_PLAN"
mkdir -p "\$PLAN_ROOT"

run_one() {
  local run_name="\$1"
  local node_feature="\$2"
  local edge_feature="\$3"
  local hidden_dim="\$4"
  local num_layers="\$5"
  shift 5

  echo "===== TRAIN \$run_name ====="
  python -u -m GNN.GNN \\
    --dataset-dir "$dataset_dir" \\
    --split-file "$split_file" \\
    --runs-root "\$PLAN_ROOT" \\
    --run-name "\$run_name" \\
    --epochs 60 \\
    --batch-size-scenes 512 \\
    --num-workers 16 \\
    --cache-chunks 2 \\
    --log-every-batches 200 \\
    --worker-timeout-sec 1800 \\
    --early-stop-patience 40 \\
    --early-stop-min-delta 0.0 \\
    --early-stop-monitor val_loss \\
    --device cuda \\
    --seed 12345 \\
    --top-n-choices 4 \\
    --top-n-mode max \\
    --graph-connectivity fully \\
    --node-feature-mode "\$node_feature" \\
    --edge-feature-mode "\$edge_feature" \\
    --quad-combinations-top-n 0 \\
    --class-scope-top-n 4 \\
    --hidden-dim "\$hidden_dim" \\
    --num-layers "\$num_layers" \\
    --heads 4 \\
    --dropout 0.2 \\
    "\$@"

  echo "===== REAL EVAL TOP4 \$run_name ====="
  python -u -m GNN.eval_examples \\
    --checkpoint "\$PLAN_ROOT/\$run_name/best_checkpoint.pt" \\
    --image "$real_image" \\
    --device cuda \\
    --quad-combinations-top-n 0 \\
    --brightest-k 4 \\
    --topk 10 \\
    > "\$PLAN_ROOT/\$run_name/eval_real_image_top4.txt"
}

run_one "T0_l3_h128" "none" "distance_max" 128 3
run_one "T0_l3_h256" "none" "distance_max" 256 3
run_one "T0_l5_h128" "none" "distance_max" 128 5
run_one "T0_l5_h256" "none" "distance_max" 256 5

run_one "T1_dist_raw" "none" "distance_raw" 256 3
run_one "T1_dist_diagonal" "none" "distance_diagonal" 256 3

run_one "T2_magnitude" "magnitude" "distance_max" 256 3
run_one "T2_subtracted" "magnitude_subtracted" "distance_max" 256 3
run_one "T2_norm_max" "magnitude_norm_max" "distance_max" 256 3
run_one "T2_norm_median" "magnitude_norm_median" "distance_max" 256 3
run_one "T2_sub_norm_max" "magnitude_subtracted_norm_max" "distance_max" 256 3

run_one "T3_edge_dmag" "magnitude" "distance_max_dmag" 256 3
run_one "T3_edge_dmag_median" "magnitude_norm_median" "distance_max_dmag_node" 256 3

run_one "T4_rank" "magnitude_rank" "distance_max" 256 3
run_one "T5_rank_loss" "magnitude_rank" "distance_max" 256 3 \\
  --class-distance-loss-weight 0.2 \\
  --class-rank-loss-weight 0.2
SH

  echo "Submitting $job_file"
  sbatch "$job_file"
}

submit_full() {
  local job_name="$1"
  local node_feature="$2"
  local edge_feature="$3"
  local run_name="$4"
  shift 4

  echo "Submitting full ${job_name} -> ${run_name}"
  GNN_BRIGHTEST_STARS=4 sbatch \
    --export=ALL,GNN_BRIGHTEST_STARS=4 \
    --job-name="$job_name" \
    --output="logs/${job_name}_%j.out" \
    --error="logs/${job_name}_%j.err" \
    scripts/deucalion_train_full_and_compare.sh \
    "$node_feature" \
    "$edge_feature" \
    "$run_name" \
    "run5_expD_all" \
    "expD_dataset" \
    "${COMMON_TOP4_ARGS[@]}" \
    "$@"
}

write_subset_job \
  "top4_t0t5_lino" \
  "synth_dataset/runs/1000ms_18-50_subset_run1/run1" \
  "GNN/split/runs/run_1000ms_18-50_subset/guide_split_seed12345.npz" \
  "imgs_extras/imgs_teste/img1_1000ms_18-50/1000ms_18-50-26-712529.tiff" \
  "GNN/runs/expD_dataset/magnitude_as_is/img1_1000ms_18-50/Deucalion_runs"

write_subset_job \
  "top4_t0t5_img3" \
  "synth_dataset/runs/img3_obs016_25732_subset_run5_expD/run1" \
  "GNN/split/runs/run_img3_obs016_25732_subset/guide_split_seed12345.npz" \
  "imgs_extras/imgs_teste/img3_obs016_25732_img4_201303/img_4_tiff_2026_03_23_20_13_03_738288_695013.tiff" \
  "GNN/runs/expD_dataset/magnitude_as_is/img3_obs016_25732/Deucalion_runs"

submit_full "top4_t0_l3_h128" "none" "distance_max" "${FULL_PREFIX}_T0_l3_h128" --hidden-dim 128 --num-layers 3
submit_full "top4_t0_l3_h256" "none" "distance_max" "${FULL_PREFIX}_T0_l3_h256" --hidden-dim 256 --num-layers 3
submit_full "top4_t0_l3_h512" "none" "distance_max" "${FULL_PREFIX}_T0_l3_h512" --hidden-dim 512 --num-layers 3
submit_full "top4_t0_l5_h128" "none" "distance_max" "${FULL_PREFIX}_T0_l5_h128" --hidden-dim 128 --num-layers 5
submit_full "top4_t0_l5_h256" "none" "distance_max" "${FULL_PREFIX}_T0_l5_h256" --hidden-dim 256 --num-layers 5
submit_full "top4_t0_l5_h512" "none" "distance_max" "${FULL_PREFIX}_T0_l5_h512" --hidden-dim 512 --num-layers 5

submit_full "top4_t1_raw" "none" "distance_raw" "${FULL_PREFIX}_T1_dist_raw"
submit_full "top4_t1_diag" "none" "distance_diagonal" "${FULL_PREFIX}_T1_dist_diagonal"

submit_full "top4_t2_mag" "magnitude" "distance_max" "${FULL_PREFIX}_T2_magnitude"
submit_full "top4_t2_sub" "magnitude_subtracted" "distance_max" "${FULL_PREFIX}_T2_subtracted"
submit_full "top4_t2_nmax" "magnitude_norm_max" "distance_max" "${FULL_PREFIX}_T2_norm_max"
submit_full "top4_t2_nmed" "magnitude_norm_median" "distance_max" "${FULL_PREFIX}_T2_norm_median"
submit_full "top4_t2_submax" "magnitude_subtracted_norm_max" "distance_max" "${FULL_PREFIX}_T2_sub_norm_max"

submit_full "top4_t3_dmag" "magnitude" "distance_max_dmag" "${FULL_PREFIX}_T3_edge_dmag"
submit_full "top4_t3_dmagnm" "magnitude_norm_median" "distance_max_dmag_node" "${FULL_PREFIX}_T3_edge_dmag_median"

submit_full "top4_t4_rank" "magnitude_rank" "distance_max" "${FULL_PREFIX}_T4_rank"
submit_full "top4_t5_rankloss" "magnitude_rank" "distance_max" "${FULL_PREFIX}_T5_rank_loss" \
  --class-distance-loss-weight 0.2 \
  --class-rank-loss-weight 0.2
