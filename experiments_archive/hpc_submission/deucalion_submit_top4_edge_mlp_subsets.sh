#!/bin/bash
set -euo pipefail

REPO="/projects/F202603931CPCAA0/goncalo/tetra4"
cd "$REPO"
mkdir -p logs scripts/generated_jobs

PLAN_NAME="top4_edge_mlp_rank1to4_b512"

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

PLAN_ROOT="$runs_root/$PLAN_NAME"
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
    --model-backbone edge_mlp \\
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

run_one "T4_rank" "magnitude_rank_1based" "distance_max" 256 3
run_one "T5_rank_loss" "magnitude_rank_1based" "distance_max" 256 3 \\
  --class-distance-loss-weight 0.2 \\
  --class-rank-loss-weight 0.2
SH

  echo "Submitting $job_file"
  sbatch "$job_file"
}

write_subset_job \
  "mlp_top4_lino" \
  "synth_dataset/runs/1000ms_18-50_subset_run1/run1" \
  "GNN/split/runs/run_1000ms_18-50_subset/guide_split_seed12345.npz" \
  "imgs_extras/imgs_teste/img1_1000ms_18-50/1000ms_18-50-26-712529.tiff" \
  "GNN/runs/expD_dataset/magnitude_as_is/img1_1000ms_18-50/Deucalion_runs"

write_subset_job \
  "mlp_top4_img3" \
  "synth_dataset/runs/img3_obs016_25732_subset_run5_expD/run1" \
  "GNN/split/runs/run_img3_obs016_25732_subset/guide_split_seed12345.npz" \
  "imgs_extras/imgs_teste/img3_obs016_25732_img4_201303/img_4_tiff_2026_03_23_20_13_03_738288_695013.tiff" \
  "GNN/runs/expD_dataset/magnitude_as_is/img3_obs016_25732/Deucalion_runs"
