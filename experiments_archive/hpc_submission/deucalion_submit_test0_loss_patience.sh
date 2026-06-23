#!/bin/bash
set -euo pipefail

REPO="/projects/F202603931CPCAA0/goncalo/tetra4"

cd "$REPO"
mkdir -p logs scripts/generated_jobs

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

python -u scripts/correr_testes.py \\
  --dataset-dir "$dataset_dir" \\
  --split-file "$split_file" \\
  --real-image "$real_image" \\
  --runs-root "$runs_root" \\
  --plan-name test0_loss_patience_b512 \\
  --epochs 60 \\
  --batch-size-scenes 512 \\
  --num-workers 16 \\
  --cache-chunks 2 \\
  --log-every-batches 200 \\
  --worker-timeout-sec 1800 \\
  --early-stop-patience 40 \\
  --early-stop-min-delta 0.0 \\
  --early-stop-monitor val_loss \\
  --selection-metric val_loss \\
  --device cuda \\
  --seed 12345 \\
  --quad-top-n 8 \\
  --graph-regimes balanced \\
  --only-test0 \\
  --test0-hidden-dims 128,256 \\
  --real-brightest-k 8 \\
  --real-topk 10
SH

  echo "Submitting $job_file"
  sbatch "$job_file"
}

submit_full_t0() {
  local job_name="$1"
  local run_name="$2"
  local hidden_dim="$3"
  local num_layers="$4"

  echo "Submitting full ${run_name}"
  sbatch \
    --job-name="$job_name" \
    --output="logs/${job_name}_%j.out" \
    --error="logs/${job_name}_%j.err" \
    scripts/deucalion_train_full_and_compare.sh \
    none \
    distance_max \
    "$run_name" \
    run5_expD_all \
    expD_dataset \
    --hidden-dim "$hidden_dim" \
    --num-layers "$num_layers"
}

write_subset_job \
  "loss_t0_img3" \
  "synth_dataset/runs/img3_obs016_25732_subset_run5_expD/run1" \
  "GNN/split/runs/run_img3_obs016_25732_subset/guide_split_seed12345.npz" \
  "imgs_extras/imgs_teste/img3_obs016_25732_img4_201303/img_4_tiff_2026_03_23_20_13_03_738288_695013.tiff" \
  "GNN/runs/expD_dataset/magnitude_as_is/img3_obs016_25732/Deucalion_runs"

write_subset_job \
  "loss_t0_lino" \
  "synth_dataset/runs/1000ms_18-50_subset_run1/run1" \
  "GNN/split/runs/run_1000ms_18-50_subset/guide_split_seed12345.npz" \
  "imgs_extras/imgs_teste/img1_1000ms_18-50/1000ms_18-50-26-712529.tiff" \
  "GNN/runs/expD_dataset/magnitude_as_is/img1_1000ms_18-50/Deucalion_runs"

submit_full_t0 "loss_full_l3_h128" "loss_T0_l3_h128" 128 3
submit_full_t0 "loss_full_l3_h256" "loss_T0_l3_h256" 256 3
submit_full_t0 "loss_full_l3_h512" "loss_T0_l3_h512" 512 3
submit_full_t0 "loss_full_l5_h128" "loss_T0_l5_h128" 128 5
submit_full_t0 "loss_full_l5_h256" "loss_T0_l5_h256" 256 5
submit_full_t0 "loss_full_l5_h512" "loss_T0_l5_h512" 512 5
