#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-12345}"
BASELINE_SEED="${BASELINE_SEED:-6353103531848264806}"
SYNTHD_SEED="${SYNTHD_SEED:-577215227560855758}"
DEVICE="${DEVICE:-auto}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/artifacts/thesis_pipeline}"
NUM_WORKERS="${NUM_WORKERS:-0}"
mkdir -p "$WORK_ROOT"

latest_run_dir() {
  local root="$1"
  find "$root" -maxdepth 1 -type d -name 'run*' | sort -V | tail -n 1
}

train_final_r3() {
  local dataset_dir="$1"
  local split_file="$2"
  local runs_root="$3"
  local run_name="$4"
  local epochs="$5"
  local batch_size="$6"

  python3 -u -m GNN.GNN \
    --dataset-dir "$dataset_dir" \
    --split-file "$split_file" \
    --runs-root "$runs_root" \
    --run-name "$run_name" \
    --epochs "$epochs" \
    --batch-size-scenes "$batch_size" \
    --num-workers "$NUM_WORKERS" \
    --top-n-choices 4,5,6,7,8 \
    --top-n-mode expand \
    --graph-connectivity fully \
    --node-feature-mode none \
    --edge-feature-mode distance_diagonal \
    --model-backbone edge_mlp \
    --hidden-dim 512 \
    --num-layers 5 \
    --dropout 0.2 \
    --lr 1e-3 \
    --weight-decay 1e-5 \
    --grad-clip-norm 1.0 \
    --early-stop-monitor val_loss \
    --early-stop-patience 40 \
    --loss-group-by-scene \
    --seed "$SEED" \
    --device "$DEVICE"
}

eval_final_r3() {
  local dataset_dir="$1"
  local split_file="$2"
  local checkpoint="$3"
  local batch_size="$4"

  python3 -u -m GNN.GNN \
    --eval-only \
    --checkpoint "$checkpoint" \
    --dataset-dir "$dataset_dir" \
    --split-file "$split_file" \
    --runs-root "$WORK_ROOT/eval_runs" \
    --run-name eval_final_r3 \
    --batch-size-scenes "$batch_size" \
    --num-workers "$NUM_WORKERS" \
    --top-n-choices 4,5,6,7,8 \
    --top-n-mode expand \
    --graph-connectivity fully \
    --node-feature-mode none \
    --edge-feature-mode distance_diagonal \
    --model-backbone edge_mlp \
    --hidden-dim 512 \
    --num-layers 5 \
    --dropout 0.2 \
    --loss-group-by-scene \
    --seed "$SEED" \
    --device "$DEVICE"
}

compare_real_images() {
  local checkpoint="$1"
  local image_root="${REAL_IMAGE_ROOT:-}"
  if [[ -z "$image_root" ]]; then
    echo "REAL_IMAGE_ROOT is not set; skipping proprietary 147-image comparison."
    return 0
  fi
  python3 scripts/compare_tetra_vs_gnn_batch.py \
    --image-root "$image_root" \
    --gnn-run "$(dirname "$checkpoint")" \
    --output-dir "$WORK_ROOT/real_image_comparison" \
    --label final_r3_synthd \
    --device "$DEVICE" \
    --gnn-topk 10 \
    --gnn-brightest-stars 8 \
    --gnn-anchor-stars 8 \
    --gnn-pair-topk 10 \
    --gnn-verification-budget 65
}

case "$MODE" in
  smoke)
    echo "Running smoke pipeline with a small synthetic dataset."
    SMOKE_SCENES="${SMOKE_SCENES:-512}"
    SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
    SMOKE_BATCH="${SMOKE_BATCH:-128}"
    SMOKE_SYNTH_ROOT="$WORK_ROOT/smoke_synth"
    SMOKE_SPLIT="$WORK_ROOT/smoke_split_seed${SEED}.npz"
    SMOKE_RUNS="$WORK_ROOT/smoke_gnn_runs"

    python3 synth_dataset/generate_dataset_aletorio.py \
      --stop-mode scene_budget \
      --scene-budget "$SMOKE_SCENES" \
      --seed "$SEED" \
      --runs-root "$SMOKE_SYNTH_ROOT" \
      --chunk-size-mb 32

    DATASET_DIR="$(latest_run_dir "$SMOKE_SYNTH_ROOT")"
    python3 -m GNN.split.split \
      --dataset-dir "$DATASET_DIR" \
      --seed "$SEED" \
      --output "$SMOKE_SPLIT"

    train_final_r3 "$DATASET_DIR" "$SMOKE_SPLIT" "$SMOKE_RUNS" smoke_r3 "$SMOKE_EPOCHS" "$SMOKE_BATCH"
    eval_final_r3 "$DATASET_DIR" "$SMOKE_SPLIT" "$SMOKE_RUNS/smoke_r3/best_checkpoint.pt" "$SMOKE_BATCH"
    ;;

  full)
    echo "Running full thesis pipeline. This is HPC-scale."
    FULL_BATCH="${FULL_BATCH:-2048}"
    FULL_EPOCHS="${FULL_EPOCHS:-100}"
    BASELINE_RUN="${BASELINE_RUN:-}"
    DATASET_DIR="${SYNTHD_DIR:-}"

    if [[ -z "$DATASET_DIR" ]]; then
      if [[ -z "$BASELINE_RUN" ]]; then
        BASELINE_ROOT="$WORK_ROOT/reference_star_centered"
        python3 synth_dataset/generate_dataset_baseline.py \
          --guide-stars 0 \
          --num-repeats 1 \
          --instrument-coverage \
          --seed "$BASELINE_SEED" \
          --runs-root "$BASELINE_ROOT"
        BASELINE_RUN="$(latest_run_dir "$BASELINE_ROOT")"
      fi

      SYNTHD_ROOT="$WORK_ROOT/synthD"
      python3 synth_dataset/generate_dataset_aletorio.py \
        --stop-mode appear_band_target \
        --baseline-run "$BASELINE_RUN" \
        --appear-band-margin 500 \
        --seed "$SYNTHD_SEED" \
        --runs-root "$SYNTHD_ROOT"
      DATASET_DIR="$(latest_run_dir "$SYNTHD_ROOT")"
    fi

    FULL_SPLIT="$WORK_ROOT/final_r3_synthd_split_seed${SEED}.npz"
    FULL_RUNS="$WORK_ROOT/final_r3_gnn_runs"
    python3 -m GNN.split.split \
      --dataset-dir "$DATASET_DIR" \
      --seed "$SEED" \
      --output "$FULL_SPLIT"

    train_final_r3 "$DATASET_DIR" "$FULL_SPLIT" "$FULL_RUNS" final_r3_synthd "$FULL_EPOCHS" "$FULL_BATCH"
    FINAL_CKPT="$FULL_RUNS/final_r3_synthd/best_checkpoint.pt"
    eval_final_r3 "$DATASET_DIR" "$FULL_SPLIT" "$FINAL_CKPT" "$FULL_BATCH"
    compare_real_images "$FINAL_CKPT"
    ;;

  eval-real)
    CHECKPOINT="${CHECKPOINT:-}"
    if [[ -z "$CHECKPOINT" ]]; then
      echo "Set CHECKPOINT=/path/to/best_checkpoint.pt for eval-real." >&2
      exit 2
    fi
    compare_real_images "$CHECKPOINT"
    ;;

  *)
    echo "Usage: $0 [smoke|full|eval-real]" >&2
    exit 2
    ;;
esac
