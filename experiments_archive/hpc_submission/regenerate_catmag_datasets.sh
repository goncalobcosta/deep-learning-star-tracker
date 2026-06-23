#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

old_root="synth_dataset/runs/[EXPS]runs_tmp_validation"
new_root="synth_dataset/runs/[EXPS]runs_tmp_validation_catmag"
baseline_run="${old_root}/run1_baseline_all"

if [[ ! -d "$baseline_run" ]]; then
  echo "Baseline reference not found: $baseline_run" >&2
  exit 1
fi
if [[ -e "$new_root" ]]; then
  echo "Output already exists, refusing to overwrite: $new_root" >&2
  exit 1
fi

mkdir -p "$new_root"

python synth_dataset/generate_dataset_aletorio.py \
  --stop-mode scene_budget \
  --baseline-run "$baseline_run" \
  --chunk-size-mb 256 \
  --runs-root "$new_root" \
  --magnitude-cutoff 8 \
  --magnitude-perturb-mean 0 \
  --magnitude-perturb-sigma 0 \
  --seed 8966032090162210385

python synth_dataset/generate_dataset_aletorio.py \
  --stop-mode appear_band_target \
  --baseline-run "$baseline_run" \
  --appear-band-margin 500 \
  --chunk-size-mb 256 \
  --runs-root "$new_root" \
  --magnitude-cutoff 8 \
  --magnitude-perturb-mean 0 \
  --magnitude-perturb-sigma 0 \
  --seed 577215227560855758

mv "${new_root}/run1" "${new_root}/run2_expA_all_catmag"
mv "${new_root}/run2" "${new_root}/run5_expD_all_catmag"

python - <<'PY'
import json
from pathlib import Path

root = Path("synth_dataset/runs/[EXPS]runs_tmp_validation_catmag")
for name in ("run2_expA_all_catmag", "run5_expD_all_catmag"):
    manifest = json.loads((root / name / "dataset_manifest.json").read_text())
    params = manifest["run"]["parameters"]
    print(
        name,
        "scenes=", manifest["counts"]["generated_scene_count"],
        "mean=", params["magnitude_perturb_mean"],
        "sigma=", params["magnitude_perturb_sigma"],
        "seed=", params["seed"],
    )
PY
