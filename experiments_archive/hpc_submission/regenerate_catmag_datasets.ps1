$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$oldRoot = Join-Path $repo "synth_dataset\runs\[EXPS]runs_tmp_validation"
$newRoot = Join-Path $repo "synth_dataset\runs\[EXPS]runs_tmp_validation_catmag"
$baselineRun = Join-Path $oldRoot "run1_baseline_all"

if (-not (Test-Path -LiteralPath $baselineRun)) {
    throw "Baseline reference not found: $baselineRun"
}
if (Test-Path -LiteralPath $newRoot) {
    throw "Output already exists, refusing to overwrite: $newRoot"
}

New-Item -ItemType Directory -Force -Path $newRoot | Out-Null

python synth_dataset\generate_dataset_aletorio.py `
    --stop-mode scene_budget `
    --baseline-run $baselineRun `
    --chunk-size-mb 256 `
    --runs-root $newRoot `
    --magnitude-cutoff 8 `
    --magnitude-perturb-mean 0 `
    --magnitude-perturb-sigma 0 `
    --seed 8966032090162210385

python synth_dataset\generate_dataset_aletorio.py `
    --stop-mode appear_band_target `
    --baseline-run $baselineRun `
    --appear-band-margin 500 `
    --chunk-size-mb 256 `
    --runs-root $newRoot `
    --magnitude-cutoff 8 `
    --magnitude-perturb-mean 0 `
    --magnitude-perturb-sigma 0 `
    --seed 577215227560855758

Rename-Item -LiteralPath (Join-Path $newRoot "run1") -NewName "run2_expA_all_catmag"
Rename-Item -LiteralPath (Join-Path $newRoot "run2") -NewName "run5_expD_all_catmag"

foreach ($name in "run2_expA_all_catmag", "run5_expD_all_catmag") {
    $manifest = Join-Path $newRoot "$name\dataset_manifest.json"
    $json = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
    $params = $json.run.parameters
    [pscustomobject]@{
        dataset = $name
        scenes = $json.counts.generated_scene_count
        magnitude_perturb_mean = $params.magnitude_perturb_mean
        magnitude_perturb_sigma = $params.magnitude_perturb_sigma
        seed = $params.seed
    }
}
