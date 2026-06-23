param(
    [string]$DatasetDir = "c:\Users\User\Desktop\tudo\feup\mestrado\tese\tetra4\synth_dataset\runs_1000ms_18-50_expd\run1",
    [string]$SplitFile = "c:\Users\User\Desktop\tudo\feup\mestrado\tese\tetra4\GNN\split\runs\run_1000ms_18-50_expd\guide_split_seed12345.npz",
    [string]$Device = "cuda",
    [int]$Epochs = 100,
    [int]$BatchSizeScenes = 256,
    [int]$NumWorkers = 4,
    [string]$TopNChoices = "4,8,12,16,20,22",
    [int]$KNeighbors = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = "c:\Users\User\Desktop\tudo\feup\mestrado\tese\tetra4"
Set-Location $repoRoot

$configs = @(
    @{ Name = "grid_h128_l3"; Hidden = 128; Layers = 3 },
    @{ Name = "grid_h192_l3"; Hidden = 192; Layers = 3 },
    @{ Name = "grid_h128_l4"; Hidden = 128; Layers = 4 },
    @{ Name = "grid_h192_l4"; Hidden = 192; Layers = 4 }
)

foreach ($cfg in $configs) {
    Write-Host ""
    Write-Host "=== Running $($cfg.Name) ==="
    python -u -m GNN.GNN `
      --dataset-dir "$DatasetDir" `
      --split-file "$SplitFile" `
      --epochs $Epochs `
      --k-neighbors $KNeighbors `
      --top-n-choices "$TopNChoices" `
      --batch-size-scenes $BatchSizeScenes `
      --num-workers $NumWorkers `
      --device "$Device" `
      --hidden-dim $cfg.Hidden `
      --num-layers $cfg.Layers `
      --run-name $cfg.Name
}

Write-Host ""
Write-Host "Grid search finished."
