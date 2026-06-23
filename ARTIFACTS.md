# External artifacts

The code runs end to end without any of the large thesis artifacts via the
smoke pipeline:

```bash
./scripts/reproduce_thesis_pipeline.sh smoke
```

The HPC-scale datasets, checkpoints and proprietary real images live on
Deucalion under:

```text
/projects/F202603931CPCAA0/goncalo/tetra4
```

This file records where each artifact is, what is committed, what is backed up
locally (but git-ignored), and what stays remote because it is regenerable from
the recorded seeds.

## Committed in this repository

The small result files that back the thesis tables are committed under
[`results/`](results/) (CSVs + coverage summaries, verified against the thesis:
classical 132/147, R3 `synthD` 132/147, R3 clean 126/147; `synthD` coverage
16324 / 16639 / 17323). See [`results/README.md`](results/README.md).

## Backed up locally (git-ignored)

Pulled from Deucalion and kept under git-ignored paths so a local copy exists,
without bloating the repository:

| Artifact | Local path (git-ignored) | Size |
|---|---|---:|
| R3 `synthD` checkpoint (epoch 14) | `GNN/runs/_deucalion_backup/R3_synthD_epoch14_best_checkpoint.pt` | 64M |
| R3 clean checkpoint (epoch 12) | `GNN/runs/_deucalion_backup/R3_clean_epoch12_best_checkpoint.pt` | 64M |
| R0 top4 checkpoint | `GNN/runs/_deucalion_backup/R0_top4_synthD_best_checkpoint.pt` | 28M |
| Final closed-set split | `GNN/split/runs/run5_expD_all/` | 8.8M |
| Proprietary real images (147 + outputs) | `imgs_extras/imgs_teste/` | ~180M |

Run the final real-image comparison directly from a backed-up checkpoint:

```bash
export CHECKPOINT=GNN/runs/_deucalion_backup/R3_synthD_epoch14_best_checkpoint.pt
export REAL_IMAGE_ROOT=imgs_extras/imgs_teste
./scripts/reproduce_thesis_pipeline.sh eval-real
```

## Remote only (regenerable — not pulled)

The synthetic datasets are multi-GB and fully determined by the recorded seeds,
so they are left on Deucalion rather than copied to a near-full local disk.

| Dataset | Remote path | Size | Seed |
|---|---|---:|---|
| Star-centred reference | `synth_dataset/runs/run1_baseline_all` | 2.0G | `6353103531848264806` |
| Final `synthD` | `synth_dataset/runs/run5_expD_all_rawmag` | 1.9G | `577215227560855758` |
| Clean control | `synth_dataset/runs/run5_expD_all_clean` | 2.9G | same poses, perturbations off |

Regenerate them from the seeds (no Deucalion needed):

```bash
./scripts/reproduce_thesis_pipeline.sh full   # see configs/final_r3_synthd.yaml
```

For a hard backup to external storage instead (recommended over the local repo,
which has limited free disk):

```bash
rsync -avP \
  goncalobcosta@login.deucalion.macc.fccn.pt:/projects/F202603931CPCAA0/goncalo/tetra4/synth_dataset/runs/run5_expD_all_rawmag \
  /Volumes/<external-drive>/tetra4_datasets/
```

## Re-pulling an individual artifact

```bash
rsync -avP \
  goncalobcosta@login.deucalion.macc.fccn.pt:/projects/F202603931CPCAA0/goncalo/tetra4/GNN/runs/expD_dataset/magnitude_as_is/full_run5_expD_all/Deucalion_runs/edge_mlp_dynamic_top4to8/TX_top4to8_T1diag_l5_h512/comparisons_anchor_attitude_147/by_epoch/epoch_014/best_checkpoint.pt \
  GNN/runs/_deucalion_backup/R3_synthD_epoch14_best_checkpoint.pt
```

Do not commit the checkpoints, datasets or proprietary images to Git; they are
covered by `.gitignore`. The real images are proprietary and must not be
published.
