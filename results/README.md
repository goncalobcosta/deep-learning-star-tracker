# Final result artifacts

Small result files behind the thesis tables, pulled from the Deucalion run
directories. They let an evaluator re-check the headline numbers without
re-running the HPC pipeline. The trained checkpoints, the proprietary real
images and the multi-GB datasets are not committed; see [`../ARTIFACTS.md`](../ARTIFACTS.md).

## Per-image comparison CSVs (147 real images)

One row per real image, with both the classical Tetra3 and the Tetra3+GNN
outcome (solved flag, attitude, timings, candidate and verification counts).

| File | Thesis | Solve rate (verified) |
|---|---|---|
| `tetra3_classical_reference_147.csv` | Classical Tetra3 reference | `classic_ok` = 132/147 |
| `tetra3_vs_gnn_R3_synthD_epoch14_M8_N8_K10_B65.csv` | Tetra3+GNN R3 (`synthD`), Tables 5.17-5.20 | `gnn_ok` = 132/147 |
| `tetra3_vs_gnn_R3_clean_epoch12_M8_N8_K10_B65.csv` | Tetra3+GNN R3 (clean control) | `gnn_ok` = 126/147 |

Re-check a solve-rate column:

```bash
# gnn_ok column ($4); use $3 for the classical classic_ok column
awk -F, 'NR>1 && tolower($4)=="true"{n++} END{print n"/147"}' \
  tetra3_vs_gnn_R3_synthD_epoch14_M8_N8_K10_B65.csv
```

## Dataset coverage (Table 5.1)

| File | Reports |
|---|---|
| `synthD_coverage_summary.json` | final `synthD`: 3,265,893 scenes; coverage min 16324 / mean 16639 / max 17323 (range 999) |
| `baseline_coverage_summary.json` | star-centred reference: 3,174,480 scenes; min 8723 / mean 16823 / max 25937 (range 17214) |
| `synthD_dataset_manifest.json`, `synthD_summary.txt` | full generation parameters of the final `synthD` run |
