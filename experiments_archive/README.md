# Experiments archive

Supporting material from the thesis experiments that is **not** part of the
reproducible pipeline. Nothing here is needed to run
`scripts/reproduce_thesis_pipeline.sh`; it is kept only for traceability of how
the HPC sweeps were run and how the figures and analyses were produced.

| Folder | Contents |
|---|---|
| `hpc_submission/` | Deucalion SLURM submission templates for dataset generation, the architecture/feature sweeps (T0-T4) and the final runs, plus `correr_testes.py` (the historical sweep runner) and `regenerate_catmag_datasets.*`. Paths inside are Deucalion-specific. |
| `analysis/` | Diagnostic utilities used while developing the GNN (`audit_gnn_synthetic_errors.py`, `explain_gnn_graph_decision.py`, `analyze_star_frequency.py`, `check_global_ranking.py`). `magnitude_checks/` holds the catalog-magnitude vs measured-flux comparison behind Section 5.2.1.4 / Table 5.10, computed on the public `examples/data` images. |
| `grid_search_arch/` | Architecture grid-search driver and result summariser. The selected configuration is the one recorded in `configs/final_r3_synthd.yaml`. |
| `dataset_helpers/` | Visualisation and bookkeeping scripts for the synthetic dataset (sky-map plots, FOV footprints, learning-curve plots, scene-match overlays) used to produce thesis figures. |

The canonical, reproducible code lives in `tetra3/`, `synth_dataset/`, `GNN/`
and `scripts/`.
