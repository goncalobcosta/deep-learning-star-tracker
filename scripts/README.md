# Scripts

The reproducible thesis pipeline.

- `reproduce_thesis_pipeline.sh`: end-to-end entry point.
  - `smoke` builds a tiny synthetic dataset, creates the closed-set split,
    trains the final R3 GNN for one epoch and evaluates it (workstation check);
  - `full` runs the thesis-scale dataset/training/evaluation;
  - `eval-real` runs the classical-vs-GNN comparison from a checkpoint and a
    real-image folder.
- `compare_tetra_vs_gnn_batch.py`: classical Tetra3 vs Tetra3+GNN over a folder
  of real images, using the final anchor-pair search (`M=8, N=8, K=10, B=65`).
- `run_tetra4_gnn_image.py`: runs Tetra3+GNN on a single image and reports
  timings.

Historical HPC submission templates and the analysis/visualisation utilities
used during experimentation are kept under `../experiments_archive/`.
