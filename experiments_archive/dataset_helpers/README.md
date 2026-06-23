# Dataset Helper Scripts

The official dataset generators are one level above this folder:

- `generate_dataset_baseline.py`
- `generate_dataset_aletorio.py`

This folder contains auxiliary visualization, validation and historical
experiment helpers. They are not part of the final reproduction pipeline.

- `visualize_*.py`, `create_timelapse.py`, `plot_learning_curves.py`: plotting
  and inspection helpers.
- `find_best_scene_match.py`, `semantic_overlay_real_vs_scene.py`,
  `overlay_match_visuals.py`: real-image versus synthetic-scene diagnostics.
- `[EXPS]scripts_for_run_two_aproachs/`: historical dataset-generation harness
  used while comparing generation strategies.
