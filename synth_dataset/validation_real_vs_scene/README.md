# Fidelity validation: real image vs synthetic scene

This folder holds the external fidelity check of Section 5.1.5, which confirms
that a `synthD` scene approximates what the sensor actually observes. Because the
real images carry no manual annotation, the Tetra3 solution is used as the
reference.

The validation follows the two steps of Section 5.1.5. First, a real image was
solved with Tetra3 and the closest scene was selected from `synthD`
(`scene_idx = 57961`). Second, that synthetic scene was solved with Tetra3 under
the same parameters, and the two solutions were compared. The 20 catalogue IDs
identified in the real image are all present in the scene (none exclusive to the
real image), and the angular separation between the two pointings is 0.075°
(≈ 0.44% of a 17° field of view); the roll difference is irrelevant, since
identification is invariant to in-plane rotation. These numbers are reported in
Table 5.2 of the dissertation.

Solver settings: `distortion=0`, `return_matches=True`, `return_visual=True`.

## Contents

- `inputs/` — the real image and the selected simulated scene.
- `real_image/` and `simulated_scene/` — the Tetra3 solution, matched stars and
  visual for each, solved independently.
- `comparison/comparison_summary.{txt,json}` — the quantitative comparison of the
  two solutions (the source of the Table 5.2 figures).
- `comparison/matched_catalog_overlap.csv` — the per-star catalogue overlap
  (20 shared IDs).
- `comparison/real_vs_scene_tetra3_board.png` — the side-by-side board.

The script that regenerates this comparison is `run_tetra3_real_vs_scene.py` in
this folder. The real image itself is proprietary (provided by José Lino) and is
kept here only as the single validation example.
