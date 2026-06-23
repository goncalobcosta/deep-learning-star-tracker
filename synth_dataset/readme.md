# Synthetic dataset generation

This folder generates the centroid-level synthetic scenes used to train and
evaluate the GNN (Section 4.2 of the dissertation). A *scene* is not a
photorealistic image but the set of detections that would fall within the
camera's field of view for a given attitude — the expected output of the
centroiding stage — so the dataset is aligned with the exact point of the Tetra3
pipeline this work intervenes on.

Two generators implement the two strategies of Section 4.2.8:

- `generate_dataset_baseline.py` — the star-centred reference strategy, which
  centres the camera on each catalogue star and sweeps the 360 integer rolls. It
  is used only to measure the baseline catalogue coverage.
- `generate_dataset_aletorio.py` — the final random-boresight strategy that
  produces `synthD`, with boresights drawn uniformly over the celestial sphere
  and a coverage band that caps already-frequent stars.

Both read the Tetra3 catalogue from `tetra3/data/default_database.npz`, so the
scene classes are exactly the 8818 stars of the `star_table` on which the
reference operates.

## Fixed methodological parameters

These match Section 4.2 and `configs/final_r3_synthd.yaml`:

- catalogue: Tetra3 `default_database.npz`, 8818 stars from `star_table`;
- sensor: `1280 × 960` px; field of view `17.2°` (h), `13.0°` (v), `21.0°` (d);
- per-scene perturbations: `0–5` false detections, `0–5` dropped real stars, and
  a per-point positional uncertainty sampled in `[0.25, 1.0]` px (Section 4.2.7);
- `point_magnitude` is the **catalogue** magnitude; magnitude perturbation is
  disabled in the final dataset (`mean=0.0, sigma=0.0`), which is why the run is
  named `…_rawmag`;
- scene guide star: the brightest real star remaining after dropout;
- seeds: star-centred reference `6353103531848264806`; `synthD`
  `577215227560855758`; split/training `12345`.

## Star-centred reference

```bash
python3 synth_dataset/generate_dataset_baseline.py \
  --guide-stars 0 --num-repeats 1 --instrument-coverage \
  --seed 6353103531848264806 \
  --runs-root synth_dataset/runs
```

With the 8818 catalogue stars and one repetition this yields
`8818 × 1 × 360 = 3 174 480` scenes (Eq. 4.20). Because the boresights always
coincide with catalogue stars, this strategy over-observes dense regions; its
coverage range is reported in Section 5.1 as the imbalance the final dataset
corrects.

## Final `synthD`

The random-boresight generator takes the reference run folder and keeps each
star's coverage inside a band of the reference mean ± 500 appearances
(Section 4.2.8):

```bash
python3 synth_dataset/generate_dataset_aletorio.py \
  --stop-mode appear_band_target \
  --baseline-run <reference-run-dir> \
  --appear-band-margin 500 \
  --seed 577215227560855758 \
  --runs-root synth_dataset/runs
```

The thesis run (`run5_expD_all_rawmag`) produces `3 265 893` scenes with a
coverage range of `999` (min 16 324, mean 16 639, max 17 323), an order of
magnitude tighter than the reference (Table 5.1).

Each run folder contains the chunked dataset under `dataset/` plus
`dataset_manifest.json` and `coverage_summary.json` (the manifest and coverage
of the thesis run are committed under `results/`).

### Stored fields

Per detected point (Section 4.2.6):

- `point_yx` — centroid position on the sensor;
- `point_star_id` — catalogue index, `-1` for a false detection;
- `point_is_false_star` — `0` for a real detection, `-1` for a false one;
- `point_magnitude` — catalogue magnitude (a synthetic in-range value for false
  detections).

Per scene: `scene_point_count`, `guide_star_index`,
`pre_dropout_real_star_count`, `scene_dropout_count`, `scene_false_stars_count`,
`scene_real_star_count`, `scene_total_point_count`, `scene_seed`, `roll_degree`.

## Reproducibility

The generator is deterministic at the scene level: every scene is produced from
the seed recorded in its `scene_seed` field, so any subset — including the full
`synthD` — can be regenerated exactly by replaying those seeds. For a quick local
check, build a small dataset and exercise the split/training/evaluation through
the root script:

```bash
./scripts/reproduce_thesis_pipeline.sh smoke
```

## Related material

`validation_real_vs_scene/` holds the fidelity check of Section 5.1.5 (a `synthD`
scene compared, through Tetra3, against the corresponding real image).
