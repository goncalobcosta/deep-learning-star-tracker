Deep learning enabled star tracker — Tetra3 + GNN
=================================================

This repository contains the code, configuration and result artifacts for the
master's dissertation *Deep learning enabled star tracker* (Gonçalo Bessa Costa,
FCUP/FEUP, 2026). The work replaces a single stage of the Tetra3 lost-in-space
pipeline — the candidate-generation stage, that is, the tolerance-expanded hash
lookup — with a graph neural network (GNN), while leaving Tetra3's centroid
extraction and final geometric verification unchanged. The GNN is not a
standalone identifier: it proposes, for each detected centroid, a short ordered
list of catalogue candidates, and the geometric verification of Tetra3 remains
the final acceptance criterion. The objective is therefore not to replace
Tetra3, but to make its candidate-generation stage more selective — fewer
hypotheses and geometric verifications, and lower latency — while preserving its
reliability. The methodology and results implemented here correspond to
Chapters 4 and 5 of the dissertation.

Reproducing the pipeline
------------------------

Synthetic dataset generation, the closed-set split, GNN training, synthetic
evaluation and the classical-vs-hybrid comparison on real images are all driven
by a single entry point:

.. code-block:: bash

   ./scripts/reproduce_thesis_pipeline.sh smoke   # wiring check on a workstation
   ./scripts/reproduce_thesis_pipeline.sh full    # thesis-scale run (HPC)

The ``smoke`` mode builds a small synthetic dataset, creates the split, trains
the final R3 graph formulation for one epoch and evaluates it on the synthetic
split; it checks that the pipeline is correctly wired, not that it reproduces the
thesis-scale numbers. The ``full`` mode regenerates the thesis dataset and
trains the final model from the recorded seeds, and is HPC-scale. The exact
numbers behind the result tables are committed under ``results/``; the trained
checkpoints and the multi-GB datasets are catalogued in ``ARTIFACTS.md``.

Final configuration
-------------------

The canonical configuration is recorded in ``configs/final_r3_synthd.yaml`` and
matches Chapter 4 of the dissertation.

**Dataset (Section 4.2).**

* Tetra3 ``default_database.npz``: the 8818-star ``star_table`` (Section 4.2.1),
  so the model's classes are exactly the catalogue on which Tetra3 operates.
* Sensor: 1280 × 960 px; field of view 17.2° horizontal, 13.0° vertical, 21.0°
  diagonal (Section 4.2.4).
* Final ``synthD``: random-boresight scenes (boresights uniform over the sphere)
  with coverage control inside a band of the reference mean ± 500 appearances
  (Section 4.2.8).
* Per-scene perturbations: 0–5 false detections, 0–5 dropped real stars, and a
  per-point positional uncertainty between 0.25 and 1.0 px (Section 4.2.7).
* ``point_magnitude`` stores the catalogue magnitude; magnitude perturbation is
  disabled in the final dataset (hence the run name ``run5_expD_all_rawmag``).
* Seeds: split/training ``12345``; star-centred reference ``6353103531848264806``;
  ``synthD`` ``577215227560855758``.

**GNN (Section 4.3).**

* Regime R3: five graphs per scene, on the top-4 to top-8 brightest centroids.
* Fully connected graphs; ``edge_mlp`` backbone; 5 layers, hidden dimension 512,
  dropout 0.2.
* No node features; the single edge feature is the centroid distance normalised
  by the sensor diagonal (Section 4.3.5).
* AdamW, learning rate 1e-3, weight decay 1e-5, gradient clipping 1.0; early
  stopping on the validation loss with patience 40.

**Tetra3 + GNN integration (Section 4.4).**

* The GNN only generates attitude hypotheses; centroid extraction and the final
  geometric verification remain Tetra3's, so every accepted attitude is
  geometrically validated regardless of the candidates' origin.
* Anchor-pair search: M = 8 brightest centroids, N = 8 anchors, K = 10 candidates
  per anchor, B = 65 complete geometric verifications.

Results
-------

On the 147 real images, the hybrid pipeline matches the coverage of the classical
Tetra3 — both solve 132/147 — while, in the median, an image now needs a single
complete geometric verification instead of about 180, and the pipeline runs
roughly 2.3× faster overall, at the same attitude (Section 5.3). The per-image
comparison CSVs and the dataset coverage summaries behind these numbers are
committed under ``results/`` and can be re-checked without re-running the HPC
pipeline.

Repository layout
-----------------

``tetra3/``
    The classical Tetra3 implementation plus ``tetra4_GNN.py``, the integrated
    anchor-pair-search variant used in the thesis (Section 4.4).

``synth_dataset/``
    Synthetic dataset generation (Section 4.2). ``generate_dataset_baseline.py``
    builds the star-centred reference; ``generate_dataset_aletorio.py`` builds
    the final random-boresight ``synthD`` dataset. ``validation_real_vs_scene/``
    holds the real-image vs synthetic-scene fidelity check (Section 5.1.5).

``GNN/``
    GNN training, evaluation and the closed-set scene split (Section 4.3).

``scripts/``
    The reproduction entry point and the classical-vs-hybrid comparison on real
    images.

``results/``
    The small result files behind the thesis tables, verified against the
    dissertation (see ``results/README.md`` and ``ARTIFACTS.md``).

``experiments_archive/``
    Non-pipeline supporting material kept for traceability: HPC submission
    templates, analysis utilities and dataset-visualisation scripts
    (see ``experiments_archive/README.md``).

Real images and external artifacts
----------------------------------

The 147 real star-tracker images are proprietary and are not required for the
``smoke`` check. The multi-GB datasets, the trained checkpoints and the real
images are catalogued in ``ARTIFACTS.md``; the datasets are regenerable from the
recorded seeds. To run the real-image comparison directly from an existing
checkpoint:

.. code-block:: bash

   export CHECKPOINT=/path/to/best_checkpoint.pt
   export REAL_IMAGE_ROOT=/path/to/imgs_teste
   ./scripts/reproduce_thesis_pipeline.sh eval-real

Environment
-----------

The classical Tetra3 dependencies are in ``requirements.txt``; the GNN training
path additionally requires the packages listed in ``GNN/requirements.txt``. The
full dataset generation and training are HPC-scale, whereas the ``smoke``
pipeline is the recommended first check on a normal workstation.
