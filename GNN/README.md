# GNN candidate generator

This folder holds the graph neural network that replaces Tetra3's
candidate-generation stage (Sections 4.3 and 4.4 of the dissertation).
Identification is cast as a node-classification task: once an image is reduced to
a set of detected centroids, those centroids and their geometric relationships
form a graph, and the GNN assigns each node an ordered list of catalogue
candidates. The network is not the final identifier — the geometric verification
of Tetra3 remains the acceptance criterion; the GNN only makes the candidate
search more selective.

## Final model (regime R3)

The configuration selected in Chapter 5 and recorded in
`configs/final_r3_synthd.yaml`:

- input: five graphs per scene, on the top-4 to top-8 brightest centroids
  (`--top-n-choices 4,5,6,7,8 --top-n-mode expand`);
- fully connected graphs (`--graph-connectivity fully`);
- backbone: a single concatenation aggregator followed by a multilayer
  perceptron (`--model-backbone edge_mlp`), 5 layers, hidden dimension 512,
  dropout 0.2 (Section 4.3.6);
- node features: none; edge feature: centroid distance normalised by the sensor
  diagonal (`--node-feature-mode none --edge-feature-mode distance_diagonal`);
- AdamW, `lr=1e-3`, `weight_decay=1e-5`, gradient clipping `1.0`;
- early stopping on `val_loss` with patience `40`; seed `12345`.

The cross-entropy loss is computed only over the real nodes; false detections
are kept in the graph for geometric context but carry no class (Section 4.3.8).

## Files

- `GNN.py`: training and evaluation entry point.
- `split/split.py`: builds the closed-set scene split inside each `guide_star`,
  so the model is evaluated and tested on scenes disjoint from training
  (Section 4.3.9).
- `eval_examples.py`: runs a trained checkpoint on a single real image and
  reports the top-k identity ranks of each centroid against the Tetra3 solution.

The recommended way to run the whole sequence is the root script
`./scripts/reproduce_thesis_pipeline.sh`; the commands below document the
individual steps it wires together. They assume a `synthD` run directory (on
Deucalion, `run5_expD_all_rawmag`; locally, the directory the generator writes).

## Closed-set split

```bash
python3 -m GNN.split.split \
  --dataset-dir <synthD-run-dir> \
  --seed 12345 \
  --output GNN/split/runs/final_r3_synthd_seed12345.npz
```

## Training

```bash
python3 -u -m GNN.GNN \
  --dataset-dir <synthD-run-dir> \
  --split-file GNN/split/runs/final_r3_synthd_seed12345.npz \
  --run-name final_r3_synthd \
  --epochs 100 --batch-size-scenes 2048 \
  --top-n-choices 4,5,6,7,8 --top-n-mode expand \
  --graph-connectivity fully \
  --node-feature-mode none --edge-feature-mode distance_diagonal \
  --model-backbone edge_mlp --hidden-dim 512 --num-layers 5 --dropout 0.2 \
  --lr 1e-3 --weight-decay 1e-5 --grad-clip-norm 1.0 \
  --early-stop-monitor val_loss --early-stop-patience 40 \
  --loss-group-by-scene --seed 12345 --device cuda
```

Each run writes `best_checkpoint.pt`, `last_checkpoint.pt`, `train_history.jsonl`
and `train_summary.json`. The thesis result uses the `synthD` checkpoint at
epoch 14 (`ARTIFACTS.md`).

## Synthetic evaluation

Add `--eval-only --checkpoint <best_checkpoint.pt>` to the same command to report
the top-1, top-5 and top-10 metrics over the real nodes of the test split.

## Node-level evaluation on a real image

```bash
python3 -u GNN/eval_examples.py \
  --checkpoint GNN/runs/final_r3_synthd/best_checkpoint.pt \
  --image /path/to/image.tiff \
  --top-n-choices 4,5,6,7,8 --graph-connectivity fully \
  --node-feature-mode none --edge-feature-mode distance_diagonal \
  --brightest-k 8 --topk 10 --device cuda
```

This evaluates the candidates per centroid against the Tetra3 reference, as in
Section 4.3.12. The full classical-vs-hybrid attitude comparison on a folder of
real images is run instead through `scripts/compare_tetra_vs_gnn_batch.py`.
