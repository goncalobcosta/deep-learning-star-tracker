#!/usr/bin/env python3
"""
Count how many times each star appears as a node in the training quads,
and compare with the top-K predictions of GLOBAL models.

Usage:
  python scripts/analyze_star_frequency.py \
      --dataset-dir /path/to/dataset \
      --eval-file GNN/runs_testes/my_plan/T0_l3_h128/eval_real_image.txt \
      [--split train]  \
      [--top-n 20]

  # Or just count frequencies without comparing eval:
  python scripts/analyze_star_frequency.py --dataset-dir /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_manifest(dataset_dir: Path) -> dict:
    with (dataset_dir / "dataset_manifest.json").open("r", encoding="utf-8") as fp:
        return json.load(fp)


def chunk_paths(dataset_dir: Path, manifest: dict) -> list[Path]:
    return [(dataset_dir / item["path"]).resolve() for item in manifest.get("chunks", [])]


def load_split_refs(split_npz: Path, split_name: str) -> list[tuple[int, int]]:
    with np.load(split_npz, allow_pickle=False) as d:
        prefix = split_name
        chunk_idx = d[f"{prefix}_chunk_idx"].astype(np.int64)
        scene_idx = d[f"{prefix}_scene_idx"].astype(np.int64)
    return list(zip(chunk_idx.tolist(), scene_idx.tolist()))


def count_star_appearances(
    dataset_dir: Path,
    split_name: str = "train",
    source_top_n: int = 4,
) -> Counter:
    """Return Counter of how many times each star_id appears as a node in training quads."""
    manifest = load_manifest(dataset_dir)
    chunks = chunk_paths(dataset_dir, manifest)

    split_npz = dataset_dir / "split.npz"
    if not split_npz.exists():
        print(f"[WARN] split.npz not found at {split_npz}; scanning all chunks.", file=sys.stderr)
        refs = None
    else:
        refs = load_split_refs(split_npz, split_name)

    counter: Counter = Counter()

    if refs is not None:
        # Group by chunk to avoid re-loading
        from collections import defaultdict
        by_chunk: dict[int, list[int]] = defaultdict(list)
        for ci, si in refs:
            by_chunk[ci].append(si)

        for chunk_i, scene_indices in sorted(by_chunk.items()):
            chunk_path = chunks[chunk_i]
            with np.load(chunk_path, allow_pickle=False) as data:
                point_star_id = data["point_star_id"].astype(np.int64)  # (N_scenes * scene_size,)
                scene_size = data.get("scene_size", np.array(source_top_n))[()]

            for si in scene_indices:
                start = int(si) * int(scene_size)
                ids = point_star_id[start : start + source_top_n]
                counter.update(ids.tolist())
    else:
        for chunk_path in chunks:
            with np.load(chunk_path, allow_pickle=False) as data:
                point_star_id = data["point_star_id"].astype(np.int64)
                scene_size = int(data.get("scene_size", np.array(source_top_n))[()])
            n_scenes = len(point_star_id) // scene_size
            for si in range(n_scenes):
                start = si * scene_size
                ids = point_star_id[start : start + source_top_n]
                counter.update(ids.tolist())

    return counter


def parse_global_top10(eval_path: Path) -> list[int]:
    """Extract the (common) top-10 catalog IDs from a GLOBAL eval file (first centroid)."""
    text = eval_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?=^centroid_rank=)", text, flags=re.MULTILINE)
    for block in blocks:
        if not re.match(r"centroid_rank=\d+", block):
            continue
        ids = re.findall(r"catalog_id=(\d+)", block)
        if ids:
            return [int(x) for x in ids[:10]]
    return []


def print_frequency_table(counter: Counter, top_n: int, global_top10: list[int]) -> None:
    total = sum(counter.values())
    n_classes = len(counter)
    avg = total / n_classes if n_classes else 0
    expected_uniform = total / n_classes if n_classes else 0

    global_set = set(global_top10)

    print(f"\n{'─'*70}")
    print(f"Total appearances: {total:,}  |  Unique stars: {n_classes}  |  Avg/star: {avg:.1f}")
    print(f"Uniform baseline: {expected_uniform:.1f} appearances/star  ({100/n_classes:.3f}% each)")
    print(f"{'─'*70}")
    print(f"{'Rank':<6} {'star_id':<12} {'appearances':>12} {'%':>8} {'vs_avg':>8}  {'in GLOBAL top10'}")
    print(f"{'─'*70}")

    for rank, (star_id, count) in enumerate(counter.most_common(top_n), 1):
        pct = 100 * count / total
        vs_avg = count / avg if avg > 0 else 0
        flag = "  ← PREDICTED" if star_id in global_set else ""
        print(f"{rank:<6} {star_id:<12} {count:>12,} {pct:>7.3f}% {vs_avg:>7.2f}x{flag}")

    # How many of the global top-10 are in the top-N most frequent?
    if global_top10:
        top_n_ids = {star_id for star_id, _ in counter.most_common(top_n)}
        overlap = sum(1 for sid in global_top10 if sid in top_n_ids)
        print(f"\nGLOBAL top10 overlap with training top-{top_n}: {overlap}/10")

        print(f"\n{'─'*50}")
        print("Frequency of the GLOBAL top-10 predicted stars:")
        print(f"{'Rank':<6} {'catalog_id':<12} {'appearances':>12} {'%':>8} {'vs_avg':>8}")
        print(f"{'─'*50}")
        all_ids = list(counter.keys())
        sorted_ids = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
        rank_lookup = {sid: i+1 for i, sid in enumerate(sorted_ids)}
        for catalog_id in global_top10:
            count = counter.get(catalog_id, 0)
            pct = 100 * count / total if total else 0
            vs_avg = count / avg if avg > 0 else 0
            freq_rank = rank_lookup.get(catalog_id, "?")
            print(f"{freq_rank!s:<6} {catalog_id:<12} {count:>12,} {pct:>7.3f}% {vs_avg:>7.2f}x")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze star appearance frequency in training data.")
    p.add_argument("--dataset-dir", type=Path, required=True, help="Path to dataset directory with dataset_manifest.json")
    p.add_argument("--eval-file", type=Path, default=None, help="eval_real_image.txt from a GLOBAL model to compare")
    p.add_argument("--split", default="train", help="Split to analyse (train/val/test)")
    p.add_argument("--top-n", type=int, default=20, help="Show top-N most frequent stars")
    p.add_argument("--source-top-n", type=int, default=4, help="Nodes per quad (default 4)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    print(f"Counting star appearances in '{args.split}' split of {dataset_dir.name} ...")
    counter = count_star_appearances(dataset_dir, args.split, args.source_top_n)

    global_top10: list[int] = []
    if args.eval_file:
        eval_path = args.eval_file.expanduser().resolve()
        if eval_path.exists():
            global_top10 = parse_global_top10(eval_path)
            print(f"GLOBAL top10 from eval: {global_top10}")
        else:
            print(f"[WARN] eval file not found: {eval_path}", file=sys.stderr)

    print_frequency_table(counter, args.top_n, global_top10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
