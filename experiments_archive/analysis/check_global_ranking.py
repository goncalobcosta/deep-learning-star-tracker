#!/usr/bin/env python3
"""
Parse eval_real_image.txt files and report which models produce
identical top-K predictions for all centroids (GLOBAL ranking)
vs per-centroid predictions.

Usage examples:

  # Parse all eval files under a plan root:
  python scripts/check_global_ranking.py --plan-root GNN/runs_testes/my_plan

  # Parse specific eval files:
  python scripts/check_global_ranking.py --eval-files path/to/T0_l3_h128/eval_real_image.txt ...

  # Run new evals on all checkpoints in a plan root (requires checkpoints):
  python scripts/check_global_ranking.py --plan-root GNN/runs_testes/my_plan --run-eval \
      --image imgs_extras/imgs_teste/1000ms_18-50/1000ms_18-50-26-712529.tiff \
      --brightest-k 8 --quad-combinations-top-n 8
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[1]


class CentroidPrediction(NamedTuple):
    centroid_rank: int
    top_k_ids: tuple[int, ...]


class ModelResult(NamedTuple):
    name: str
    train_dataset: str   # expD / expA / b360 / ?
    eval_image: str      # 1000ms / img3 / ?
    eval_path: Path
    node_feature_mode: str
    predictions: list[CentroidPrediction]  # one per centroid
    uniq_n: str       # e.g. "4/4" or "1/8"
    label: str        # GLOBAL / MISTO / PER-CENTRÓIDE


def parse_top_k_ids(line: str) -> tuple[int, ...]:
    """Extract catalog IDs from a prediction line like:
      01. star_id=123 catalog_id=456 prob=0.123456 tetra3_match=no
    """
    match = re.search(r"catalog_id=(\d+)", line)
    if match:
        return (int(match.group(1)),)
    return ()


def parse_eval_file(path: Path) -> list[CentroidPrediction]:
    """Parse eval_real_image.txt and return per-centroid top-K prediction lists."""
    text = path.read_text(encoding="utf-8", errors="replace")

    # Find topk used
    topk_match = re.search(r"summary_top(\d+)_hits", text)
    topk = int(topk_match.group(1)) if topk_match else 10

    predictions: list[CentroidPrediction] = []

    # Split by centroid block: each starts with "centroid_rank=N"
    # Pattern: centroid_rank=N y=... x=... flux=...
    #   (summary line)
    #   01. star_id=... catalog_id=... ...
    #   02. ...
    centroid_blocks = re.split(r"(?=^centroid_rank=)", text, flags=re.MULTILINE)

    for block in centroid_blocks:
        rank_match = re.match(r"centroid_rank=(\d+)", block)
        if not rank_match:
            continue
        centroid_rank = int(rank_match.group(1))

        # Extract all catalog IDs from numbered prediction lines
        id_lines = re.findall(r"^\s+\d+\. .*catalog_id=(\d+).*$", block, re.MULTILINE)
        ids = tuple(int(x) for x in id_lines[:topk])
        if ids:
            predictions.append(CentroidPrediction(centroid_rank=centroid_rank, top_k_ids=ids))

    return predictions


def classify_predictions(predictions: list[CentroidPrediction]) -> tuple[str, str]:
    """Return (uniq_n string, label) for a set of centroid predictions."""
    if not predictions:
        return "?", "UNKNOWN"

    n = len(predictions)
    unique_lists = {p.top_k_ids for p in predictions}
    n_unique = len(unique_lists)

    uniq_n = f"{n_unique}/{n}"

    if n_unique == 1:
        label = "GLOBAL"
    elif n_unique == n:
        label = "PER-CENTRÓIDE"
    else:
        label = "MISTO"

    return uniq_n, label


def infer_node_feature_mode(run_name: str, eval_text: str) -> str:
    """Try to guess node_feature_mode from run name or eval file content."""
    # Try to read from eval file header (if present)
    m = re.search(r"node_feature_mode=(\S+)", eval_text)
    if m:
        return m.group(1)

    # Fallback: infer from run name
    name = run_name.lower()
    if "sub_norm_max" in name:
        return "magnitude_subtracted_norm_max"
    if "sub_norm_med" in name:
        return "magnitude_subtracted_norm_median"
    if "subtracted" in name or "subtr" in name:
        return "magnitude_subtracted"
    if "norm_max" in name:
        return "magnitude_norm_max"
    if "norm_med" in name:
        return "magnitude_norm_median"
    if "rank_loss" in name or "t5" in name:
        return "magnitude_rank"
    if "rank" in name and "t4" in name:
        return "magnitude_rank"
    if "magnitude" in name or "mag" in name:
        return "magnitude"
    if "none" in name or "t0" in name or "t1" in name:
        return "none"
    return "?"


def infer_train_dataset(eval_path: Path) -> str:
    """Extract the training dataset name (expD / expA / b360) from the path."""
    parts = [p.name for p in eval_path.parents]
    for part in parts:
        if part.startswith("expD"):
            return "expD"
        if part.startswith("expA"):
            return "expA"
        if part.startswith("baseline360"):
            return "b360"
        if part.startswith("exp"):
            return part[:8]
    return "?"


def infer_eval_image(eval_path: Path, eval_text: str) -> str:
    """Extract the evaluation image label from the file content or path."""
    # First try reading the image= line written by eval_examples.py
    img_match = re.search(r"^image=(.+)$", eval_text, re.MULTILINE)
    if img_match:
        img_path = Path(img_match.group(1).strip())
        name = img_path.name
        if "1000ms" in name or "1000ms" in str(img_path):
            return "1000ms"
        if "obs016" in name or "img3" in name or "img_3" in name or "img_4" in name:
            return "img3"
        return img_path.stem[:16]
    # Fallback: look in the file name
    fname = eval_path.name
    if "1000ms" in fname:
        return "1000ms"
    if "img3" in fname:
        return "img3"
    # Fallback: look in parent directory names
    for part in [p.name for p in eval_path.parents]:
        if "1000ms" in part:
            return "1000ms"
        if "img3" in part or "obs016" in part:
            return "img3"
    return "?"


def run_eval(
    checkpoint: Path,
    image: Path,
    brightest_k: int,
    quad_top_n: int,
    quad_mode: str,
    topk: int,
    device: str,
) -> str | None:
    """Run eval_examples.py on a checkpoint and return the output text."""
    cmd = [
        sys.executable, "-u", "-m", "GNN.eval_examples",
        "--checkpoint", str(checkpoint),
        "--image", str(image),
        "--brightest-k", str(brightest_k),
        "--topk", str(topk),
        "--quad-combinations-top-n", str(quad_top_n),
        "--quad-combination-mode", quad_mode,
        "--device", device,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  [WARN] eval failed: {result.stderr[:200]}", file=sys.stderr)
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        print("  [WARN] eval timed out", file=sys.stderr)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check global vs per-centroid ranking across GNN models.")
    p.add_argument("--plan-root", type=Path, default=None, help="Root dir with one subdir per model run.")
    p.add_argument("--eval-files", type=Path, nargs="+", default=None, help="Explicit list of eval_real_image.txt files.")
    p.add_argument("--eval-glob", type=str, default="**/eval_real_image*.txt", help="Glob pattern relative to --plan-root.")

    p.add_argument("--run-eval", action="store_true", help="Re-run eval_examples.py for each checkpoint found.")
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--brightest-k", type=int, default=4)
    p.add_argument("--quad-combinations-top-n", type=int, default=8)
    p.add_argument("--quad-mode", choices=("all", "sample", "balanced_sample"), default="all")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--sort-by", choices=("name", "label", "node_feature"), default="name")
    return p.parse_args()


def collect_eval_files(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Return list of (model_name, eval_path)."""
    results: list[tuple[str, Path]] = []

    if args.eval_files:
        for f in args.eval_files:
            f = f.expanduser().resolve()
            if f.exists():
                results.append((f.parent.name, f))
        return results

    if args.plan_root:
        plan_root = args.plan_root.expanduser().resolve()

        if args.run_eval:
            if args.image is None:
                print("[ERROR] --image is required with --run-eval", file=sys.stderr)
                sys.exit(1)
            image = args.image.expanduser().resolve()
            for checkpoint in sorted(plan_root.glob("**/best_checkpoint.pt")):
                run_name = checkpoint.parent.name
                eval_out = checkpoint.parent / f"eval_real_image_k{args.brightest_k}.txt"
                print(f"Running eval for {run_name}...", flush=True)
                text = run_eval(
                    checkpoint, image,
                    args.brightest_k,
                    args.quad_combinations_top_n,
                    args.quad_mode,
                    args.topk,
                    args.device,
                )
                if text:
                    eval_out.write_text(text, encoding="utf-8")
                    results.append((run_name, eval_out))
            return results

        # Just find existing eval files
        for eval_path in sorted(plan_root.glob(args.eval_glob)):
            run_name = eval_path.parent.name
            results.append((run_name, eval_path))
        return results

    print("[ERROR] Provide --plan-root or --eval-files", file=sys.stderr)
    sys.exit(1)


def print_table(rows: list[ModelResult]) -> None:
    col_train = max(len(r.train_dataset) for r in rows)
    col_img = max(len(r.eval_image) for r in rows)
    col_name = max(len(r.name) for r in rows)
    col_node = max(len(r.node_feature_mode) for r in rows)
    col_uniq = max(len(r.uniq_n) for r in rows)
    col_label = max(len(r.label) for r in rows)

    fmt = (f"{{:<{col_train}}}  {{:<{col_img}}}  {{:<{col_name}}}  "
           f"{{:<{col_node}}}  {{:>{col_uniq}}}  {{:<{col_label}}}")
    header = fmt.format("Train dataset", "Eval image", "Modelo", "Node feature", "uniq/n", "Tipo")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(fmt.format(r.train_dataset, r.eval_image, r.name, r.node_feature_mode, r.uniq_n, r.label))


def main() -> int:
    args = parse_args()
    entries = collect_eval_files(args)

    if not entries:
        print("No eval files found.", file=sys.stderr)
        return 1

    rows: list[ModelResult] = []
    for run_name, eval_path in entries:
        try:
            text = eval_path.read_text(encoding="utf-8", errors="replace")
            predictions = parse_eval_file(eval_path)
            if not predictions:
                print(f"  [WARN] No predictions parsed from {eval_path}", file=sys.stderr)
                continue
            uniq_n, label = classify_predictions(predictions)
            node_feature_mode = infer_node_feature_mode(run_name, text)
            rows.append(ModelResult(
                name=run_name,
                train_dataset=infer_train_dataset(eval_path),
                eval_image=infer_eval_image(eval_path, text),
                eval_path=eval_path,
                node_feature_mode=node_feature_mode,
                predictions=predictions,
                uniq_n=uniq_n,
                label=label,
            ))
        except Exception as exc:
            print(f"  [ERROR] {eval_path}: {exc}", file=sys.stderr)

    if not rows:
        print("No results to display.", file=sys.stderr)
        return 1

    if args.sort_by == "label":
        rows.sort(key=lambda r: (r.label, r.name))
    elif args.sort_by == "node_feature":
        rows.sort(key=lambda r: (r.node_feature_mode, r.name))
    else:
        rows.sort(key=lambda r: r.name)

    print_table(rows)

    # Summary counts
    counts = {"GLOBAL": 0, "MISTO": 0, "PER-CENTRÓIDE": 0}
    for r in rows:
        counts[r.label] = counts.get(r.label, 0) + 1
    print()
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
