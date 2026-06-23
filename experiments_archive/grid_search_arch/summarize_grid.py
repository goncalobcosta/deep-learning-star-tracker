#!/usr/bin/env python3
"""Summarize architecture grid-search runs from train_history.jsonl."""

from __future__ import annotations

import json
from pathlib import Path


RUNS = (
    "grid_h128_l3",
    "grid_h192_l3",
    "grid_h128_l4",
    "grid_h192_l4",
)


def best_by_metric(rows: list[dict], metric: str) -> tuple[int, float]:
    best_row = max(rows, key=lambda row: float(row.get(metric, float("-inf"))))
    return int(best_row["epoch"]), float(best_row[metric])


def main() -> int:
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    print(
        "run,"
        "best_val_top1_epoch,best_val_top1_real,"
        "best_val_top5_epoch,best_val_top5_real,"
        "best_val_top10_epoch,best_val_top10_real,"
        "test_top1_real,test_top5_real,test_top10_real"
    )
    for run_name in RUNS:
        history_path = runs_root / run_name / "train_history.jsonl"
        summary_path = runs_root / run_name / "train_summary.json"
        if not history_path.exists():
            print(f"{run_name},missing,missing,missing,missing,missing,missing,missing,missing,missing")
            continue

        rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            print(f"{run_name},empty,empty,empty,empty,empty,empty,empty,empty,empty")
            continue

        top1_epoch, top1_value = best_by_metric(rows, "val_top1_real")
        top5_epoch, top5_value = best_by_metric(rows, "val_top5_real")
        top10_epoch, top10_value = best_by_metric(rows, "val_top10_real")
        test_top1 = float("nan")
        test_top5 = float("nan")
        test_top10 = float("nan")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            test_metrics = summary.get("test_metrics", {})
            if isinstance(test_metrics, dict):
                test_top1 = float(test_metrics.get("top1_real", float("nan")))
                test_top5 = float(test_metrics.get("top5_real", float("nan")))
                test_top10 = float(test_metrics.get("top10_real", float("nan")))
        print(
            f"{run_name},"
            f"{top1_epoch},{top1_value:.4f},"
            f"{top5_epoch},{top5_value:.4f},"
            f"{top10_epoch},{top10_value:.4f},"
            f"{test_top1:.4f},{test_top5:.4f},{test_top10:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
