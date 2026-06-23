#!/usr/bin/env python3
"""Run the staged GNN tests and evaluate each checkpoint on one real image."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent if (SCRIPT_PATH.parent / "GNN").exists() else SCRIPT_PATH.parents[1]
DEFAULT_REAL_IMAGE = REPO_ROOT / "imgs_extras" / "imgs_teste" / "1000ms_18-50" / "1000ms_18-50-26-712529.tiff"
DEFAULT_SPLIT = REPO_ROOT / "GNN" / "split" / "runs" / "run_1000ms_18-50_expd" / "guide_split_seed12345.npz"


@dataclass(frozen=True)
class RunSpec:
    graph_regime: str
    quad_mode: str
    phase: str
    name: str
    test_id: int
    hidden_dim: int
    num_layers: int
    node_feature_mode: str
    edge_feature_mode: str
    extra_args: tuple[str, ...] = ()


@dataclass
class RunResult:
    spec: RunSpec
    run_dir: Path
    summary_path: Path
    eval_path: Path | None
    metric: float
    train_wall_min: float
    eval_wall_min: float | None
    summary: dict[str, Any]
    eval_metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged GNN tests for the professor baseline plan.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Synthetic dataset run directory.")
    parser.add_argument("--split-file", type=Path, default=None, help="Closed-set split .npz file.")
    parser.add_argument("--real-image", type=Path, default=None, help="Real TIFF image for eval_examples.py.")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "GNN" / "runs_testes")
    parser.add_argument("--plan-name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size-scenes", type=int, default=256)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader workers. Default: auto-detect from scheduler env vars "
            "such as SLURM_CPUS_PER_TASK, then fall back to os.cpu_count()."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--cache-chunks", type=int, default=2)
    parser.add_argument("--log-every-batches", type=int, default=200)
    parser.add_argument("--worker-timeout-sec", type=int, default=90)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-monitor", type=str, default="val_loss")

    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--quad-top-n", type=int, default=8)
    parser.add_argument("--quad-mode", choices=("balanced_sample", "sample", "all"), default="balanced_sample")
    parser.add_argument(
        "--graph-regimes",
        choices=("balanced", "all", "both"),
        default="both",
        help=(
            "Which graph input regimes to run: balanced=one balanced 4-star graph per scene; "
            "all=all C(topN,4) graphs per scene; both=runs the staged plan for both regimes."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "val_loss",
            "val_top10_real",
            "val_top5_real",
            "val_top1_real",
        ),
        default="val_loss",
    )
    parser.add_argument("--only-test0", action="store_true", help="Run only the TestID 0 architecture sweep.")
    parser.add_argument(
        "--test0-hidden-dims",
        type=str,
        default="128,256",
        help="Comma-separated hidden dimensions for TestID 0, for example 128,256,512.",
    )

    parser.add_argument("--real-brightest-k", type=int, default=8)
    parser.add_argument("--real-topk", type=int, default=10)
    parser.add_argument("--skip-real-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--deucalion-submit", action="store_true", help="Write and submit a Deucalion Slurm job for this run.")
    parser.add_argument("--deucalion-job-file", type=Path, default=Path("run_testes_gnn.generated.sh"))
    parser.add_argument("--deucalion-account", type=str, default="f202603931cpcaa0g")
    parser.add_argument("--deucalion-qos", type=str, default="normal")
    parser.add_argument("--deucalion-partition", type=str, default="normal-a100-40")
    parser.add_argument("--deucalion-gres", type=str, default="gpu:a100:1")
    parser.add_argument("--deucalion-time", type=str, default="24:00:00")
    parser.add_argument("--deucalion-cpus", type=int, default=16)
    parser.add_argument("--deucalion-mem", type=str, default="64G")
    parser.add_argument("--deucalion-python-module", type=str, default="Python/3.11.5-GCCcore-13.2.0")
    parser.add_argument("--deucalion-venv", type=str, default=".venv-gnn")
    return parser.parse_args()


def resolve_num_workers(raw: int | None) -> int:
    if raw is not None:
        return max(0, int(raw))

    import os

    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "NSLOTS", "PBS_NP"):
        value = os.environ.get(env_name)
        if value:
            try:
                return max(0, int(value))
            except ValueError:
                continue
    return max(1, os.cpu_count() or 1)


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def lower_is_better_metric(metric_name: str) -> bool:
    return metric_name.endswith("_loss") or metric_name == "val_loss"


def better_metric_value(values: list[float], metric_name: str) -> float:
    if not values:
        return 0.0
    return min(values) if lower_is_better_metric(metric_name) else max(values)


def resolve_split_file(raw: Path | None) -> Path:
    if raw is not None:
        return raw.expanduser().resolve()
    if DEFAULT_SPLIT.exists():
        return DEFAULT_SPLIT.resolve()
    candidates = sorted((REPO_ROOT / "GNN" / "split" / "runs").glob("**/guide_split_seed*.npz"))
    if candidates:
        return candidates[-1].resolve()
    raise FileNotFoundError("No split file found. Pass --split-file explicitly.")


def resolve_real_image(raw: Path | None) -> Path | None:
    if raw is not None:
        return raw.expanduser().resolve()
    if DEFAULT_REAL_IMAGE.exists():
        return DEFAULT_REAL_IMAGE.resolve()
    return None


def run_command(cmd: list[str], *, cwd: Path, log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_text = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(cmd_text, flush=True)
    log_path.write_text(cmd_text + "\n\n", encoding="utf-8")
    if dry_run:
        return

    with log_path.open("a", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_fp.write(line)
            log_fp.flush()
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def metric_from_summary(summary: dict[str, Any], metric_name: str) -> float:
    if metric_name == "val_loss" and "best_val_loss" in summary:
        return float(summary["best_val_loss"])
    if metric_name == "val_top1_real" and "best_val_top1_real" in summary:
        return float(summary["best_val_top1_real"])
    if metric_name == "val_top5_real" and "best_val_top5_real" in summary:
        return float(summary["best_val_top5_real"])
    if metric_name == "val_top10_real" and "best_val_top10_real" in summary:
        return float(summary["best_val_top10_real"])
    if summary.get("best_monitor_metric") == metric_name and "best_monitor_value" in summary:
        return float(summary["best_monitor_value"])
    history = summary.get("history")
    if isinstance(history, list) and history:
        values = [float(row.get(metric_name, 0.0)) for row in history if isinstance(row, dict)]
        if values:
            return better_metric_value(values, metric_name)
    files = summary.get("files", {})
    history_path = files.get("history") if isinstance(files, dict) else None
    if isinstance(history_path, str) and history_path:
        path = Path(history_path)
        if path.exists():
            values = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and metric_name in row:
                    values.append(float(row[metric_name]))
            if values:
                return better_metric_value(values, metric_name)
    test_metrics = summary.get("test_metrics", {})
    if isinstance(test_metrics, dict) and metric_name in test_metrics:
        return float(test_metrics[metric_name])
    return 0.0


def nested_metric(summary: dict[str, Any], split_name: str, metric_name: str) -> float | None:
    metrics = summary.get(f"{split_name}_metrics")
    if isinstance(metrics, dict) and metric_name in metrics:
        return float(metrics[metric_name])
    history = summary.get("history")
    if split_name == "val" and isinstance(history, list) and history:
        values = [float(row.get(f"val_{metric_name}", 0.0)) for row in history if isinstance(row, dict)]
        if values:
            return better_metric_value(values, f"val_{metric_name}")
    files = summary.get("files", {})
    history_path = files.get("history") if isinstance(files, dict) else None
    if split_name == "val" and isinstance(history_path, str) and history_path:
        path = Path(history_path)
        if path.exists():
            values = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = f"val_{metric_name}"
                if isinstance(row, dict) and key in row:
                    values.append(float(row[key]))
            if values:
                return better_metric_value(values, f"val_{metric_name}")
    return None


def parse_real_eval_metrics(eval_path: Path | None, brightest_k: int) -> dict[str, Any]:
    empty = {
        "real_top10_any_hits": "",
        "real_top10_any_total": "",
        "real_top10_exact_hits": "",
        "real_top10_exact_total": "",
        "real_ref_count": "",
    }
    for idx in range(1, int(brightest_k) + 1):
        empty[f"real_c{idx}_rank"] = ""

    if eval_path is None or not eval_path.exists():
        return empty

    text = eval_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = dict(empty)

    any_match = re.search(r"summary_top\d+_hits_any_tetra3_match_among_brightest_\d+=(\d+)/(\d+)", text)
    if any_match:
        out["real_top10_any_hits"] = int(any_match.group(1))
        out["real_top10_any_total"] = int(any_match.group(2))

    exact_match = re.search(r"summary_top\d+_hits_nearest_tetra_match_among_brightest_\d+=(\d+)/(\d+)", text)
    if exact_match:
        out["real_top10_exact_hits"] = int(exact_match.group(1))
        out["real_top10_exact_total"] = int(exact_match.group(2))

    ref_count = 0
    block_pattern = re.compile(
        r"centroid_rank=(\d+).*?\n"
        r"\s+top\d+_hit_any_tetra3_match=(yes|no) "
        r"matched_catID_nearest=([^ ]+) "
        r"top\d+_hit_nearest_tetra_match=(yes|no) "
        r"nearest_match_rank=([^\s]+)"
    )
    for match in block_pattern.finditer(text):
        centroid_idx = int(match.group(1))
        if centroid_idx < 1 or centroid_idx > int(brightest_k):
            continue
        nearest = str(match.group(3))
        exact = str(match.group(4)) == "yes"
        rank = str(match.group(5))
        if nearest == "none":
            out[f"real_c{centroid_idx}_rank"] = "n/a"
        else:
            ref_count += 1
            out[f"real_c{centroid_idx}_rank"] = rank if exact and rank != "none" else "-"
    out["real_ref_count"] = int(ref_count)
    return out


def train_one(
    spec: RunSpec,
    *,
    args: argparse.Namespace,
    num_workers: int,
    split_file: Path,
    plan_root: Path,
    real_image: Path | None,
) -> RunResult:
    run_dir = plan_root / spec.name
    summary_path = run_dir / "train_summary.json"
    command_log = run_dir / "command_train.log"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "GNN.GNN",
        "--split-file",
        str(split_file),
        "--runs-root",
        str(plan_root),
        "--run-name",
        spec.name,
        "--epochs",
        str(int(args.epochs)),
        "--batch-size-scenes",
        str(int(args.batch_size_scenes)),
        "--num-workers",
        str(int(num_workers)),
        "--cache-chunks",
        str(int(args.cache_chunks)),
        "--log-every-batches",
        str(int(args.log_every_batches)),
        "--worker-timeout-sec",
        str(int(args.worker_timeout_sec)),
        "--early-stop-patience",
        str(int(args.early_stop_patience)),
        "--early-stop-min-delta",
        str(float(args.early_stop_min_delta)),
        "--early-stop-monitor",
        str(args.early_stop_monitor),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--top-n-choices",
        "4",
        "--top-n-mode",
        "max",
        "--graph-connectivity",
        "fully",
        "--node-feature-mode",
        spec.node_feature_mode,
        "--edge-feature-mode",
        spec.edge_feature_mode,
        "--quad-combinations-top-n",
        str(int(args.quad_top_n)),
        "--quad-combination-mode",
        str(spec.quad_mode),
        "--hidden-dim",
        str(int(spec.hidden_dim)),
        "--num-layers",
        str(int(spec.num_layers)),
        "--heads",
        str(int(args.heads)),
        "--dropout",
        str(float(args.dropout)),
    ]
    cmd.extend(spec.extra_args)
    if args.dataset_dir is not None:
        cmd.extend(["--dataset-dir", str(args.dataset_dir.expanduser().resolve())])
    if spec.quad_mode == "all":
        cmd.append("--loss-group-by-scene")

    train_started = time.perf_counter()
    run_command(cmd, cwd=REPO_ROOT, log_path=command_log, dry_run=bool(args.dry_run))
    train_wall_min = float((time.perf_counter() - train_started) / 60.0)

    summary: dict[str, Any] = {}
    metric = 0.0
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metric = metric_from_summary(summary, str(args.selection_metric))

    eval_path = None
    eval_wall_min = None
    if not args.skip_real_eval and real_image is not None:
        eval_path = run_dir / "eval_real_image.txt"
        eval_cmd = [
            sys.executable,
            "-u",
            "-m",
            "GNN.eval_examples",
            "--checkpoint",
            str(run_dir / "best_checkpoint.pt"),
            "--image",
            str(real_image),
            "--device",
            str(args.device),
            "--quad-combinations-top-n",
            str(int(args.quad_top_n)),
            "--quad-combination-mode",
            "all",
            "--brightest-k",
            str(int(args.real_brightest_k)),
            "--topk",
            str(int(args.real_topk)),
        ]
        eval_started = time.perf_counter()
        run_command(eval_cmd, cwd=REPO_ROOT, log_path=eval_path, dry_run=bool(args.dry_run))
        eval_wall_min = float((time.perf_counter() - eval_started) / 60.0)
    eval_metrics = parse_real_eval_metrics(eval_path, int(args.real_brightest_k))

    return RunResult(
        spec=spec,
        run_dir=run_dir,
        summary_path=summary_path,
        eval_path=eval_path,
        metric=metric,
        train_wall_min=train_wall_min,
        eval_wall_min=eval_wall_min,
        summary=summary,
        eval_metrics=eval_metrics,
    )


def choose_best(results: list[RunResult], metric_name: str) -> RunResult:
    if lower_is_better_metric(metric_name):
        return min(results, key=lambda item: item.metric)
    return max(results, key=lambda item: item.metric)


def dmag_edge_mode(edge_mode: str) -> str:
    if edge_mode.endswith("_dmag"):
        return edge_mode
    return f"{edge_mode}_dmag"


def write_results_csv(
    plan_root: Path,
    results: list[RunResult],
    final_best: RunResult | None,
    selection_metric_name: str,
) -> None:
    out = plan_root / "correr_testes_summary.csv"
    real_rank_cols = sorted(
        {
            key
            for result in results
            for key in result.eval_metrics.keys()
            if re.fullmatch(r"real_c\d+_rank", str(key))
        },
        key=lambda item: int(re.search(r"\d+", item).group(0)) if re.search(r"\d+", item) else 0,
    )

    def fmt_optional(value: float | None) -> str:
        return "" if value is None else f"{float(value):.8f}"

    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "graph_regime",
                "quad_mode",
                "graphs_per_scene",
                "phase",
                "test_id",
                "run_name",
                "selection_metric_name",
                "selection_metric_value",
                "val_loss",
                "val_top1_real",
                "val_top5_real",
                "val_top10_real",
                "test_top1_real",
                "test_top5_real",
                "test_top10_real",
                "hidden_dim",
                "num_layers",
                "node_feature_mode",
                "edge_feature_mode",
                "train_wall_min",
                "train_summary_duration_min",
                "min_per_epoch",
                "eval_wall_min",
                "real_top10_any_hits",
                "real_top10_any_total",
                "real_top10_exact_hits",
                "real_top10_exact_total",
                "real_ref_count",
                *real_rank_cols,
                "run_dir",
                "summary_path",
                "eval_real_image",
                "is_final_best",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.spec.graph_regime,
                    result.spec.quad_mode,
                    "70" if result.spec.quad_mode == "all" else "1",
                    result.spec.phase,
                    result.spec.test_id,
                    result.spec.name,
                    selection_metric_name,
                    f"{result.metric:.8f}",
                    fmt_optional(nested_metric(result.summary, "val", "loss")),
                    fmt_optional(nested_metric(result.summary, "val", "top1_real")),
                    fmt_optional(nested_metric(result.summary, "val", "top5_real")),
                    fmt_optional(nested_metric(result.summary, "val", "top10_real")),
                    fmt_optional(nested_metric(result.summary, "test", "top1_real")),
                    fmt_optional(nested_metric(result.summary, "test", "top5_real")),
                    fmt_optional(nested_metric(result.summary, "test", "top10_real")),
                    result.spec.hidden_dim,
                    result.spec.num_layers,
                    result.spec.node_feature_mode,
                    result.spec.edge_feature_mode,
                    f"{result.train_wall_min:.3f}",
                    f"{float(result.summary.get('duration_min', 0.0)):.3f}" if result.summary else "",
                    f"{float(result.summary.get('min_per_epoch', 0.0)):.3f}" if result.summary else "",
                    "" if result.eval_wall_min is None else f"{result.eval_wall_min:.3f}",
                    result.eval_metrics.get("real_top10_any_hits", ""),
                    result.eval_metrics.get("real_top10_any_total", ""),
                    result.eval_metrics.get("real_top10_exact_hits", ""),
                    result.eval_metrics.get("real_top10_exact_total", ""),
                    result.eval_metrics.get("real_ref_count", ""),
                    *[result.eval_metrics.get(col, "") for col in real_rank_cols],
                    str(result.run_dir),
                    str(result.summary_path),
                    "" if result.eval_path is None else str(result.eval_path),
                    "yes" if final_best is not None and result.spec.name == final_best.spec.name else "no",
                ]
            )
    print(f"Summary CSV: {out}")


def quoted_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def deucalion_python_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        "-u",
        "scripts/correr_testes.py",
        "--epochs",
        str(int(args.epochs)),
        "--batch-size-scenes",
        str(int(args.batch_size_scenes)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--cache-chunks",
        str(int(args.cache_chunks)),
        "--log-every-batches",
        str(int(args.log_every_batches)),
        "--worker-timeout-sec",
        str(int(args.worker_timeout_sec)),
        "--early-stop-patience",
        str(int(args.early_stop_patience)),
        "--early-stop-min-delta",
        str(float(args.early_stop_min_delta)),
        "--early-stop-monitor",
        str(args.early_stop_monitor),
        "--heads",
        str(int(args.heads)),
        "--dropout",
        str(float(args.dropout)),
        "--quad-top-n",
        str(int(args.quad_top_n)),
        "--quad-mode",
        str(args.quad_mode),
        "--graph-regimes",
        str(args.graph_regimes),
        "--selection-metric",
        str(args.selection_metric),
        "--test0-hidden-dims",
        str(args.test0_hidden_dims),
        "--real-brightest-k",
        str(int(args.real_brightest_k)),
        "--real-topk",
        str(int(args.real_topk)),
    ]
    if args.dataset_dir is not None:
        cmd += ["--dataset-dir", str(args.dataset_dir)]
    if args.split_file is not None:
        cmd += ["--split-file", str(args.split_file)]
    if args.real_image is not None:
        cmd += ["--real-image", str(args.real_image)]
    if args.runs_root is not None:
        cmd += ["--runs-root", str(args.runs_root)]
    if args.plan_name:
        cmd += ["--plan-name", str(args.plan_name)]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(int(args.num_workers))]
    if args.skip_real_eval:
        cmd.append("--skip-real-eval")
    if args.only_test0:
        cmd.append("--only-test0")
    return cmd


def write_deucalion_job(args: argparse.Namespace) -> Path:
    job_path = args.deucalion_job_file.expanduser().resolve()
    gres_line = f"#SBATCH --gres={args.deucalion_gres}\n" if str(args.deucalion_gres).strip() else ""
    cuda_check = ""
    if str(args.device).startswith("cuda"):
        cuda_check = """
if not torch.cuda.is_available():
    sys.exit("CUDA is not available inside the Slurm job")
print("gpu", torch.cuda.get_device_name(0))
"""
    content = f"""#!/bin/bash
#SBATCH --qos={args.deucalion_qos}
#SBATCH --account={args.deucalion_account}
#SBATCH --job-name=gnn_testes
#SBATCH --output=logs/gnn_testes_%j.out
#SBATCH --error=logs/gnn_testes_%j.err
#SBATCH --time={args.deucalion_time}
#SBATCH --partition={args.deucalion_partition}
{gres_line.rstrip()}
#SBATCH --cpus-per-task={int(args.deucalion_cpus)}
#SBATCH --mem={args.deucalion_mem}

set -euo pipefail

cd {shlex.quote(str(REPO_ROOT))}
mkdir -p logs

module purge
module load {shlex.quote(args.deucalion_python_module)}
source {shlex.quote(args.deucalion_venv)}/bin/activate

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-}}"
echo "SLURM_JOB_GPUS=${{SLURM_JOB_GPUS:-}}"
nvidia-smi || true

python - <<'PY'
import sys
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
{cuda_check.rstrip()}
PY

{quoted_cmd(deucalion_python_args(args))}
"""
    job_path.write_text(content, encoding="utf-8", newline="\n")
    return job_path


def submit_deucalion_job(args: argparse.Namespace) -> int:
    job_path = write_deucalion_job(args)
    print(f"Wrote Slurm job: {job_path}")
    cmd = ["sbatch", str(job_path)]
    if args.dry_run:
        print(quoted_cmd(cmd))
        return 0
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return 0


def graph_regime_modes(raw: str, fallback_quad_mode: str) -> list[tuple[str, str]]:
    if raw == "balanced":
        return [("balanced", "balanced_sample")]
    if raw == "all":
        return [("all70", "all")]
    if raw == "both":
        return [("balanced", "balanced_sample"), ("all70", "all")]
    return [(str(fallback_quad_mode), str(fallback_quad_mode))]


def prefixed_spec(
    graph_regime: str,
    quad_mode: str,
    phase: str,
    base_name: str,
    test_id: int,
    hidden_dim: int,
    num_layers: int,
    node_feature_mode: str,
    edge_feature_mode: str,
    extra_args: tuple[str, ...] = (),
) -> RunSpec:
    return RunSpec(
        graph_regime=graph_regime,
        quad_mode=quad_mode,
        phase=phase,
        name=f"{graph_regime}_{base_name}",
        test_id=test_id,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        node_feature_mode=node_feature_mode,
        edge_feature_mode=edge_feature_mode,
        extra_args=tuple(extra_args),
    )


def subtracted_norm_mode_for(best_norm_mode: str) -> str:
    if best_norm_mode == "magnitude_norm_median":
        return "magnitude_subtracted_norm_median"
    return "magnitude_subtracted_norm_max"


def run_staged_plan_for_regime(
    *,
    graph_regime: str,
    quad_mode: str,
    args: argparse.Namespace,
    num_workers: int,
    split_file: Path,
    plan_root: Path,
    real_image: Path | None,
) -> tuple[list[RunResult], dict[str, str]]:
    results: list[RunResult] = []

    def spec(
        phase: str,
        base_name: str,
        test_id: int,
        hidden_dim: int,
        num_layers: int,
        node: str,
        edge: str,
        extra_args: tuple[str, ...] = (),
    ) -> RunSpec:
        return prefixed_spec(
            graph_regime,
            quad_mode,
            phase,
            base_name,
            test_id,
            hidden_dim,
            num_layers,
            node,
            edge,
            extra_args,
        )

    test0_specs = [
        spec("T0_arch", f"T0_l{num_layers}_h{hidden_dim}", 0, hidden_dim, num_layers, "none", "distance_max")
        for num_layers in (3, 5)
        for hidden_dim in parse_int_list(str(args.test0_hidden_dims))
    ]
    t0_results = [
        train_one(item, args=args, num_workers=num_workers, split_file=split_file, plan_root=plan_root, real_image=real_image)
        for item in test0_specs
    ]
    results.extend(t0_results)
    best_t0 = choose_best(t0_results, str(args.selection_metric))
    print(f"[{graph_regime}] Best T0 analysis: {best_t0.spec.name} {args.selection_metric}={best_t0.metric:.6f}")
    if args.only_test0:
        return results, {
            "best_t0_analysis": best_t0.spec.name,
            "best_final": best_t0.spec.name,
        }

    baseline_l3_h256 = next(item for item in t0_results if item.spec.name.endswith("T0_l3_h256"))
    print(f"[{graph_regime}] Fixed next architecture: {baseline_l3_h256.spec.name}")

    t1_specs = [
        spec("T1_edge_distance", "T1_dist_raw", 1, 256, 3, "none", "distance_raw"),
        spec("T1_edge_distance", "T1_dist_diagonal", 1, 256, 3, "none", "distance_diagonal"),
    ]
    t1_new_results = [
        train_one(item, args=args, num_workers=num_workers, split_file=split_file, plan_root=plan_root, real_image=real_image)
        for item in t1_specs
    ]
    results.extend(t1_new_results)
    best_edge = choose_best([baseline_l3_h256, *t1_new_results], str(args.selection_metric))
    print(f"[{graph_regime}] Best T1 edge: {best_edge.spec.name} {args.selection_metric}={best_edge.metric:.6f}")

    base_node_modes = [
        "magnitude_rank",
        "magnitude",
        "magnitude_subtracted",
        "magnitude_norm_max",
        "magnitude_norm_median",
    ]
    t2_base_specs = [
        spec("T2_node_magnitude", f"T2_node_{node_mode}", 2, 256, 3, node_mode, best_edge.spec.edge_feature_mode)
        for node_mode in base_node_modes
    ]
    t2_base_results = [
        train_one(item, args=args, num_workers=num_workers, split_file=split_file, plan_root=plan_root, real_image=real_image)
        for item in t2_base_specs
    ]
    results.extend(t2_base_results)

    t2_rank_improved_result = train_one(
        spec(
            "T2_node_magnitude_rank_loss",
            "T2_node_magnitude_rank_improve_loss",
            2,
            256,
            3,
            "magnitude_rank",
            best_edge.spec.edge_feature_mode,
            (
                "--class-distance-loss-weight",
                "0.2",
                "--class-rank-loss-weight",
                "0.2",
            ),
        ),
        args=args,
        num_workers=num_workers,
        split_file=split_file,
        plan_root=plan_root,
        real_image=real_image,
    )
    results.append(t2_rank_improved_result)

    norm_results = [item for item in t2_base_results if item.spec.node_feature_mode in {"magnitude_norm_max", "magnitude_norm_median"}]
    best_norm = choose_best(norm_results, str(args.selection_metric))
    sub_norm_mode = subtracted_norm_mode_for(best_norm.spec.node_feature_mode)
    sub_norm_result = train_one(
        spec("T2_node_magnitude", f"T2_node_{sub_norm_mode}", 2, 256, 3, sub_norm_mode, best_edge.spec.edge_feature_mode),
        args=args,
        num_workers=num_workers,
        split_file=split_file,
        plan_root=plan_root,
        real_image=real_image,
    )
    results.append(sub_norm_result)

    t2_results = [*t2_base_results, t2_rank_improved_result, sub_norm_result]
    best_node = choose_best(t2_results, str(args.selection_metric))
    print(f"[{graph_regime}] Best T2 node: {best_node.spec.name} {args.selection_metric}={best_node.metric:.6f}")

    t3_result = train_one(
        spec(
            "T3_edge_dmag",
            "T3_edge_dmag",
            3,
            256,
            3,
            best_node.spec.node_feature_mode,
            dmag_edge_mode(best_edge.spec.edge_feature_mode),
        ),
        args=args,
        num_workers=num_workers,
        split_file=split_file,
        plan_root=plan_root,
        real_image=real_image,
    )
    results.append(t3_result)

    t3_node_result = train_one(
        spec(
            "T3_edge_dmag_node",
            "T3_edge_dmag_nmedian",
            3,
            256,
            3,
            "magnitude_norm_median",
            "distance_max_dmag_node",
        ),
        args=args,
        num_workers=num_workers,
        split_file=split_file,
        plan_root=plan_root,
        real_image=real_image,
    )
    results.append(t3_node_result)

    t3_rank_result = train_one(
        spec(
            "T3_edge_rank_node",
            "T3_edge_rank_drank",
            3,
            256,
            3,
            "magnitude_rank",
            "distance_max_dmag_node",
        ),
        args=args,
        num_workers=num_workers,
        split_file=split_file,
        plan_root=plan_root,
        real_image=real_image,
    )
    results.append(t3_rank_result)

    best_final = choose_best([best_node, t3_result, t3_node_result, t3_rank_result], str(args.selection_metric))
    print(f"[{graph_regime}] Best final: {best_final.spec.name} {args.selection_metric}={best_final.metric:.6f}")

    return results, {
        "best_t0_analysis": best_t0.spec.name,
        "fixed_t1_architecture": baseline_l3_h256.spec.name,
        "best_t1_edge": best_edge.spec.name,
        "best_t2_node": best_node.spec.name,
        "best_final": best_final.spec.name,
    }


def main() -> int:
    plan_started = time.perf_counter()
    args = parse_args()
    if args.deucalion_submit:
        return submit_deucalion_job(args)
    num_workers = resolve_num_workers(args.num_workers)
    split_file = resolve_split_file(args.split_file)
    real_image = resolve_real_image(args.real_image)
    plan_name = args.plan_name or f"testes_gnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plan_root = (args.runs_root.expanduser().resolve() / plan_name).resolve()
    plan_root.mkdir(parents=True, exist_ok=True)

    print(f"Plan root: {plan_root}")
    print(f"Split file: {split_file}")
    print(f"Num workers: {num_workers}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Early-stop monitor: {args.early_stop_monitor}")
    if args.only_test0:
        print(f"Only TestID 0: hidden_dims={parse_int_list(str(args.test0_hidden_dims))}")
    if args.dataset_dir is not None:
        print(f"Dataset dir: {args.dataset_dir.expanduser().resolve()}")
    else:
        print("Dataset dir: latest synth_dataset/runs resolved by GNN.GNN")
    if args.skip_real_eval:
        print("Real image eval: skipped")
    elif real_image is None:
        print("Real image eval: skipped because no image was found; pass --real-image")
    else:
        print(f"Real image: {real_image}")

    all_results: list[RunResult] = []
    regime_summaries: dict[str, dict[str, str]] = {}
    for graph_regime, quad_mode in graph_regime_modes(str(args.graph_regimes), str(args.quad_mode)):
        print(f"Running graph regime: {graph_regime} (quad_mode={quad_mode})")
        regime_results, regime_summary = run_staged_plan_for_regime(
            graph_regime=graph_regime,
            quad_mode=quad_mode,
            args=args,
            num_workers=num_workers,
            split_file=split_file,
            plan_root=plan_root,
            real_image=real_image,
        )
        all_results.extend(regime_results)
        regime_summaries[graph_regime] = regime_summary

    final_best = choose_best(all_results, str(args.selection_metric))
    print(f"Best final overall: {final_best.spec.name} {args.selection_metric}={final_best.metric:.6f}")
    write_results_csv(plan_root, all_results, final_best, str(args.selection_metric))
    total_wall_min = float((time.perf_counter() - plan_started) / 60.0)

    plan_json = plan_root / "correr_testes_plan.json"
    plan_json.write_text(
        json.dumps(
            {
                "created": datetime.now().isoformat(timespec="seconds"),
                "selection_metric": args.selection_metric,
                "early_stop_monitor": args.early_stop_monitor,
                "only_test0": bool(args.only_test0),
                "test0_hidden_dims": parse_int_list(str(args.test0_hidden_dims)),
                "num_workers": int(num_workers),
                "split_file": str(split_file),
                "dataset_dir": None if args.dataset_dir is None else str(args.dataset_dir.expanduser().resolve()),
                "real_image": None if real_image is None else str(real_image),
                "graph_regimes": regime_summaries,
                "best_final": final_best.spec.name,
                "total_wall_min": total_wall_min,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Plan JSON: {plan_json}")
    print(f"Total wall time: {total_wall_min:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
