from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SYNTH_DIR = BASE_DIR.parents[1]
BASELINE_SCRIPT = SYNTH_DIR / "generate_dataset_baseline.py"
PROF_SCRIPT = SYNTH_DIR / "generate_dataset_aletorio.py"
STATE_FILE = BASE_DIR / "state.json"
HISTORY_FILE = BASE_DIR / "history.jsonl"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_runs_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_run_dirs(runs_root: Path) -> list[Path]:
    root = runs_root.expanduser().resolve()
    if not root.exists():
        return []
    dirs = [
        d
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith("run") and d.name[3:].isdigit()
    ]
    return sorted(dirs, key=lambda p: int(p.name[3:]))


def snapshot_run_names(runs_root: Path) -> set[str]:
    return {d.name for d in list_run_dirs(runs_root)}


def detect_new_run_dir(runs_root: Path, before_names: set[str]) -> Path:
    after = list_run_dirs(runs_root)
    new_dirs = [d for d in after if d.name not in before_names]
    if not new_dirs:
        if not after:
            raise RuntimeError(f"No runs found in {runs_root}")
        return after[-1]
    return new_dirs[-1]


def run_command(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


def read_coverage_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "coverage_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"coverage_summary.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def print_run_summary(label: str, run_dir: Path) -> None:
    cov = read_coverage_summary(run_dir)
    print(f"[{label}] {run_dir}")
    print(
        "  scene_count="
        f"{cov.get('scene_count')}  "
        f"scope_stars={cov.get('coverage_scope_stars')}  "
        f"appear_count_mean={cov.get('appear_count_mean')}  "
        f"min={cov.get('appear_count_min')}  "
        f"max={cov.get('appear_count_max')}"
    )


def export_appear_count_csv(run_dir: Path) -> Path | None:
    stats_path = run_dir / "coverage_stats.npz"
    if not stats_path.exists():
        return None

    with np.load(stats_path, allow_pickle=False) as data:
        if "final_appear_count" not in data:
            return None
        counts = np.asarray(data["final_appear_count"], dtype=np.int64)

    out = run_dir / "appear_count_per_star.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["star_index", "appear_count"])
        for idx, val in enumerate(counts.tolist()):
            writer.writerow([idx, int(val)])
    return out


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def set_last_baseline_run(run_dir: Path) -> None:
    state = load_state()
    state["last_baseline_run"] = str(run_dir.expanduser().resolve())
    state["updated_at_utc"] = now_utc_iso()
    save_state(state)


def resolve_baseline_run(baseline_run: Path | None) -> Path:
    if baseline_run is not None:
        run = baseline_run.expanduser().resolve()
        if not run.exists():
            raise FileNotFoundError(f"Baseline run not found: {run}")
        return run

    state = load_state()
    raw = state.get("last_baseline_run")
    if not raw:
        raise ValueError(
            "No baseline run provided and none saved in state.json. "
            "Run 01_run_baseline.py first or pass --baseline-run."
        )
    run = Path(str(raw)).expanduser().resolve()
    if not run.exists():
        raise FileNotFoundError(f"Saved baseline run no longer exists: {run}")
    return run


def append_history(event: dict[str, Any]) -> None:
    record = {"timestamp_utc": now_utc_iso(), **event}
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=SYNTH_DIR / "runs_tmp_validation",
        help="Root directory where runX folders are created.",
    )
    parser.add_argument("--chunk-size-mb", type=int, default=256)
    parser.add_argument("--magnitude-cutoff", type=float, default=8.0)
    parser.add_argument("--magnitude-perturb-mean", type=float, default=None)
    parser.add_argument("--magnitude-perturb-sigma", type=float, default=None)


def build_magnitude_perturb_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.magnitude_perturb_mean is not None:
        out += ["--magnitude-perturb-mean", str(float(args.magnitude_perturb_mean))]
    if args.magnitude_perturb_sigma is not None:
        out += ["--magnitude-perturb-sigma", str(float(args.magnitude_perturb_sigma))]
    return out


def add_timelapse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timelapse", action="store_true")
    parser.add_argument("--timelapse-ra-min", type=float, default=150.0)
    parser.add_argument("--timelapse-ra-max", type=float, default=180.0)
    parser.add_argument("--timelapse-dec-min", type=float, default=-90.0)
    parser.add_argument("--timelapse-dec-max", type=float, default=90.0)
    parser.add_argument("--timelapse-plot-ra-min", type=float, default=None)
    parser.add_argument("--timelapse-plot-ra-max", type=float, default=None)
    parser.add_argument("--timelapse-plot-dec-min", type=float, default=None)
    parser.add_argument("--timelapse-plot-dec-max", type=float, default=None)
    parser.add_argument("--every-nth-scene", type=int, default=120)
    parser.add_argument("--timelapse-require-full-fov-inside", action="store_true")
    parser.add_argument("--timelapse-fov-edge-samples", type=int, default=24)


def build_timelapse_args(args: argparse.Namespace) -> list[str]:
    if not bool(args.timelapse):
        return []
    out = [
        "--timelapse",
        "--timelapse-ra-min",
        str(float(args.timelapse_ra_min)),
        "--timelapse-ra-max",
        str(float(args.timelapse_ra_max)),
        "--timelapse-dec-min",
        str(float(args.timelapse_dec_min)),
        "--timelapse-dec-max",
        str(float(args.timelapse_dec_max)),
        "--timelapse-plot-ra-min",
        str(float(args.timelapse_plot_ra_min)) if args.timelapse_plot_ra_min is not None else str(float(args.timelapse_ra_min)),
        "--timelapse-plot-ra-max",
        str(float(args.timelapse_plot_ra_max)) if args.timelapse_plot_ra_max is not None else str(float(args.timelapse_ra_max)),
        "--timelapse-plot-dec-min",
        str(float(args.timelapse_plot_dec_min)) if args.timelapse_plot_dec_min is not None else str(float(args.timelapse_dec_min)),
        "--timelapse-plot-dec-max",
        str(float(args.timelapse_plot_dec_max)) if args.timelapse_plot_dec_max is not None else str(float(args.timelapse_dec_max)),
        "--every-nth-scene",
        str(int(args.every_nth_scene)),
    ]
    if bool(args.timelapse_require_full_fov_inside):
        out.append("--timelapse-require-full-fov-inside")
    out += ["--timelapse-fov-edge-samples", str(int(args.timelapse_fov_edge_samples))]
    return out
