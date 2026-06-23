from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    PROF_SCRIPT,
    add_common_dataset_args,
    add_timelapse_args,
    append_history,
    build_timelapse_args,
    build_magnitude_perturb_args,
    detect_new_run_dir,
    ensure_runs_root,
    print_run_summary,
    resolve_baseline_run,
    run_command,
    snapshot_run_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Professor Experiment A: same scene budget as baseline, no cap.")
    add_common_dataset_args(parser)
    add_timelapse_args(parser)
    parser.add_argument("--baseline-run", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    runs_root = ensure_runs_root(args.runs_root)
    baseline_run = resolve_baseline_run(args.baseline_run)
    before = snapshot_run_names(runs_root)

    cmd = [
        sys.executable,
        str(PROF_SCRIPT),
        "--stop-mode",
        "scene_budget",
        "--baseline-run",
        str(baseline_run),
        "--chunk-size-mb",
        str(int(args.chunk_size_mb)),
        "--runs-root",
        str(runs_root),
        "--magnitude-cutoff",
        str(float(args.magnitude_cutoff)),
    ]
    cmd += build_magnitude_perturb_args(args)
    if args.seed is not None:
        cmd += ["--seed", str(int(args.seed))]
    cmd += build_timelapse_args(args)

    run_command(cmd)
    run_dir = detect_new_run_dir(runs_root, before)
    print_run_summary("PROF_A", run_dir)

    append_history(
        {
            "experiment": "prof_a_scene_budget_no_cap",
            "run_dir": str(run_dir),
            "runs_root": str(runs_root),
            "baseline_run": str(baseline_run),
            "chunk_size_mb": int(args.chunk_size_mb),
            "magnitude_cutoff": float(args.magnitude_cutoff),
            "magnitude_perturb_mean": float(args.magnitude_perturb_mean) if args.magnitude_perturb_mean is not None else None,
            "magnitude_perturb_sigma": float(args.magnitude_perturb_sigma) if args.magnitude_perturb_sigma is not None else None,
            "seed": int(args.seed) if args.seed is not None else None,
            "timelapse": bool(args.timelapse),
        }
    )


if __name__ == "__main__":
    main()
