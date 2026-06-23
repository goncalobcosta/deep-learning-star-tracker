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
    parser = argparse.ArgumentParser(
        description=(
            "Professor Experiment D: stop when scoped appear_count satisfies "
            "the baseline mean +/- margin band."
        )
    )
    add_common_dataset_args(parser)
    add_timelapse_args(parser)
    parser.add_argument("--baseline-run", type=Path, default=None)
    parser.add_argument("--appear-band-margin", type=float, default=500.0)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    runs_root = ensure_runs_root(args.runs_root)
    baseline_run = resolve_baseline_run(args.baseline_run)
    before = snapshot_run_names(runs_root)

    cmd = [
        sys.executable,
        str(PROF_SCRIPT),
        "--stop-mode",
        "appear_band_target",
        "--baseline-run",
        str(baseline_run),
        "--appear-band-margin",
        str(float(args.appear_band_margin)),
        "--chunk-size-mb",
        str(int(args.chunk_size_mb)),
        "--runs-root",
        str(runs_root),
        "--magnitude-cutoff",
        str(float(args.magnitude_cutoff)),
    ]
    cmd += build_magnitude_perturb_args(args)
    if args.max_attempts is not None:
        cmd += ["--max-attempts", str(int(args.max_attempts))]
    if args.seed is not None:
        cmd += ["--seed", str(int(args.seed))]
    cmd += build_timelapse_args(args)

    run_command(cmd)
    run_dir = detect_new_run_dir(runs_root, before)
    print_run_summary("PROF_D", run_dir)

    append_history(
        {
            "experiment": "prof_d_appear_band_target",
            "run_dir": str(run_dir),
            "runs_root": str(runs_root),
            "baseline_run": str(baseline_run),
            "appear_band_margin": float(args.appear_band_margin),
            "max_attempts": int(args.max_attempts) if args.max_attempts is not None else None,
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
