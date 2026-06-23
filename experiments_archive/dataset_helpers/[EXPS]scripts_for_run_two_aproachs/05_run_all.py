from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    BASELINE_SCRIPT,
    PROF_SCRIPT,
    add_common_dataset_args,
    add_timelapse_args,
    append_history,
    build_timelapse_args,
    build_magnitude_perturb_args,
    detect_new_run_dir,
    ensure_runs_root,
    export_appear_count_csv,
    print_run_summary,
    run_command,
    set_last_baseline_run,
    snapshot_run_names,
)


def _run_baseline(args: argparse.Namespace, runs_root: Path) -> Path:
    before = snapshot_run_names(runs_root)
    cmd = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--guide-stars",
        str(int(args.guide_stars)),
        "--num-repeats",
        str(int(args.num_repeats)),
        "--chunk-size-mb",
        str(int(args.chunk_size_mb)),
        "--num-workers",
        str(int(args.num_workers)),
        "--runs-root",
        str(runs_root),
        "--instrument-coverage",
        "--magnitude-cutoff",
        str(float(args.magnitude_cutoff)),
    ]
    cmd += build_magnitude_perturb_args(args)
    cmd += build_timelapse_args(args)
    run_command(cmd)
    run_dir = detect_new_run_dir(runs_root, before)
    set_last_baseline_run(run_dir)
    if not bool(args.no_export_csv):
        export_appear_count_csv(run_dir)
    print_run_summary("BASELINE", run_dir)
    return run_dir


def _run_prof(args: argparse.Namespace, runs_root: Path, baseline_run: Path, mode: str) -> Path:
    before = snapshot_run_names(runs_root)
    cmd = [
        sys.executable,
        str(PROF_SCRIPT),
        "--stop-mode",
        mode,
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
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline + experiments A, B, C in sequence.")
    add_common_dataset_args(parser)
    add_timelapse_args(parser)
    parser.add_argument("--guide-stars", type=int, default=8818)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--appear-cap-margin", type=float, default=500.0)
    parser.add_argument("--appear-band-margin", type=float, default=500.0)
    parser.add_argument("--include-exp-d", action="store_true")
    parser.add_argument("--no-export-csv", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    runs_root = ensure_runs_root(args.runs_root)

    baseline_run = _run_baseline(args, runs_root)
    run_a = _run_prof(args, runs_root, baseline_run, mode="scene_budget")
    print_run_summary("PROF_A", run_a)
    run_b = _run_prof(args, runs_root, baseline_run, mode="appear_mean_target")
    print_run_summary("PROF_B", run_b)

    before = snapshot_run_names(runs_root)
    cmd_c = [
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
    cmd_c += build_magnitude_perturb_args(args)
    cmd_c += ["--appear-cap-margin", str(float(args.appear_cap_margin))]
    if args.seed is not None:
        cmd_c += ["--seed", str(int(args.seed))]
    cmd_c += build_timelapse_args(args)
    run_command(cmd_c)
    run_c = detect_new_run_dir(runs_root, before)
    print_run_summary("PROF_C", run_c)

    run_d = None
    if bool(args.include_exp_d):
        before = snapshot_run_names(runs_root)
        cmd_d = [
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
        cmd_d += build_magnitude_perturb_args(args)
        if args.seed is not None:
            cmd_d += ["--seed", str(int(args.seed))]
        cmd_d += build_timelapse_args(args)
        run_command(cmd_d)
        run_d = detect_new_run_dir(runs_root, before)
        print_run_summary("PROF_D", run_d)

    append_history(
        {
            "experiment": "run_all",
            "runs_root": str(runs_root),
            "baseline_run": str(baseline_run),
            "run_a": str(run_a),
            "run_b": str(run_b),
            "run_c": str(run_c),
            "run_d": str(run_d) if run_d is not None else None,
            "guide_stars": int(args.guide_stars),
            "num_repeats": int(args.num_repeats),
            "num_workers": int(args.num_workers),
            "chunk_size_mb": int(args.chunk_size_mb),
            "magnitude_cutoff": float(args.magnitude_cutoff),
            "magnitude_perturb_mean": float(args.magnitude_perturb_mean) if args.magnitude_perturb_mean is not None else None,
            "magnitude_perturb_sigma": float(args.magnitude_perturb_sigma) if args.magnitude_perturb_sigma is not None else None,
            "appear_cap_margin": float(args.appear_cap_margin),
            "appear_band_margin": float(args.appear_band_margin),
            "include_exp_d": bool(args.include_exp_d),
            "timelapse": bool(args.timelapse),
            "seed": int(args.seed) if args.seed is not None else None,
        }
    )


if __name__ == "__main__":
    main()
