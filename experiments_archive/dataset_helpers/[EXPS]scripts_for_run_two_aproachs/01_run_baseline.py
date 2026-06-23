from __future__ import annotations

import argparse
import sys

from _common import (
    BASELINE_SCRIPT,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline dataset generation and save baseline reference.")
    add_common_dataset_args(parser)
    add_timelapse_args(parser)
    parser.add_argument("--guide-stars", type=int, default=8818)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-export-appear-csv", action="store_true")
    args = parser.parse_args()

    runs_root = ensure_runs_root(args.runs_root)
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

    csv_path = export_appear_count_csv(run_dir) if not bool(args.no_export_appear_csv) else None
    print_run_summary("BASELINE", run_dir)
    if csv_path is not None:
        print(f"  appear_count CSV: {csv_path}")

    append_history(
        {
            "experiment": "baseline",
            "run_dir": str(run_dir),
            "runs_root": str(runs_root),
            "guide_stars": int(args.guide_stars),
            "num_repeats": int(args.num_repeats),
            "chunk_size_mb": int(args.chunk_size_mb),
            "num_workers": int(args.num_workers),
            "magnitude_cutoff": float(args.magnitude_cutoff),
            "magnitude_perturb_mean": float(args.magnitude_perturb_mean) if args.magnitude_perturb_mean is not None else None,
            "magnitude_perturb_sigma": float(args.magnitude_perturb_sigma) if args.magnitude_perturb_sigma is not None else None,
            "timelapse": bool(args.timelapse),
        }
    )


if __name__ == "__main__":
    main()
