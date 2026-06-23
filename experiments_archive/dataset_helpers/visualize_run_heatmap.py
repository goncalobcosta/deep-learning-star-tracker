#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_sky_fov import style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate boresight and approximate FOV coverage heatmaps for "
            "instrumented dataset runs."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "Run names or paths. Examples: run1_baseline_all "
            "runs_tmp_validation/run5_expD_all "
            "tetra4/synth_dataset/runs_tmp_validation/run9_expC_timelapse"
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs_tmp_validation",
        help="Base directory used when a run name is provided.",
    )
    parser.add_argument(
        "--window-mode",
        choices=("auto", "full"),
        default="auto",
        help=(
            "Use the run timelapse plot window when available, or force full-sky plots. "
            "Default: auto."
        ),
    )
    parser.add_argument("--ra-bins", type=int, default=360, help="Number of RA bins. Default: 360.")
    parser.add_argument("--dec-bins", type=int, default=180, help="Number of Dec bins. Default: 180.")
    parser.add_argument(
        "--scene-step",
        type=int,
        default=1,
        help="Use every Nth scene center to build the heatmaps. Default: 1.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--boresight-name",
        default="boresight_heatmap_ra_dec.png",
        help="Output filename for the boresight heatmap inside each run directory.",
    )
    parser.add_argument(
        "--fov-name",
        default="fov_coverage_heatmap_approx_ra_dec.png",
        help="Output filename for the approximate FOV coverage heatmap inside each run directory.",
    )
    return parser.parse_args()


def resolve_run_dir(run_arg: str, base_dir: Path) -> Path:
    candidate = Path(run_arg)
    if candidate.exists():
        return candidate.expanduser().resolve()
    return (base_dir / run_arg).expanduser().resolve()


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "dataset_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_run_params(run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    run_info = manifest.get("run", {})
    if not isinstance(run_info, dict):
        return {}
    params = run_info.get("parameters", {})
    return params if isinstance(params, dict) else {}


def get_plot_window(params: dict[str, Any], window_mode: str) -> tuple[float, float, float, float] | None:
    if window_mode == "full":
        return None
    if not bool(params.get("timelapse")):
        return None

    ra_plot = params.get("timelapse_plot_ra_window_deg")
    dec_plot = params.get("timelapse_plot_dec_window_deg")
    if (
        isinstance(ra_plot, list)
        and len(ra_plot) == 2
        and ra_plot[0] is not None
        and ra_plot[1] is not None
        and isinstance(dec_plot, list)
        and len(dec_plot) == 2
        and dec_plot[0] is not None
        and dec_plot[1] is not None
    ):
        return (float(ra_plot[0]), float(ra_plot[1]), float(dec_plot[0]), float(dec_plot[1]))

    ra_win = params.get("timelapse_ra_window_deg")
    dec_win = params.get("timelapse_dec_window_deg")
    if (
        isinstance(ra_win, list)
        and len(ra_win) == 2
        and ra_win[0] is not None
        and ra_win[1] is not None
        and isinstance(dec_win, list)
        and len(dec_win) == 2
        and dec_win[0] is not None
        and dec_win[1] is not None
    ):
        return (float(ra_win[0]), float(ra_win[1]), float(dec_win[0]), float(dec_win[1]))
    return None


def unwrap_ra_to_center(ra_deg: np.ndarray, center_deg: float) -> np.ndarray:
    ra = np.mod(np.asarray(ra_deg, dtype=np.float64), 360.0)
    candidates = np.stack((ra - 360.0, ra, ra + 360.0), axis=0)
    idx = np.argmin(np.abs(candidates - float(center_deg)), axis=0)
    cols = np.arange(ra.shape[0], dtype=np.int64)
    return candidates[idx, cols]


def build_bin_edges(
    *,
    window: tuple[float, float, float, float] | None,
    ra_bins: int,
    dec_bins: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None, tuple[float, float] | None, float]:
    if window is None:
        ra_edges = np.linspace(0.0, 360.0, int(ra_bins) + 1, dtype=np.float64)
        dec_edges = np.linspace(-90.0, 90.0, int(dec_bins) + 1, dtype=np.float64)
        return ra_edges, dec_edges, None, None, 180.0

    ra_left, ra_right, dec_a, dec_b = window
    ra_lo = min(float(ra_left), float(ra_right))
    ra_hi = max(float(ra_left), float(ra_right))
    dec_lo = min(float(dec_a), float(dec_b))
    dec_hi = max(float(dec_a), float(dec_b))
    ra_edges = np.linspace(ra_lo, ra_hi, int(ra_bins) + 1, dtype=np.float64)
    dec_edges = np.linspace(dec_lo, dec_hi, int(dec_bins) + 1, dtype=np.float64)
    return (
        ra_edges,
        dec_edges,
        (float(ra_left), float(ra_right)),
        (float(dec_a), float(dec_b)),
        0.5 * (ra_lo + ra_hi),
    )


def histogram_scene_centers(
    *,
    scene_ra: np.ndarray,
    scene_dec: np.ndarray,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
    window_center_ra: float,
    full_sky: bool,
) -> np.ndarray:
    if full_sky:
        ra_vals = np.mod(scene_ra.astype(np.float64, copy=False), 360.0)
    else:
        ra_vals = unwrap_ra_to_center(scene_ra.astype(np.float64, copy=False), float(window_center_ra))
    dec_vals = scene_dec.astype(np.float64, copy=False)

    mask = (
        (ra_vals >= float(ra_edges[0]))
        & (ra_vals <= float(ra_edges[-1]))
        & (dec_vals >= float(dec_edges[0]))
        & (dec_vals <= float(dec_edges[-1]))
    )
    hist, _, _ = np.histogram2d(
        dec_vals[mask],
        ra_vals[mask],
        bins=(dec_edges, ra_edges),
    )
    return hist.astype(np.float64, copy=False)


def box_sum_with_ra_wrap(hist: np.ndarray, *, ra_radius: int, dec_radius: int) -> np.ndarray:
    if int(ra_radius) < 0 or int(dec_radius) < 0:
        raise ValueError("Box radii must be >= 0")
    if hist.size == 0:
        return hist.copy()

    work = hist.astype(np.float64, copy=False)
    if int(ra_radius) > 0:
        work = np.pad(work, ((0, 0), (int(ra_radius), int(ra_radius))), mode="wrap")
    if int(dec_radius) > 0:
        work = np.pad(work, ((int(dec_radius), int(dec_radius)), (0, 0)), mode="constant")

    integral = np.pad(work.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")
    kernel_h = 2 * int(dec_radius) + 1
    kernel_w = 2 * int(ra_radius) + 1
    out_h, out_w = hist.shape
    return (
        integral[kernel_h : kernel_h + out_h, kernel_w : kernel_w + out_w]
        - integral[:out_h, kernel_w : kernel_w + out_w]
        - integral[kernel_h : kernel_h + out_h, :out_w]
        + integral[:out_h, :out_w]
    )


def get_fov_from_params(params: dict[str, Any]) -> tuple[float, float]:
    fov_h = float(params.get("fov_horizontal_deg", 17.2))
    fov_v = float(params.get("fov_vertical_deg", 13.0))
    return fov_h, fov_v


def choose_norm(data: np.ndarray) -> LogNorm | None:
    masked = np.ma.masked_less_equal(np.asarray(data, dtype=np.float64), 0.0)
    if masked.count() == 0:
        return None
    vmax = float(masked.max())
    if vmax <= 1.0:
        return None
    return LogNorm(vmin=1.0, vmax=vmax)


def save_heatmap(
    *,
    out_file: Path,
    heatmap: np.ndarray,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
    plot_ra: tuple[float, float] | None,
    plot_dec: tuple[float, float] | None,
    title: str,
    colorbar_label: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.2), facecolor="#020617")
    if plot_ra is None or plot_dec is None:
        style_axis(ax, title)
    else:
        style_axis(
            ax,
            title,
            ra_min=float(plot_ra[0]),
            ra_max=float(plot_ra[1]),
            dec_min=float(plot_dec[0]),
            dec_max=float(plot_dec[1]),
        )

    masked = np.ma.masked_less_equal(np.asarray(heatmap, dtype=np.float64), 0.0)
    if masked.count() == 0:
        pcm = ax.pcolormesh(
            ra_edges,
            dec_edges,
            np.zeros_like(heatmap, dtype=np.float64),
            shading="auto",
            cmap="magma",
        )
    else:
        pcm = ax.pcolormesh(
            ra_edges,
            dec_edges,
            masked,
            shading="auto",
            cmap="magma",
            norm=choose_norm(masked.filled(0.0)),
        )

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(colorbar_label, color="#e2e8f0")
    cbar.ax.tick_params(colors="#cbd5e1", labelsize=8)
    cbar.outline.set_edgecolor("#334155")

    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)


def process_run(run_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    run = run_dir.expanduser().resolve()
    scene_meta_path = run / "scene_metadata.npz"
    if not scene_meta_path.exists():
        raise FileNotFoundError(f"scene_metadata.npz not found: {scene_meta_path}")

    params = load_run_params(run)
    window = get_plot_window(params, str(args.window_mode))
    ra_edges, dec_edges, plot_ra, plot_dec, ra_center = build_bin_edges(
        window=window,
        ra_bins=int(args.ra_bins),
        dec_bins=int(args.dec_bins),
    )

    with np.load(scene_meta_path, allow_pickle=False) as scene_meta:
        if "scene_center_ra_deg" not in scene_meta or "scene_center_dec_deg" not in scene_meta:
            raise RuntimeError(f"scene_metadata.npz is missing scene center arrays: {scene_meta_path}")
        scene_ra = np.asarray(scene_meta["scene_center_ra_deg"], dtype=np.float64)
        scene_dec = np.asarray(scene_meta["scene_center_dec_deg"], dtype=np.float64)

    step = max(int(args.scene_step), 1)
    if step > 1:
        scene_ra = scene_ra[::step]
        scene_dec = scene_dec[::step]

    hist = histogram_scene_centers(
        scene_ra=scene_ra,
        scene_dec=scene_dec,
        ra_edges=ra_edges,
        dec_edges=dec_edges,
        window_center_ra=ra_center,
        full_sky=(window is None),
    )

    fov_h_deg, fov_v_deg = get_fov_from_params(params)
    ra_bin_width = abs(float(ra_edges[-1] - ra_edges[0])) / max(int(args.ra_bins), 1)
    dec_bin_width = abs(float(dec_edges[-1] - dec_edges[0])) / max(int(args.dec_bins), 1)
    ra_radius = max(0, int(math.ceil((0.5 * float(fov_h_deg)) / max(ra_bin_width, 1e-9))))
    dec_radius = max(0, int(math.ceil((0.5 * float(fov_v_deg)) / max(dec_bin_width, 1e-9))))
    fov_hist = box_sum_with_ra_wrap(hist, ra_radius=ra_radius, dec_radius=dec_radius)

    boresight_path = run / str(args.boresight_name)
    save_heatmap(
        out_file=boresight_path,
        heatmap=hist,
        ra_edges=ra_edges,
        dec_edges=dec_edges,
        plot_ra=plot_ra,
        plot_dec=plot_dec,
        title=f"Boresight Heatmap | {run.name} | scenes={scene_ra.shape[0]}",
        colorbar_label="Boresight centers per bin",
        dpi=int(args.dpi),
    )

    fov_path = run / str(args.fov_name)
    save_heatmap(
        out_file=fov_path,
        heatmap=fov_hist,
        ra_edges=ra_edges,
        dec_edges=dec_edges,
        plot_ra=plot_ra,
        plot_dec=plot_dec,
        title=(
            f"Approx. FOV Coverage Heatmap | {run.name} | "
            f"FOV={float(fov_h_deg):.1f}x{float(fov_v_deg):.1f} deg"
        ),
        colorbar_label="Approx. FOV passes per bin",
        dpi=int(args.dpi),
    )

    print(f"\nProcessed run: {run}")
    print(f"Scenes used: {int(scene_ra.shape[0])} (scene_step={step})")
    if plot_ra is None or plot_dec is None:
        print("Plot window: full sky")
    else:
        print(
            "Plot window: "
            f"RA[{float(plot_ra[0]):.3f}, {float(plot_ra[1]):.3f}] "
            f"Dec[{float(plot_dec[0]):.3f}, {float(plot_dec[1]):.3f}]"
        )
    print(f"Boresight heatmap: {boresight_path}")
    print(f"Approx. FOV heatmap: {fov_path}")
    return boresight_path, fov_path


def main() -> None:
    args = parse_args()
    if int(args.ra_bins) <= 0 or int(args.dec_bins) <= 0:
        raise SystemExit("--ra-bins and --dec-bins must be > 0")
    if int(args.scene_step) <= 0:
        raise SystemExit("--scene-step must be > 0")

    base_dir = args.base_dir.expanduser().resolve()
    for run_arg in args.runs:
        run_dir = resolve_run_dir(run_arg, base_dir)
        process_run(run_dir, args)


if __name__ == "__main__":
    main()
