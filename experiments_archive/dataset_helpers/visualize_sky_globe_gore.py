#!/usr/bin/env python3
"""Render RA/Dec sky plots as Sky Globe Gore panels."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_sky_fov import build_fov_polygon_radec
from visualize_sky_fov import load_database
from visualize_sky_fov import sample_boresights
from visualize_sky_fov import sample_rolls
from visualize_sky_fov import split_by_ra_wrap
from visualize_sky_fov import vectors_to_radec_deg


SQRT2 = math.sqrt(2.0)


def mollweide_project(lon_rad: np.ndarray, lat_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude radians into equal-area Mollweide x/y."""
    lon = np.asarray(lon_rad, dtype=np.float64)
    lat = np.asarray(lat_rad, dtype=np.float64)

    theta = lat.copy()
    pole = np.isclose(np.abs(lat), 0.5 * math.pi)
    work = ~pole
    for _ in range(12):
        if not np.any(work):
            break
        th = theta[work]
        delta = (2.0 * th + np.sin(2.0 * th) - math.pi * np.sin(lat[work])) / (
            2.0 + 2.0 * np.cos(2.0 * th)
        )
        theta[work] = th - delta
        if float(np.max(np.abs(delta))) < 1e-12:
            break
    theta[pole] = np.sign(lat[pole]) * 0.5 * math.pi

    x = (2.0 * SQRT2 / math.pi) * lon * np.cos(theta)
    y = SQRT2 * np.sin(theta)
    return x, y


def gore_indices(ra_deg: np.ndarray, gore_width_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ra = np.mod(np.asarray(ra_deg, dtype=np.float64), 360.0)
    n_gores = int(round(360.0 / float(gore_width_deg)))
    display_pos = np.mod(360.0 - ra, 360.0)
    idx = np.floor(display_pos / float(gore_width_deg)).astype(np.int64)
    idx = np.clip(idx, 0, n_gores - 1)
    centers = 360.0 - (idx.astype(np.float64) + 0.5) * float(gore_width_deg)
    centers = np.mod(centers, 360.0)
    return idx, centers


def normalize_signed_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (np.asarray(angle_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def radec_to_interrupted(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    gore_width_deg: float,
    gore_gap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx, centers = gore_indices(ra_deg, gore_width_deg)
    local_lon_deg = normalize_signed_deg(centers - np.mod(np.asarray(ra_deg, dtype=np.float64), 360.0))
    x, y = mollweide_project(np.radians(local_lon_deg), np.radians(np.asarray(dec_deg, dtype=np.float64)))

    n_gores = int(round(360.0 / float(gore_width_deg)))
    half_width_rad = math.radians(float(gore_width_deg) / 2.0)
    max_local_x = (2.0 * SQRT2 / math.pi) * half_width_rad
    pitch = 2.0 * max_local_x + float(gore_gap)
    offsets = (idx.astype(np.float64) - 0.5 * (n_gores - 1)) * pitch
    return x + offsets, y, idx


def draw_gore_grid(ax: plt.Axes, *, gore_width_deg: float, gore_gap: float) -> None:
    n_gores = int(round(360.0 / float(gore_width_deg)))
    half_width_rad = math.radians(float(gore_width_deg) / 2.0)
    max_local_x = (2.0 * SQRT2 / math.pi) * half_width_rad
    pitch = 2.0 * max_local_x + float(gore_gap)
    grid_color = "#64748b"
    boundary_color = "#cbd5e1"
    grid_linewidth = 0.85
    boundary_linewidth = 1.25
    grid_zorder = 6
    boundary_zorder = 9

    lon_samples = np.linspace(-half_width_rad, half_width_rad, 120)
    lat_samples = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 240)

    for i in range(n_gores):
        offset = (i - 0.5 * (n_gores - 1)) * pitch
        center_ra = (360.0 - (i + 0.5) * float(gore_width_deg)) % 360.0

        for dec in np.arange(-60.0, 61.0, 30.0):
            lat = np.full_like(lon_samples, math.radians(float(dec)))
            x, y = mollweide_project(lon_samples, lat)
            ax.plot(x + offset, y, color=grid_color, linewidth=grid_linewidth, alpha=0.85, zorder=grid_zorder)

        for local_lon in (-half_width_rad, 0.0, half_width_rad):
            lon = np.full_like(lat_samples, local_lon)
            x, y = mollweide_project(lon, lat_samples)
            ax.plot(x + offset, y, color=grid_color, linewidth=grid_linewidth, alpha=0.85, zorder=grid_zorder)

        # Outer boundary for the gore.
        left_x, left_y = mollweide_project(np.full_like(lat_samples, -half_width_rad), lat_samples)
        right_x, right_y = mollweide_project(np.full_like(lat_samples, half_width_rad), lat_samples)
        top_x, top_y = mollweide_project(lon_samples, np.full_like(lon_samples, 0.5 * math.pi))
        bottom_x, bottom_y = mollweide_project(lon_samples, np.full_like(lon_samples, -0.5 * math.pi))
        ax.plot(left_x + offset, left_y, color=boundary_color, linewidth=boundary_linewidth, alpha=0.9, zorder=boundary_zorder)
        ax.plot(right_x + offset, right_y, color=boundary_color, linewidth=boundary_linewidth, alpha=0.9, zorder=boundary_zorder)
        ax.plot(top_x + offset, top_y, color=boundary_color, linewidth=boundary_linewidth, alpha=0.9, zorder=boundary_zorder)
        ax.plot(bottom_x + offset, bottom_y, color=boundary_color, linewidth=boundary_linewidth, alpha=0.9, zorder=boundary_zorder)

        ax.text(offset, -SQRT2 * 1.08, f"{int(round(center_ra))}", color="#cbd5e1", ha="center", va="top", fontsize=7)

    for dec in np.arange(-60.0, 61.0, 30.0):
        ax.text(
            -0.5 * (n_gores - 1) * pitch - max_local_x - 0.12,
            SQRT2 * math.sin(math.radians(float(dec)) / 2.0),
            f"{int(dec)} deg",
            color="#cbd5e1",
            ha="right",
            va="center",
            fontsize=7,
        )

    total_half_width = 0.5 * (n_gores - 1) * pitch + max_local_x
    ax.set_xlim(-total_half_width - 0.22, total_half_width + 0.22)
    ax.set_ylim(-SQRT2 * 1.18, SQRT2 * 1.16)


def new_figure(title: str, *, gore_width_deg: float, gore_gap: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13.8, 6.2), facecolor="#020617")
    ax.set_facecolor("#020617")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, color="#f8fafc", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    draw_gore_grid(ax, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    ax.text(0.0, -SQRT2 * 1.18, "Right Ascension gore center (deg)", color="#e2e8f0", ha="center", va="top", fontsize=10)
    ax.text(ax.get_xlim()[0] + 0.03, 0.0, "Declination", color="#e2e8f0", ha="left", va="center", rotation=90, fontsize=10)
    return fig, ax


def split_projected_path(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    gore_width_deg: float,
    gore_gap: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    x, y, idx = radec_to_interrupted(ra_deg, dec_deg, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    if x.size <= 1:
        return [(x, y)]

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i in range(1, x.size):
        if int(idx[i]) != int(idx[i - 1]):
            if i - start >= 2:
                segments.append((x[start:i], y[start:i]))
            start = i
    if x.size - start >= 2:
        segments.append((x[start:], y[start:]))
    return segments


def save_catalog(
    *,
    out_file: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    dpi: int,
    gore_width_deg: float,
    gore_gap: float,
) -> None:
    fig, ax = new_figure(
        f"Tetra Catalog Sky Map - Sky Globe Gores - N={ra_deg.shape[0]}",
        gore_width_deg=gore_width_deg,
        gore_gap=gore_gap,
    )
    x, y, _ = radec_to_interrupted(ra_deg, dec_deg, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    ax.scatter(x, y, s=2.0, c="#e2e8f0", alpha=0.72, linewidths=0.0, zorder=4)
    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)


def save_fov(
    *,
    out_file: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    boresights: np.ndarray,
    rolls: np.ndarray,
    fov_h_rad: float,
    fov_v_rad: float,
    edge_samples: int,
    dpi: int,
    gore_width_deg: float,
    gore_gap: float,
) -> None:
    fig, ax = new_figure(
        "FOV footprint on Sky Globe Gores",
        gore_width_deg=gore_width_deg,
        gore_gap=gore_gap,
    )
    x, y, _ = radec_to_interrupted(ra_deg, dec_deg, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    ax.scatter(x, y, s=1.0, c="#94a3b8", alpha=0.25, linewidths=0.0, zorder=3)

    center_ra, center_dec = vectors_to_radec_deg(boresights)
    cx, cy, _ = radec_to_interrupted(center_ra, center_dec, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    ax.scatter(cx, cy, s=24.0, c="#f59e0b", alpha=0.95, linewidths=0.0, zorder=8, label="Boresight")

    for b, roll in zip(boresights, rolls):
        poly_ra, poly_dec = build_fov_polygon_radec(
            boresight=b,
            roll_rad=float(roll),
            fov_h_rad=float(fov_h_rad),
            fov_v_rad=float(fov_v_rad),
            edge_samples=int(edge_samples),
        )
        for seg_ra, seg_dec in split_by_ra_wrap(poly_ra, poly_dec):
            for sx, sy in split_projected_path(seg_ra, seg_dec, gore_width_deg=gore_width_deg, gore_gap=gore_gap):
                ax.plot(sx, sy, color="#22d3ee", alpha=0.75, linewidth=1.0, zorder=7)

    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)


def save_boresight_scatter(
    *,
    out_file: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    title: str,
    dpi: int,
    gore_width_deg: float,
    gore_gap: float,
) -> None:
    fig, ax = new_figure(title, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    x, y, _ = radec_to_interrupted(ra_deg, dec_deg, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    ax.scatter(x, y, s=2.0, c="#f8e98c", alpha=0.78, linewidths=0.0, zorder=4)
    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)


def save_boresight_hexbin(
    *,
    out_file: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    title: str,
    dpi: int,
    gore_width_deg: float,
    gore_gap: float,
    gridsize: int,
) -> None:
    fig, ax = new_figure(title, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    x, y, _ = radec_to_interrupted(ra_deg, dec_deg, gore_width_deg=gore_width_deg, gore_gap=gore_gap)
    hb = ax.hexbin(
        x,
        y,
        gridsize=int(gridsize),
        mincnt=1,
        bins="log",
        cmap="magma",
        linewidths=0.0,
        zorder=4,
    )
    cbar = fig.colorbar(hb, ax=ax, pad=0.015, fraction=0.035)
    cbar.set_label("Boresight centers per projected bin", color="#e2e8f0")
    cbar.ax.tick_params(colors="#cbd5e1", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)


def random_scene_centers(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(int(count), 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return vectors_to_radec_deg(raw)


def load_scene_centers(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    scene_meta = run_dir.expanduser().resolve() / "scene_metadata.npz"
    if not scene_meta.exists():
        raise FileNotFoundError(f"scene_metadata.npz not found: {scene_meta}")
    with np.load(scene_meta, allow_pickle=False) as data:
        return (
            np.asarray(data["scene_center_ra_deg"], dtype=np.float64),
            np.asarray(data["scene_center_dec_deg"], dtype=np.float64),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Sky Globe Gore sky plots.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tetra3" / "data" / "default_database.npz",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "sky_plots_gores")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gore-width-deg", type=float, default=30.0)
    parser.add_argument("--gore-gap", type=float, default=0.075)
    parser.add_argument("--num-footprints", type=int, default=1)
    parser.add_argument("--boresight-mode", choices=("random", "fibonacci", "catalog"), default="random")
    parser.add_argument("--roll-mode", choices=("random", "zero", "sweep"), default="random")
    parser.add_argument("--fov-h-deg", type=float, default=17.2)
    parser.add_argument("--fov-v-deg", type=float, default=13.0)
    parser.add_argument("--edge-samples", type=int, default=32)
    parser.add_argument("--hexbin-gridsize", type=int, default=145)
    parser.add_argument(
        "--include-random-expd-preview",
        action="store_true",
        help=(
            "Generate a synthetic isotropic Exp. D preview. This is not the real "
            "experiment output; use --run-heatmap with scene_metadata.npz for the exact run."
        ),
    )
    parser.add_argument(
        "--run-heatmap",
        nargs=3,
        action="append",
        metavar=("RUN_DIR", "LABEL", "OUTPUT_NAME"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if abs(360.0 / float(args.gore_width_deg) - round(360.0 / float(args.gore_width_deg))) > 1e-9:
        raise ValueError("--gore-width-deg must divide 360 exactly")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ra_deg, dec_deg, vectors = load_database(args.database)

    rng = np.random.default_rng(int(args.seed))
    boresights = sample_boresights(
        mode=str(args.boresight_mode),
        count=int(args.num_footprints),
        catalog_vectors=vectors,
        rng=rng,
    )
    rolls = sample_rolls(str(args.roll_mode), int(args.num_footprints), rng)

    catalog_png = out_dir / "tetra_catalog_map_sky_globe_gore.png"
    fov_png = out_dir / "fov_footprint_sky_globe_gore.png"
    baseline_png = out_dir / "boresight_baseline_sky_globe_gore.png"

    save_catalog(
        out_file=catalog_png,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        dpi=int(args.dpi),
        gore_width_deg=float(args.gore_width_deg),
        gore_gap=float(args.gore_gap),
    )
    save_fov(
        out_file=fov_png,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        boresights=boresights,
        rolls=rolls,
        fov_h_rad=math.radians(float(args.fov_h_deg)),
        fov_v_rad=math.radians(float(args.fov_v_deg)),
        edge_samples=int(args.edge_samples),
        dpi=int(args.dpi),
        gore_width_deg=float(args.gore_width_deg),
        gore_gap=float(args.gore_gap),
    )
    save_boresight_scatter(
        out_file=baseline_png,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        title="Boresight centers | Baseline | Sky Globe Gores",
        dpi=int(args.dpi),
        gore_width_deg=float(args.gore_width_deg),
        gore_gap=float(args.gore_gap),
    )

    written = [catalog_png, fov_png, baseline_png]

    if bool(args.include_random_expd_preview):
        expd_ra, expd_dec = random_scene_centers(3_265_893, int(args.seed) + 1000)
        expd_png = out_dir / "boresight_expD_sky_globe_gore_preview.png"
        save_boresight_hexbin(
            out_file=expd_png,
            ra_deg=expd_ra,
            dec_deg=expd_dec,
            title="Boresight centers | Exp. D preview | Sky Globe Gores",
            dpi=int(args.dpi),
            gore_width_deg=float(args.gore_width_deg),
            gore_gap=float(args.gore_gap),
            gridsize=int(args.hexbin_gridsize),
        )
        written.append(expd_png)

    for run_args in args.run_heatmap or []:
        run_dir = Path(run_args[0])
        label = str(run_args[1])
        output_name = str(run_args[2])
        run_ra, run_dec = load_scene_centers(run_dir)
        out_file = out_dir / output_name
        save_boresight_hexbin(
            out_file=out_file,
            ra_deg=run_ra,
            dec_deg=run_dec,
            title=f"Boresight centers | {label} | Sky Globe Gores",
            dpi=int(args.dpi),
            gore_width_deg=float(args.gore_width_deg),
            gore_gap=float(args.gore_gap),
            gridsize=int(args.hexbin_gridsize),
        )
        written.append(out_file)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
