#!/usr/bin/env python3
"""Render catalog/FOV/boresight plots in a continuous Mollweide projection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_sky_fov import build_fov_polygon_radec
from visualize_sky_fov import load_database
from visualize_sky_fov import sample_boresights
from visualize_sky_fov import sample_rolls
from visualize_sky_fov import split_by_ra_wrap
from visualize_sky_fov import vectors_to_radec_deg


SQRT2 = math.sqrt(2.0)
X_LIM = 2.0 * SQRT2
Y_LIM = SQRT2


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


def radec_to_mollweide(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert RA/Dec degrees to projected Mollweide x/y coordinates.

    The RA axis is reversed so the visual convention matches the existing RA/Dec
    figures: RA=360 deg appears on the left and RA=0 deg on the right.
    """
    ra = np.mod(np.asarray(ra_deg, dtype=np.float64), 360.0)
    lon_deg = 180.0 - ra
    lat_deg = np.asarray(dec_deg, dtype=np.float64)
    return mollweide_project(np.radians(lon_deg), np.radians(lat_deg))


def style_redondo_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, color="#f8fafc", pad=12)
    ax.set_facecolor("#020617")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-X_LIM * 1.04, X_LIM * 1.04)
    ax.set_ylim(-Y_LIM * 1.12, Y_LIM * 1.10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    boundary = Ellipse(
        (0.0, 0.0),
        width=2.0 * X_LIM,
        height=2.0 * Y_LIM,
        edgecolor="#cbd5e1",
        facecolor="none",
        linewidth=1.35,
        alpha=0.9,
        zorder=9,
    )
    ax.add_patch(boundary)

    grid_color = "#475569"
    grid_linewidth = 0.95
    grid_alpha = 0.95
    grid_zorder = 8
    lon_grid = np.linspace(-math.pi, math.pi, 400)
    for dec in np.arange(-75.0, 76.0, 15.0):
        lat_grid = np.full_like(lon_grid, math.radians(float(dec)))
        x, y = mollweide_project(lon_grid, lat_grid)
        ax.plot(x, y, color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha, zorder=grid_zorder)
        if dec != 0:
            ax.text(-X_LIM * 1.025, y[0], f"{int(dec)}°", color="#cbd5e1", ha="right", va="center", fontsize=8)

    lat_grid = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 300)
    for lon_deg in np.arange(-150.0, 151.0, 30.0):
        lon_line = np.full_like(lat_grid, math.radians(float(lon_deg)))
        x, y = mollweide_project(lon_line, lat_grid)
        ax.plot(x, y, color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha, zorder=grid_zorder)
        x0, y0 = mollweide_project(np.asarray([math.radians(float(lon_deg))]), np.asarray([0.0]))
        ax.text(
            float(x0[0]),
            float(y0[0]) + 0.035,
            f"{int((180.0 - lon_deg) % 360.0)}",
            color="#cbd5e1",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.text(0.0, -Y_LIM * 1.08, "Right Ascension (deg)", color="#e2e8f0", ha="center", va="top", fontsize=10)
    ax.text(-X_LIM * 1.105, 0.0, "Declination (deg)", color="#e2e8f0", ha="center", va="center", rotation=90, fontsize=10)


def new_redondo_figure(title: str, *, figsize: tuple[float, float] = (12.8, 6.6)) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=figsize, facecolor="#020617")
    ax = fig.add_subplot(111)
    style_redondo_axis(ax, title)
    return fig, ax


def save_catalog_plot(*, out_file: Path, ra_deg: np.ndarray, dec_deg: np.ndarray, dpi: int) -> None:
    fig, ax = new_redondo_figure(f"Tetra Catalog Sky Map (Mollweide) - N={ra_deg.shape[0]}")
    lon, lat = radec_to_mollweide(ra_deg, dec_deg)
    ax.scatter(lon, lat, s=2.0, c="#e2e8f0", alpha=0.72, linewidths=0.0)
    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)


def save_fov_plot(
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
) -> None:
    fig, ax = new_redondo_figure("FOV Footprint on Catalog Sky Map (Mollweide)")
    lon, lat = radec_to_mollweide(ra_deg, dec_deg)
    ax.scatter(lon, lat, s=1.0, c="#94a3b8", alpha=0.25, linewidths=0.0)

    center_ra, center_dec = vectors_to_radec_deg(boresights)
    center_lon, center_lat = radec_to_mollweide(center_ra, center_dec)
    ax.scatter(center_lon, center_lat, s=22.0, c="#f59e0b", alpha=0.95, linewidths=0.0, label="Boresight")

    for boresight, roll in zip(boresights, rolls):
        poly_ra, poly_dec = build_fov_polygon_radec(
            boresight=boresight,
            roll_rad=float(roll),
            fov_h_rad=float(fov_h_rad),
            fov_v_rad=float(fov_v_rad),
            edge_samples=int(edge_samples),
        )
        for seg_ra, seg_dec in split_by_ra_wrap(poly_ra, poly_dec):
            seg_lon, seg_lat = radec_to_mollweide(seg_ra, seg_dec)
            ax.plot(seg_lon, seg_lat, color="#22d3ee", alpha=0.65, linewidth=0.9)

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
    weights: np.ndarray | None = None,
) -> None:
    fig, ax = new_redondo_figure(title)
    x, y = radec_to_mollweide(ra_deg, dec_deg)
    if weights is None:
        ax.scatter(x, y, s=2.0, c="#f59e0b", alpha=0.75, linewidths=0.0)
    else:
        values = np.asarray(weights, dtype=np.float64)
        ax.scatter(
            x,
            y,
            s=2.0,
            c=values,
            cmap="magma",
            norm=LogNorm(vmin=1.0, vmax=max(float(values.max()), 1.0)),
            alpha=0.85,
            linewidths=0.0,
        )
    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)


def save_boresight_hexbin(
    *,
    out_file: Path,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    title: str,
    colorbar_label: str,
    dpi: int,
    gridsize: int,
) -> None:
    fig, ax = new_redondo_figure(title)
    x, y = radec_to_mollweide(ra_deg, dec_deg)
    hb = ax.hexbin(
        x,
        y,
        gridsize=int(gridsize),
        mincnt=1,
        bins="log",
        cmap="magma",
        linewidths=0.0,
    )
    cbar = fig.colorbar(hb, ax=ax, pad=0.04, fraction=0.05)
    cbar.set_label(colorbar_label, color="#e2e8f0")
    cbar.ax.tick_params(colors="#cbd5e1", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
    plt.close(fig)


def load_scene_centers(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    scene_meta = run_dir.expanduser().resolve() / "scene_metadata.npz"
    if not scene_meta.exists():
        raise FileNotFoundError(f"scene_metadata.npz not found: {scene_meta}")
    with np.load(scene_meta, allow_pickle=False) as data:
        if "scene_center_ra_deg" not in data or "scene_center_dec_deg" not in data:
            raise RuntimeError(f"scene center arrays not found in {scene_meta}")
        return (
            np.asarray(data["scene_center_ra_deg"], dtype=np.float64),
            np.asarray(data["scene_center_dec_deg"], dtype=np.float64),
        )


def random_scene_centers(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(int(count), 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return vectors_to_radec_deg(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create continuous Mollweide versions of RA/Dec sky plots.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tetra3" / "data" / "default_database.npz",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "sky_plots_redondo")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-footprints", type=int, default=1)
    parser.add_argument("--boresight-mode", choices=("random", "fibonacci", "catalog"), default="random")
    parser.add_argument("--roll-mode", choices=("random", "zero", "sweep"), default="random")
    parser.add_argument("--fov-h-deg", type=float, default=17.2)
    parser.add_argument("--fov-v-deg", type=float, default=13.0)
    parser.add_argument("--edge-samples", type=int, default=32)
    parser.add_argument("--hexbin-gridsize", type=int, default=170)
    parser.add_argument(
        "--run-heatmap",
        nargs=3,
        action="append",
        metavar=("RUN_DIR", "LABEL", "OUTPUT_NAME"),
        help="Render a run heatmap from scene_metadata.npz.",
    )
    parser.add_argument(
        "--random-heatmap",
        nargs=3,
        action="append",
        metavar=("COUNT", "LABEL", "OUTPUT_NAME"),
        help="Render an isotropic random preview heatmap.",
    )
    parser.add_argument(
        "--include-random-expd-preview",
        action="store_true",
        help=(
            "Generate a synthetic isotropic Exp. D preview. This is not the real "
            "experiment output; use --run-heatmap with scene_metadata.npz for the exact run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    catalog_png = out_dir / "tetra_catalog_map_redondo.png"
    fov_png = out_dir / "fov_footprint_redondo.png"
    baseline_png = out_dir / "boresight_baseline_redondo.png"
    save_catalog_plot(out_file=catalog_png, ra_deg=ra_deg, dec_deg=dec_deg, dpi=int(args.dpi))
    save_fov_plot(
        out_file=fov_png,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        boresights=boresights,
        rolls=rolls,
        fov_h_rad=math.radians(float(args.fov_h_deg)),
        fov_v_rad=math.radians(float(args.fov_v_deg)),
        edge_samples=int(args.edge_samples),
        dpi=int(args.dpi),
    )

    save_boresight_scatter(
        out_file=baseline_png,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        title="Boresight Centers | Baseline | guide stars",
        dpi=int(args.dpi),
        weights=np.full(ra_deg.shape[0], 360.0, dtype=np.float64),
    )

    written = [catalog_png, fov_png, baseline_png]

    if bool(args.include_random_expd_preview):
        random_ra, random_dec = random_scene_centers(3_265_893, int(args.seed) + 1000)
        out_file = out_dir / "boresight_expD_redondo_preview.png"
        save_boresight_hexbin(
            out_file=out_file,
            ra_deg=random_ra,
            dec_deg=random_dec,
            title="Boresight Centers | Exp. D preview | isotropic sample",
            colorbar_label="Boresight centers per projected bin",
            dpi=int(args.dpi),
            gridsize=int(args.hexbin_gridsize),
        )
        written.append(out_file)

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
            title=f"Boresight Centers | {label}",
            colorbar_label="Boresight centers per projected bin",
            dpi=int(args.dpi),
            gridsize=int(args.hexbin_gridsize),
        )
        written.append(out_file)

    for random_args in args.random_heatmap or []:
        count = int(random_args[0])
        label = str(random_args[1])
        output_name = str(random_args[2])
        rand_ra, rand_dec = random_scene_centers(count, int(args.seed) + len(written))
        out_file = out_dir / output_name
        save_boresight_hexbin(
            out_file=out_file,
            ra_deg=rand_ra,
            dec_deg=rand_dec,
            title=f"Boresight Centers | {label}",
            colorbar_label="Boresight centers per projected bin",
            dpi=int(args.dpi),
            gridsize=int(args.hexbin_gridsize),
        )
        written.append(out_file)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
