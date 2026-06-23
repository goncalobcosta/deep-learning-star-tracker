#!/usr/bin/env python3
"""Visualize Tetra stars and FOV footprints on an RA/Dec sky map.

This script creates two PNGs:
1) `catalog_ra_dec.png`: all stars in the Tetra database plotted in Right Ascension/Declination.
2) `fov_footprints_ra_dec.png`: same sky map with rectangular FOV footprints and boresight centers.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable
from typing import Sequence
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_database(database_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    db = database_path.expanduser().resolve()
    with np.load(db, allow_pickle=False) as data:
        if "star_table" not in data:
            raise RuntimeError("Database must contain star_table")
        star_table = np.asarray(data["star_table"], dtype=np.float64)

    if star_table.ndim != 2 or star_table.shape[1] < 6:
        raise RuntimeError("Invalid star_table format")

    ra_deg = np.degrees(star_table[:, 0]) % 360.0
    dec_deg = np.degrees(star_table[:, 1])
    vectors = star_table[:, 2:5].astype(np.float64, copy=False)
    return ra_deg, dec_deg, vectors


def camera_basis(boresight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, boresight))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    j0 = np.cross(up, boresight)
    norm_j = float(np.linalg.norm(j0))
    if norm_j < 1e-12:
        j0 = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), boresight)
        norm_j = float(np.linalg.norm(j0))
    j0 = j0 / norm_j

    k0 = np.cross(boresight, j0)
    k0 = k0 / np.linalg.norm(k0)
    return j0, k0


def vectors_to_radec_deg(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = v[:, 0]
    y = v[:, 1]
    z = np.clip(v[:, 2], -1.0, 1.0)
    ra = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(z))
    return ra, dec


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(n, 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw.astype(np.float64)


def fibonacci_sphere(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    z = 1.0 - (2.0 * (idx + 0.5) / float(n))
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = (2.0 * math.pi * idx) / ((1.0 + math.sqrt(5.0)) / 2.0)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack((x, y, z), axis=1).astype(np.float64)


def sample_boresights(
    *,
    mode: str,
    count: int,
    catalog_vectors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("num-footprints must be > 0")

    if mode == "random":
        return random_unit_vectors(count, rng)
    if mode == "fibonacci":
        return fibonacci_sphere(count)
    if mode == "catalog":
        replace = bool(count > catalog_vectors.shape[0])
        idx = rng.choice(catalog_vectors.shape[0], size=count, replace=replace)
        return catalog_vectors[idx].astype(np.float64, copy=False)
    raise ValueError(f"Unsupported boresight mode: {mode}")


def sample_rolls(mode: str, count: int, rng: np.random.Generator) -> np.ndarray:
    if mode == "random":
        return rng.uniform(0.0, 2.0 * math.pi, size=count).astype(np.float64)
    if mode == "zero":
        return np.zeros(count, dtype=np.float64)
    if mode == "sweep":
        return np.linspace(0.0, 2.0 * math.pi, num=count, endpoint=False, dtype=np.float64)
    raise ValueError(f"Unsupported roll mode: {mode}")


def build_fov_polygon_radec(
    *,
    boresight: np.ndarray,
    roll_rad: float,
    fov_h_rad: float,
    fov_v_rad: float,
    edge_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    tx = math.tan(fov_h_rad / 2.0)
    ty = math.tan(fov_v_rad / 2.0)

    j0, k0 = camera_basis(boresight)
    c = math.cos(float(roll_rad))
    s = math.sin(float(roll_rad))
    j = c * j0 + s * k0
    k = -s * j0 + c * k0

    u = np.linspace(-1.0, 1.0, edge_samples, dtype=np.float64)
    v = np.linspace(-1.0, 1.0, edge_samples, dtype=np.float64)

    vectors: list[np.ndarray] = []
    for uu in u:
        vectors.append(boresight + tx * uu * j + ty * (+1.0) * k)
    for vv in v[::-1]:
        vectors.append(boresight + tx * (+1.0) * j + ty * vv * k)
    for uu in u[::-1]:
        vectors.append(boresight + tx * uu * j + ty * (-1.0) * k)
    for vv in v:
        vectors.append(boresight + tx * (-1.0) * j + ty * vv * k)

    poly = np.asarray(vectors, dtype=np.float64)
    poly /= np.linalg.norm(poly, axis=1, keepdims=True)
    return vectors_to_radec_deg(poly)


def split_by_ra_wrap(ra_deg: np.ndarray, dec_deg: np.ndarray, jump_deg: float = 180.0) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if ra_deg.size <= 1:
        yield ra_deg, dec_deg
        return

    start = 0
    for i in range(1, ra_deg.size):
        if abs(float(ra_deg[i] - ra_deg[i - 1])) > jump_deg:
            if i - start >= 2:
                yield ra_deg[start:i], dec_deg[start:i]
            start = i
    if ra_deg.size - start >= 2:
        yield ra_deg[start:], dec_deg[start:]


def camera_frame_components(
    *,
    catalog_vectors: np.ndarray,
    boresight: np.ndarray,
    roll_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    j0, k0 = camera_basis(boresight)
    c = math.cos(float(roll_rad))
    s = math.sin(float(roll_rad))
    j = c * j0 + s * k0
    k = -s * j0 + c * k0

    i = catalog_vectors @ boresight
    jc = catalog_vectors @ j
    kc = catalog_vectors @ k
    return i, jc, kc


def stars_inside_fov(
    *,
    catalog_vectors: np.ndarray,
    boresight: np.ndarray,
    roll_rad: float,
    fov_h_rad: float,
    fov_v_rad: float,
) -> np.ndarray:
    i, jc, kc = camera_frame_components(
        catalog_vectors=catalog_vectors,
        boresight=boresight,
        roll_rad=roll_rad,
    )
    tx = math.tan(fov_h_rad / 2.0)
    ty = math.tan(fov_v_rad / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        qj = jc / i
        qk = kc / i
    inside = (i > 0.0) & (np.abs(qj) <= tx) & (np.abs(qk) <= ty)
    return inside


def ra_in_window(ra_deg: float, ra_min: float, ra_max: float) -> bool:
    lo = min(float(ra_min), float(ra_max))
    hi = max(float(ra_min), float(ra_max))
    if (hi - lo) >= 360.0:
        return True
    ra = float(ra_deg) % 360.0
    return bool(
        lo <= ra <= hi
        or lo <= (ra - 360.0) <= hi
        or lo <= (ra + 360.0) <= hi
    )


def dec_in_window(dec_deg: float, dec_min: float, dec_max: float) -> bool:
    lo = max(-90.0, min(float(dec_min), float(dec_max)))
    hi = min(90.0, max(float(dec_min), float(dec_max)))
    if (hi - lo) >= 180.0:
        return True
    dec = float(dec_deg)
    return bool(lo <= dec <= hi)


def iter_ra_window_segments(ra_min: float, ra_max: float) -> Iterable[tuple[float, float]]:
    lo = min(float(ra_min), float(ra_max))
    hi = max(float(ra_min), float(ra_max))
    span = hi - lo
    if span >= 360.0:
        yield (0.0, 360.0)
        return

    current = lo
    eps = 1e-9
    while current < (hi - eps):
        next_boundary = math.floor(current / 360.0 + 1.0) * 360.0
        seg_hi = min(hi, next_boundary)
        seg_lo_norm = current % 360.0
        seg_hi_norm = seg_hi % 360.0
        if abs(seg_hi_norm) < eps and seg_hi > current:
            seg_hi_norm = 360.0
        yield (seg_lo_norm, seg_hi_norm)
        current = seg_hi


def style_axis(
    ax: plt.Axes,
    title: str,
    *,
    ra_min: float | None = None,
    ra_max: float | None = None,
    dec_min: float | None = None,
    dec_max: float | None = None,
) -> None:
    ax.set_title(title, color="#f8fafc")
    ax.set_xlabel("Right Ascension (deg)", color="#e2e8f0")
    ax.set_ylabel("Declination (deg)", color="#e2e8f0")
    if ra_min is None or ra_max is None:
        ax.set_xlim(360.0, 0.0)
        ax.set_xticks(np.arange(0.0, 361.0, 30.0))
    else:
        left = float(ra_min)
        right = float(ra_max)
        ax.set_xlim(left, right)
        ax.set_xticks(np.linspace(left, right, num=7))

    if dec_min is None or dec_max is None:
        ax.set_ylim(-90.0, 90.0)
        ax.set_yticks(np.arange(-90.0, 91.0, 15.0))
    else:
        lo = float(dec_min)
        hi = float(dec_max)
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.linspace(lo, hi, num=7))

    ax.tick_params(colors="#cbd5e1", labelsize=8)
    ax.grid(color="#334155", linewidth=0.6, alpha=0.7)
    ax.set_facecolor("#020617")


def save_catalog_plot(*, out_file: Path, ra_deg: np.ndarray, dec_deg: np.ndarray, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#020617")
    style_axis(ax, f"Tetra Catalog Sky Map (RA/Dec) - N={ra_deg.shape[0]}")
    ax.scatter(ra_deg, dec_deg, s=2.0, c="#e2e8f0", alpha=0.7, linewidths=0.0)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor())
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
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#020617")
    style_axis(ax, "RA/Dec with FOV Footprints")
    ax.scatter(ra_deg, dec_deg, s=1.0, c="#94a3b8", alpha=0.25, linewidths=0.0)

    center_ra, center_dec = vectors_to_radec_deg(boresights)
    ax.scatter(center_ra, center_dec, s=18.0, c="#f59e0b", alpha=0.9, linewidths=0.0, label="Boresight centers")

    for b, roll in zip(boresights, rolls):
        poly_ra, poly_dec = build_fov_polygon_radec(
            boresight=b,
            roll_rad=float(roll),
            fov_h_rad=fov_h_rad,
            fov_v_rad=fov_v_rad,
            edge_samples=edge_samples,
        )
        for seg_ra, seg_dec in split_by_ra_wrap(poly_ra, poly_dec):
            ax.plot(seg_ra, seg_dec, color="#22d3ee", alpha=0.35, linewidth=0.7)

    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_single_scene_plot(
    *,
    out_file: Path,
    catalog_vectors: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    boresight: np.ndarray,
    roll_rad: float,
    fov_h_rad: float,
    fov_v_rad: float,
    edge_samples: int,
    dpi: int,
    scene_label: str,
) -> None:
    inside = stars_inside_fov(
        catalog_vectors=catalog_vectors,
        boresight=boresight,
        roll_rad=roll_rad,
        fov_h_rad=fov_h_rad,
        fov_v_rad=fov_v_rad,
    )
    poly_ra, poly_dec = build_fov_polygon_radec(
        boresight=boresight,
        roll_rad=float(roll_rad),
        fov_h_rad=fov_h_rad,
        fov_v_rad=fov_v_rad,
        edge_samples=edge_samples,
    )
    center_ra, center_dec = vectors_to_radec_deg(boresight[None, :])

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#020617")
    style_axis(
        ax,
        f"Scene {scene_label} | stars in FOV: {int(np.sum(inside))}",
    )
    ax.scatter(ra_deg, dec_deg, s=1.0, c="#64748b", alpha=0.30, linewidths=0.0, label="Catalog stars")
    ax.scatter(ra_deg[inside], dec_deg[inside], s=8.0, c="#facc15", alpha=0.95, linewidths=0.0, label="Stars inside FOV")
    for seg_ra, seg_dec in split_by_ra_wrap(poly_ra, poly_dec):
        ax.plot(seg_ra, seg_dec, color="#22d3ee", alpha=0.9, linewidth=1.1)
    ax.scatter(center_ra, center_dec, s=28.0, c="#fb7185", alpha=0.95, linewidths=0.0, label="Boresight")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def export_scene_plots(
    *,
    out_dir: Path,
    catalog_vectors: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    boresights: np.ndarray,
    rolls: np.ndarray,
    fov_h_rad: float,
    fov_v_rad: float,
    edge_samples: int,
    dpi: int,
    scene_indices: Sequence[int],
) -> list[Path]:
    scene_dir = out_dir / "scene_plots"
    scene_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx in scene_indices:
        if idx < 0 or idx >= boresights.shape[0]:
            raise IndexError(f"scene index out of range: {idx}")
        out_file = scene_dir / f"scene_{idx:04d}_ra_dec.png"
        save_single_scene_plot(
            out_file=out_file,
            catalog_vectors=catalog_vectors,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            boresight=boresights[idx],
            roll_rad=float(rolls[idx]),
            fov_h_rad=fov_h_rad,
            fov_v_rad=fov_v_rad,
            edge_samples=edge_samples,
            dpi=dpi,
            scene_label=f"{idx:04d}",
        )
        written.append(out_file)
    return written


def _load_scene_metadata(scene_meta_path: Path) -> dict[str, np.ndarray]:
    with np.load(scene_meta_path, allow_pickle=False) as data:
        out = {k: np.asarray(data[k]) for k in data.files}
    required = [
        "scene_boresight_xyz",
        "scene_roll_degree",
        "scene_real_star_start",
        "scene_real_star_count",
        "scene_real_star_id",
    ]
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"scene_metadata missing keys: {missing}")
    return out


def generate_run_sky_plots(
    *,
    run_dir: Path,
    database_path: Path,
    out_dir: Path,
    ra_min: float = 150.0,
    ra_max: float = 180.0,
    dec_min: float = -90.0,
    dec_max: float = 90.0,
    plot_ra_min: float | None = None,
    plot_ra_max: float | None = None,
    plot_dec_min: float | None = None,
    plot_dec_max: float | None = None,
    focus_ra_min: float | None = None,
    focus_ra_max: float | None = None,
    focus_dec_min: float | None = None,
    focus_dec_max: float | None = None,
    scene_indices: Sequence[int] | None = None,
    every_nth_scene: int = 1,
    dpi: int = 170,
    edge_samples: int = 24,
    fov_h_deg: float = 17.2,
    fov_v_deg: float = 13.0,
) -> int:
    if int(every_nth_scene) <= 0:
        raise ValueError("every_nth_scene must be > 0")

    run_dir = run_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_meta_path = run_dir / "scene_metadata.npz"
    if not scene_meta_path.exists():
        raise FileNotFoundError(f"scene_metadata.npz not found: {scene_meta_path}")

    scene_meta = _load_scene_metadata(scene_meta_path)
    boresights = np.asarray(scene_meta["scene_boresight_xyz"], dtype=np.float32)
    rolls = np.asarray(scene_meta["scene_roll_degree"], dtype=np.float32)
    real_start = np.asarray(scene_meta["scene_real_star_start"], dtype=np.int64)
    real_count = np.asarray(scene_meta["scene_real_star_count"], dtype=np.int32)
    real_ids_flat = np.asarray(scene_meta["scene_real_star_id"], dtype=np.int32)

    scene_ra = (
        np.asarray(scene_meta["scene_center_ra_deg"], dtype=np.float32)
        if "scene_center_ra_deg" in scene_meta
        else np.asarray([vectors_to_radec_deg(boresights[i : i + 1])[0][0] for i in range(boresights.shape[0])], dtype=np.float32)
    )
    scene_dec = (
        np.asarray(scene_meta["scene_center_dec_deg"], dtype=np.float32)
        if "scene_center_dec_deg" in scene_meta
        else np.asarray([vectors_to_radec_deg(boresights[i : i + 1])[1][0] for i in range(boresights.shape[0])], dtype=np.float32)
    )

    cum_unique = np.asarray(scene_meta.get("cumulative_unique_seen_stars", np.zeros(boresights.shape[0], dtype=np.int32)))
    cum_min = np.asarray(scene_meta.get("cumulative_min_appear_count", np.zeros(boresights.shape[0], dtype=np.int32)))
    cum_mean = np.asarray(scene_meta.get("cumulative_mean_appear_count", np.zeros(boresights.shape[0], dtype=np.float32)))
    cum_max = np.asarray(scene_meta.get("cumulative_max_appear_count", np.zeros(boresights.shape[0], dtype=np.int32)))

    ra_deg, dec_deg, catalog_vectors = load_database(database_path)
    seen_mask = np.zeros(catalog_vectors.shape[0], dtype=bool)
    focus_mask = None
    focus_appear_count = None
    focus_star_count = 0
    if (
        focus_ra_min is not None
        and focus_ra_max is not None
        and focus_dec_min is not None
        and focus_dec_max is not None
    ):
        focus_mask = np.array(
            [
                ra_in_window(float(ra), float(focus_ra_min), float(focus_ra_max))
                and dec_in_window(float(dec), float(focus_dec_min), float(focus_dec_max))
                for ra, dec in zip(ra_deg.tolist(), dec_deg.tolist())
            ],
            dtype=bool,
        )
        focus_star_count = int(np.sum(focus_mask))
        focus_appear_count = np.zeros(focus_mask.shape[0], dtype=np.int32)

    fov_h_rad = math.radians(float(fov_h_deg))
    fov_v_rad = math.radians(float(fov_v_deg))
    exported = 0

    selected_scenes: set[int] | None = None
    if scene_indices is not None:
        selected_scenes = set(int(x) for x in scene_indices)

    for scene_idx in range(int(boresights.shape[0])):
        start = int(real_start[scene_idx])
        count = int(real_count[scene_idx])
        current_ids = real_ids_flat[start : start + count]
        if current_ids.size > 0:
            seen_mask[current_ids] = True
            if focus_mask is not None and focus_appear_count is not None:
                focus_ids = current_ids[focus_mask[current_ids]]
                if focus_ids.size > 0:
                    focus_appear_count[focus_ids] += 1

        if selected_scenes is not None and scene_idx not in selected_scenes:
            continue
        if selected_scenes is None and scene_idx % int(every_nth_scene) != 0:
            continue
        if not ra_in_window(float(scene_ra[scene_idx]), float(ra_min), float(ra_max)):
            continue
        if not dec_in_window(float(scene_dec[scene_idx]), float(dec_min), float(dec_max)):
            continue

        boresight = boresights[scene_idx].astype(np.float64, copy=False)
        roll_rad = math.radians(float(rolls[scene_idx]))
        poly_ra, poly_dec = build_fov_polygon_radec(
            boresight=boresight,
            roll_rad=roll_rad,
            fov_h_rad=fov_h_rad,
            fov_v_rad=fov_v_rad,
            edge_samples=int(edge_samples),
        )

        fig = plt.figure(figsize=(13.6, 6), facecolor="#020617")
        grid = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.9, 1.4], wspace=0.06)
        ax = fig.add_subplot(grid[0, 0])
        info_ax = fig.add_subplot(grid[0, 1])

        style_axis(
            ax,
            f"Run Sky Coverage | scene={scene_idx}",
            ra_min=float(plot_ra_min) if plot_ra_min is not None else float(ra_min),
            ra_max=float(plot_ra_max) if plot_ra_max is not None else float(ra_max),
            dec_min=float(plot_dec_min) if plot_dec_min is not None else float(dec_min),
            dec_max=float(plot_dec_max) if plot_dec_max is not None else float(dec_max),
        )
        # Not-yet-seen stars: visible but still subdued.
        ax.scatter(ra_deg, dec_deg, s=1.1, c="#6b7280", alpha=0.55, linewidths=0.0)
        ax.scatter(ra_deg[seen_mask], dec_deg[seen_mask], s=1.8, c="#f8fafc", alpha=0.78, linewidths=0.0)
        if current_ids.size > 0:
            ax.scatter(ra_deg[current_ids], dec_deg[current_ids], s=9.0, c="#facc15", alpha=0.98, linewidths=0.0)

        for seg_ra, seg_dec in split_by_ra_wrap(poly_ra, poly_dec):
            ax.plot(seg_ra, seg_dec, color="#fb7185", alpha=0.98, linewidth=1.2)

        if (
            focus_ra_min is not None
            and focus_ra_max is not None
            and focus_dec_min is not None
            and focus_dec_max is not None
        ):
            from matplotlib.patches import Rectangle

            fra0 = float(focus_ra_min)
            fra1 = float(focus_ra_max)
            fdc0 = float(focus_dec_min)
            fdc1 = float(focus_dec_max)
            bottom = min(fdc0, fdc1)
            top = max(fdc0, fdc1)
            for left, right in iter_ra_window_segments(fra0, fra1):
                rect = Rectangle(
                    (left, bottom),
                    max(0.0, right - left),
                    max(0.0, top - bottom),
                    linewidth=2.6,
                    edgecolor="#60a5fa",
                    facecolor="none",
                    alpha=1.0,
                    zorder=10,
                )
                ax.add_patch(rect)

        info_ax.set_facecolor("#020617")
        info_ax.set_xticks([])
        info_ax.set_yticks([])
        for spine in info_ax.spines.values():
            spine.set_color("#334155")

        if focus_mask is not None and focus_appear_count is not None:
            focus_current_ids = current_ids[focus_mask[current_ids]] if current_ids.size > 0 else current_ids
            unique_seen_value = int(np.sum(focus_appear_count > 0))
            min_value = int(np.min(focus_appear_count[focus_mask])) if focus_star_count > 0 else 0
            mean_value = float(np.sum(focus_appear_count[focus_mask]) / max(focus_star_count, 1))
            max_value = int(np.max(focus_appear_count[focus_mask])) if focus_star_count > 0 else 0
            stars_in_scene_value = int(focus_current_ids.size)
            appear_lines = [
                "appear_count in",
                f"analysis area: {min_value}/{mean_value:.3f}/{max_value}",
            ]
            unique_line = f"unique_seen in area: {unique_seen_value}"
        else:
            unique_seen_value = int(cum_unique[scene_idx]) if cum_unique.size > scene_idx else int(np.sum(seen_mask))
            min_value = int(cum_min[scene_idx]) if cum_min.size > scene_idx else 0
            mean_value = float(cum_mean[scene_idx]) if cum_mean.size > scene_idx else 0.0
            max_value = int(cum_max[scene_idx]) if cum_max.size > scene_idx else 0
            stars_in_scene_value = int(current_ids.size)
            appear_lines = [f"appear_count min/mean/max: {min_value}/{mean_value:.3f}/{max_value}"]
            unique_line = f"unique_seen_stars: {unique_seen_value}"

        info_lines = [
            f"scene_idx: {scene_idx}",
            f"stars_in_scene: {stars_in_scene_value}",
            unique_line,
            *appear_lines,
        ]
        info_ax.text(0.06, 0.95, "Scene Info", va="top", ha="left", fontsize=10, color="#f8fafc", fontweight="bold")
        info_ax.text(0.06, 0.88, "\n".join(info_lines), va="top", ha="left", fontsize=8, color="#e2e8f0", linespacing=1.35)
        info_ax.text(0.06, 0.34, "Legend", va="top", ha="left", fontsize=9, color="#f8fafc", fontweight="bold")
        info_ax.text(0.06, 0.28, "Dark gray: not seen yet", color="#9ca3af", fontsize=8, va="top", ha="left")
        info_ax.text(0.06, 0.23, "White: seen up to scene", color="#f8fafc", fontsize=8, va="top", ha="left")
        info_ax.text(0.06, 0.18, "Yellow: current scene", color="#facc15", fontsize=8, va="top", ha="left")
        info_ax.text(0.06, 0.13, "Pink: FOV border", color="#fb7185", fontsize=8, va="top", ha="left")
        if (
            focus_ra_min is not None
            and focus_ra_max is not None
            and focus_dec_min is not None
            and focus_dec_max is not None
        ):
            info_ax.text(0.06, 0.08, "Blue thick box: analysis area", color="#93c5fd", fontsize=8, va="top", ha="left")

        fig.subplots_adjust(left=0.05, right=0.985, top=0.93, bottom=0.09, wspace=0.07)

        out_file = out_dir / f"scene_{scene_idx:08d}.png"
        fig.savefig(out_file, dpi=int(dpi), facecolor=fig.get_facecolor())
        plt.close(fig)
        exported += 1

    return int(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Tetra star catalog and FOV footprints on RA/Dec.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory with scene_metadata.npz. If set, script exports per-scene sky_plots for this run.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tetra3" / "data" / "default_database.npz",
        help="Path to Tetra database (.npz).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sky_plots",
        help="Output directory for generated PNG files.",
    )
    parser.add_argument(
        "--boresight-mode",
        type=str,
        choices=("random", "fibonacci", "catalog"),
        default="random",
        help="How to sample boresight centers for FOV footprints.",
    )
    parser.add_argument(
        "--roll-mode",
        type=str,
        choices=("random", "zero", "sweep"),
        default="random",
        help="How to sample roll angle for each footprint.",
    )
    parser.add_argument("--num-footprints", type=int, default=120, help="How many FOV footprints to draw.")
    parser.add_argument("--fov-h-deg", type=float, default=17.2, help="Horizontal FOV in degrees.")
    parser.add_argument("--fov-v-deg", type=float, default=13.0, help="Vertical FOV in degrees.")
    parser.add_argument("--edge-samples", type=int, default=24, help="Samples per FOV edge.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--dpi", type=int, default=180, help="Output figure DPI.")
    parser.add_argument("--ra-min", type=float, default=150.0, help="RA window minimum (run mode).")
    parser.add_argument("--ra-max", type=float, default=180.0, help="RA window maximum (run mode).")
    parser.add_argument("--dec-min", type=float, default=-90.0, help="Declination window minimum (run mode).")
    parser.add_argument("--dec-max", type=float, default=90.0, help="Declination window maximum (run mode).")
    parser.add_argument("--plot-ra-min", type=float, default=None, help="RA axis minimum for plots (run mode).")
    parser.add_argument("--plot-ra-max", type=float, default=None, help="RA axis maximum for plots (run mode).")
    parser.add_argument("--plot-dec-min", type=float, default=None, help="Declination axis minimum for plots (run mode).")
    parser.add_argument("--plot-dec-max", type=float, default=None, help="Declination axis maximum for plots (run mode).")
    parser.add_argument("--focus-ra-min", type=float, default=None, help="RA minimum of highlighted analysis box (run mode).")
    parser.add_argument("--focus-ra-max", type=float, default=None, help="RA maximum of highlighted analysis box (run mode).")
    parser.add_argument("--focus-dec-min", type=float, default=None, help="Declination minimum of highlighted analysis box (run mode).")
    parser.add_argument("--focus-dec-max", type=float, default=None, help="Declination maximum of highlighted analysis box (run mode).")
    parser.add_argument("--run-scene-index", type=int, nargs="*", default=None, help="Explicit run scene indices to render (run mode).")
    parser.add_argument("--every-nth-scene", type=int, default=1, help="Render one frame every N scenes (run mode).")
    parser.add_argument(
        "--export-scene-plots",
        type=int,
        default=0,
        help="Export first N per-scene RA/Dec plots with stars inside FOV highlighted (0 disables).",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        nargs="*",
        default=None,
        help="Specific scene indices to export (overrides only by adding to --export-scene-plots set).",
    )
    args = parser.parse_args()

    if args.num_footprints <= 0:
        raise ValueError("num-footprints must be > 0")
    if args.edge_samples < 4:
        raise ValueError("edge-samples must be >= 4")
    if args.fov_h_deg <= 0.0 or args.fov_v_deg <= 0.0:
        raise ValueError("fov-h-deg and fov-v-deg must be > 0")
    if args.every_nth_scene <= 0:
        raise ValueError("every-nth-scene must be > 0")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir is not None:
        exported = generate_run_sky_plots(
            run_dir=args.run_dir.expanduser().resolve(),
            database_path=args.database.expanduser().resolve(),
            out_dir=out_dir,
            ra_min=float(args.ra_min),
            ra_max=float(args.ra_max),
            dec_min=float(args.dec_min),
            dec_max=float(args.dec_max),
            plot_ra_min=(float(args.plot_ra_min) if args.plot_ra_min is not None else None),
            plot_ra_max=(float(args.plot_ra_max) if args.plot_ra_max is not None else None),
            plot_dec_min=(float(args.plot_dec_min) if args.plot_dec_min is not None else None),
            plot_dec_max=(float(args.plot_dec_max) if args.plot_dec_max is not None else None),
            focus_ra_min=(float(args.focus_ra_min) if args.focus_ra_min is not None else None),
            focus_ra_max=(float(args.focus_ra_max) if args.focus_ra_max is not None else None),
            focus_dec_min=(float(args.focus_dec_min) if args.focus_dec_min is not None else None),
            focus_dec_max=(float(args.focus_dec_max) if args.focus_dec_max is not None else None),
            scene_indices=(list(args.run_scene_index) if args.run_scene_index else None),
            every_nth_scene=int(args.every_nth_scene),
            dpi=int(args.dpi),
            edge_samples=int(args.edge_samples),
            fov_h_deg=float(args.fov_h_deg),
            fov_v_deg=float(args.fov_v_deg),
        )
        print(f"Run mode exported frames: {exported}")
        print(f"Output directory: {out_dir}")
        return

    ra_deg, dec_deg, vectors = load_database(args.database)
    rng = np.random.default_rng(int(args.seed))
    boresights = sample_boresights(
        mode=args.boresight_mode,
        count=int(args.num_footprints),
        catalog_vectors=vectors,
        rng=rng,
    )
    rolls = sample_rolls(args.roll_mode, int(args.num_footprints), rng)

    fov_h_rad = math.radians(float(args.fov_h_deg))
    fov_v_rad = math.radians(float(args.fov_v_deg))

    catalog_png = out_dir / "catalog_ra_dec.png"
    footprint_png = out_dir / "fov_footprints_ra_dec.png"

    save_catalog_plot(out_file=catalog_png, ra_deg=ra_deg, dec_deg=dec_deg, dpi=int(args.dpi))
    save_fov_plot(
        out_file=footprint_png,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        boresights=boresights,
        rolls=rolls,
        fov_h_rad=fov_h_rad,
        fov_v_rad=fov_v_rad,
        edge_samples=int(args.edge_samples),
        dpi=int(args.dpi),
    )

    scene_indices: list[int] = []
    if int(args.export_scene_plots) > 0:
        n = min(int(args.export_scene_plots), boresights.shape[0])
        scene_indices.extend(list(range(n)))
    if args.scene_index:
        scene_indices.extend([int(x) for x in args.scene_index])
    if scene_indices:
        scene_indices = sorted(set(scene_indices))
        written = export_scene_plots(
            out_dir=out_dir,
            catalog_vectors=vectors,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            boresights=boresights,
            rolls=rolls,
            fov_h_rad=fov_h_rad,
            fov_v_rad=fov_v_rad,
            edge_samples=int(args.edge_samples),
            dpi=int(args.dpi),
            scene_indices=scene_indices,
        )
    else:
        written = []

    print(f"Loaded stars: {ra_deg.shape[0]}")
    print(f"Catalog map saved: {catalog_png}")
    print(f"FOV map saved: {footprint_png}")
    if written:
        print(f"Per-scene maps saved: {len(written)} files in {out_dir / 'scene_plots'}")
    print(f"Boresight mode: {args.boresight_mode}")
    print(f"Roll mode: {args.roll_mode}")
    print(f"Footprints: {int(args.num_footprints)}")


if __name__ == "__main__":
    main()
