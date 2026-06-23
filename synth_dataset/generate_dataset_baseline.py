#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


WIDTH, HEIGHT = 1280, 960
FOV_HORIZONTAL_DEG, FOV_VERTICAL_DEG, FOV_DIAGONAL_DEG = 17.2, 13.0, 21.0
DEFAULT_MAGNITUDE_CUTOFF = 8.0
DEFAULT_MAG_PERTURB_MEAN = 0.0
DEFAULT_MAG_PERTURB_SIGMA = 0.0
DEFAULT_SEED = 6353103531848264806

POINT_KEYS = [
    "point_yx",
    "point_star_id",
    "point_is_false_star",
    "point_magnitude",
]

SCENE_KEYS = [
    "scene_point_count",
    "guide_star_index",
    "pre_dropout_real_star_count",
    "scene_dropout_count",
    "scene_false_stars_count",
    "scene_real_star_count",
    "scene_total_point_count",
    "scene_seed",
    "roll_degree",
]

ALL_KEYS = POINT_KEYS + SCENE_KEYS

DTYPE = {
    "point_yx": np.float32,
    "point_star_id": np.int32,
    "point_is_false_star": np.int8,
    "point_magnitude": np.float32,
    "scene_point_count": np.int32,
    "guide_star_index": np.int32,
    "pre_dropout_real_star_count": np.int32,
    "scene_dropout_count": np.int32,
    "scene_false_stars_count": np.int32,
    "scene_real_star_count": np.int32,
    "scene_total_point_count": np.int32,
    "scene_seed": np.int64,
    "roll_degree": np.int16,
}

W: dict[str, object] = {}


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def _seed(base: int, g: int, rep: int, roll: int) -> int:
    return int(
        (base + (g + 1) * 1000003 + (rep + 1) * 10007 + (roll + 1) * 101)
        & ((1 << 63) - 1)
    )


def _parse_guide_indices(raw_indices: str) -> np.ndarray:
    tokens = raw_indices.replace(",", " ").split()
    if not tokens:
        raise ValueError("--guide-indices must contain at least one integer")

    values: list[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid value in --guide-indices: {token!r}") from exc
    return np.asarray(values, dtype=np.int64)


def _choose_explicit_guides(n_stars: int, raw_indices: str, guide_offset: int) -> np.ndarray:
    parsed = _parse_guide_indices(raw_indices)
    shifted = parsed + int(guide_offset)

    invalid = shifted[(shifted < 0) | (shifted >= n_stars)]
    if invalid.size > 0:
        raise ValueError(
            f"Indices out of range [0, {n_stars - 1}] after offset: {invalid.tolist()}"
        )

    unique_ordered: list[int] = []
    seen: set[int] = set()
    for idx in shifted.tolist():
        value = int(idx)
        if value not in seen:
            seen.add(value)
            unique_ordered.append(value)
    return np.asarray(unique_ordered, dtype=np.int32)


def _init_worker(
    v: np.ndarray,
    m: np.ndarray,
    allowed: np.ndarray,
    tx: float,
    ty: float,
    dcos: float,
    mag_cutoff: float,
    mag_perturb_mean: float,
    mag_perturb_sigma: float,
) -> None:
    W.update(
        v=v,
        m=m,
        allowed=allowed,
        tx=float(tx),
        ty=float(ty),
        dcos=float(dcos),
        mag_cutoff=float(mag_cutoff),
        mag_perturb_mean=float(mag_perturb_mean),
        mag_perturb_sigma=float(mag_perturb_sigma),
    )


def _display_random_scene(
    run_dir: Path,
    chunks: list[dict[str, object]],
    seed: int | None = None,
) -> None:
    if not chunks:
        _log("No chunks available to display a scene.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        _log(
            "Could not import matplotlib "
            f"({e}). Install it to use --display-random-scene."
        )
        return

    rng = np.random.default_rng(seed)
    chunk_info = chunks[int(rng.integers(0, len(chunks)))]
    chunk_path = (run_dir / str(chunk_info["file"])).resolve()

    with np.load(chunk_path) as d:
        n_scenes = int(d["scene_point_count"].shape[0])
        if n_scenes == 0:
            _log(f"Chunk has no scenes: {chunk_path.name}")
            return

        scene_idx = int(rng.integers(0, n_scenes))
        start = int(d["scene_point_start"][scene_idx])
        count = int(d["scene_point_count"][scene_idx])
        end = start + count

        points = np.asarray(d["point_yx"][start:end], dtype=np.float32)
        is_false = np.asarray(d["point_is_false_star"][start:end], dtype=bool)
        mags = np.asarray(d["point_magnitude"][start:end], dtype=np.float32)

        guide_idx = int(d["guide_star_index"][scene_idx])
        roll = int(d["roll_degree"][scene_idx])
        dropout = int(d["scene_dropout_count"][scene_idx])
        n_false = int(d["scene_false_stars_count"][scene_idx])

    x = points[:, 1]
    y = points[:, 0]
    sizes = np.clip(9.0 - mags, 2.0, 14.0) ** 1.5 if points.shape[0] else np.array([], dtype=np.float32)
    colors = np.where(is_false, "#ff5a5a", "#ffffff") if points.shape[0] else np.array([], dtype=object)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.9, edgecolors="none")
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.set_aspect("equal", "box")
    ax.set_title(
        "Random Scene | "
        f"chunk={chunk_path.name} scene={scene_idx} guide={guide_idx} "
        f"roll={roll} pts={count} false={n_false} drop={dropout}",
        color="white",
    )
    ax.set_xlabel("x (pixels)", color="white")
    ax.set_ylabel("y (pixels)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#666666")
    plt.tight_layout()
    plt.show()


def _worker(task: tuple[int, int, int]) -> dict[str, np.ndarray]:
    g_idx, rep, base = task

    v, m = W["v"], W["m"]  # type: ignore[assignment]
    allowed = W["allowed"]  # type: ignore[assignment]
    tx, ty, dcos = float(W["tx"]), float(W["ty"]), float(W["dcos"])  # type: ignore[arg-type]
    mag_cutoff = float(W["mag_cutoff"])  # type: ignore[arg-type]
    mag_perturb_mean = float(W["mag_perturb_mean"])  # type: ignore[arg-type]
    mag_perturb_sigma = float(W["mag_perturb_sigma"])  # type: ignore[arg-type]

    g = v[g_idx]
    # Keep only catalog stars that are inside FOV cone and bright enough for Synopsis.
    cand = np.flatnonzero(((v @ g) >= dcos) & (m <= mag_cutoff) & allowed).astype(np.int32)
    cv, cm = v[cand], m[cand]

    up = np.array([0.0, 0.0, 1.0], np.float32)
    if abs(float(np.dot(up, g))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], np.float32)

    j0 = np.cross(up, g)
    n = np.linalg.norm(j0)
    if n < 1e-8:
        j0 = np.cross(np.array([1.0, 0.0, 0.0], np.float32), g)
        n = np.linalg.norm(j0)

    j0 = (j0 / n).astype(np.float32)
    k0 = np.cross(g, j0)
    k0 = (k0 / np.linalg.norm(k0)).astype(np.float32)

    focal = WIDTH / (2.0 * tx)
    pbuf = {k: [] for k in POINT_KEYS}
    sbuf: dict[str, list[int]] = {k: [] for k in SCENE_KEYS}

    for roll in range(360):
        s = _seed(base, g_idx, rep, roll)
        rng = np.random.default_rng(s)

        r = math.radians(float(roll))
        c, si = math.cos(r), math.sin(r)
        j = (c * j0 + si * k0).astype(np.float32)
        k = (-si * j0 + c * k0).astype(np.float32)

        i, jc, kc = cv @ g, cv @ j, cv @ k
        with np.errstate(divide="ignore", invalid="ignore"):
            qj, qk = jc / i, kc / i

        idx = np.flatnonzero((i > 0.0) & (np.abs(qj) <= tx) & (np.abs(qk) <= ty))

        real_ids = cand[idx]
        real_mag = cm[idx].astype(np.float32)
        real = np.column_stack(
            (
                HEIGHT / 2.0 - qk[idx] * focal,
                WIDTH / 2.0 - qj[idx] * focal,
            )
        ).astype(np.float32)

        pre_n = int(real.shape[0])
        lim = int(math.floor(0.8 * int(real.shape[0])))

        n_false = min(int(rng.integers(0, 6)), lim)
        if n_false:
            fake = np.column_stack(
                (
                    rng.uniform(0.0, float(HEIGHT), n_false),
                    rng.uniform(0.0, float(WIDTH), n_false),
                )
            ).astype(np.float32)
            if pre_n:
                lo, hi = float(np.min(real_mag)), float(np.max(real_mag))
                fake_mag = rng.uniform(lo, hi if lo != hi else lo + 0.2, n_false).astype(np.float32)
            else:
                fake_mag = rng.uniform(0.0, mag_cutoff, n_false).astype(np.float32)
        else:
            fake = np.empty((0, 2), np.float32)
            fake_mag = np.empty((0,), np.float32)

        n_drop = min(int(rng.integers(0, 6)), lim, pre_n)
        if n_drop:
            keep = np.ones(pre_n, bool)
            keep[rng.choice(pre_n, size=n_drop, replace=False)] = False
            real, real_ids, real_mag = real[keep], real_ids[keep], real_mag[keep]

        p_yx = np.concatenate((real, fake), axis=0)
        p_id = np.concatenate((real_ids.astype(np.int32), np.full(n_false, -1, np.int32)), axis=0)
        p_false = np.concatenate((np.zeros(real.shape[0], np.int8), -np.ones(n_false, np.int8)), axis=0)
        p_mag = np.concatenate((real_mag, fake_mag), axis=0).astype(np.float32)

        if p_yx.shape[0]:
            amp = rng.uniform(0.25, 1.0, p_yx.shape[0]).astype(np.float32)
            p_yx += rng.normal(0.0, 1.0, p_yx.shape).astype(np.float32) * amp[:, None]

            np.clip(p_yx[:, 0], 0.0, float(HEIGHT) - 1e-3, out=p_yx[:, 0])
            np.clip(p_yx[:, 1], 0.0, float(WIDTH) - 1e-3, out=p_yx[:, 1])

            if float(mag_perturb_sigma) == 0.0:
                p_mag += np.float32(mag_perturb_mean)
            else:
                dmag = rng.normal(mag_perturb_mean, mag_perturb_sigma, p_mag.shape[0]).astype(np.float32)
                p_mag += dmag

            order = np.argsort(p_mag, kind="stable")
            p_yx, p_id, p_false, p_mag = p_yx[order], p_id[order], p_false[order], p_mag[order]

        pbuf["point_yx"].append(p_yx)
        pbuf["point_star_id"].append(p_id)
        pbuf["point_is_false_star"].append(p_false.astype(np.int8, copy=False))
        pbuf["point_magnitude"].append(p_mag)

        sbuf["scene_point_count"].append(int(p_yx.shape[0]))
        sbuf["guide_star_index"].append(int(g_idx))
        sbuf["pre_dropout_real_star_count"].append(pre_n)
        sbuf["scene_dropout_count"].append(n_drop)
        sbuf["scene_false_stars_count"].append(n_false)
        sbuf["scene_real_star_count"].append(int(np.sum(p_false == 0)))
        sbuf["scene_total_point_count"].append(int(p_yx.shape[0]))
        sbuf["scene_seed"].append(s)
        sbuf["roll_degree"].append(roll)

    out: dict[str, np.ndarray] = {}
    for k in POINT_KEYS:
        out[k] = np.concatenate(pbuf[k], axis=0).astype(DTYPE[k], copy=False)
    for k in SCENE_KEYS:
        out[k] = np.asarray(sbuf[k], dtype=DTYPE[k])
    return out


def _vector_to_radec_deg(v: np.ndarray) -> tuple[float, float]:
    x, y, z = float(v[0]), float(v[1]), float(np.clip(v[2], -1.0, 1.0))
    ra = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    dec = math.degrees(math.asin(z))
    return ra, dec


def _camera_basis(boresight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _build_fov_polygon_radec(
    *,
    boresight: np.ndarray,
    roll_rad: float,
    fov_h_deg: float,
    fov_v_deg: float,
    edge_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    tx = math.tan(math.radians(float(fov_h_deg)) / 2.0)
    ty = math.tan(math.radians(float(fov_v_deg)) / 2.0)

    j0, k0 = _camera_basis(boresight.astype(np.float64, copy=False))
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
    ra = (np.degrees(np.arctan2(poly[:, 1], poly[:, 0])) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(np.clip(poly[:, 2], -1.0, 1.0)))
    return ra, dec


def _build_scene_metadata_from_chunks(
    *,
    run_dir: Path,
    chunks: list[dict[str, object]],
    vectors: np.ndarray,
    coverage_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_stars = int(vectors.shape[0])
    total_scenes = int(sum(int(c.get("scene_count", 0)) for c in chunks))

    appear_count = np.zeros(n_stars, dtype=np.int32)
    freq = np.zeros(total_scenes + 2, dtype=np.int64)
    freq[0] = n_stars

    scope_mask = None
    scope_index_of = None
    scope_appear_count = None
    scope_freq = None
    scope_star_count = int(n_stars)
    scope_min_count = 0
    scope_max_count = 0
    scope_total_appearances = 0
    if coverage_mask is not None:
        scope_mask = np.asarray(coverage_mask, dtype=bool)
        if scope_mask.shape[0] != n_stars:
            raise ValueError("coverage_mask shape mismatch")
        scope_indices = np.flatnonzero(scope_mask).astype(np.int32, copy=False)
        scope_star_count = int(scope_indices.shape[0])
        scope_index_of = np.full(n_stars, -1, dtype=np.int32)
        scope_index_of[scope_indices] = np.arange(scope_star_count, dtype=np.int32)
        scope_appear_count = np.zeros(scope_star_count, dtype=np.int32)
        scope_freq = np.zeros(total_scenes + 2, dtype=np.int64)
        scope_freq[0] = scope_star_count

    boresight_xyz = np.zeros((total_scenes, 3), dtype=np.float32)
    boresight_ra = np.zeros(total_scenes, dtype=np.float32)
    boresight_dec = np.zeros(total_scenes, dtype=np.float32)
    roll_deg = np.zeros(total_scenes, dtype=np.int16)
    guide_star = np.zeros(total_scenes, dtype=np.int32)

    real_start = np.zeros(total_scenes, dtype=np.int64)
    real_count = np.zeros(total_scenes, dtype=np.int32)
    real_star_ids_parts: list[np.ndarray] = []

    cumulative_unique = np.zeros(total_scenes, dtype=np.int32)
    cumulative_total = np.zeros(total_scenes, dtype=np.int64)
    cumulative_min = np.zeros(total_scenes, dtype=np.int32)
    cumulative_mean = np.zeros(total_scenes, dtype=np.float32)
    cumulative_max = np.zeros(total_scenes, dtype=np.int32)

    min_count = 0
    max_count = 0
    total_appearances = 0
    flat_cursor = 0
    global_scene_idx = 0

    for chunk in chunks:
        rel = str(chunk["file"])
        chunk_path = (run_dir / rel).resolve()
        with np.load(chunk_path, allow_pickle=False) as data:
            point_star_id = np.asarray(data["point_star_id"], dtype=np.int32)
            scene_start = np.asarray(data["scene_point_start"], dtype=np.int64)
            scene_count_chunk = np.asarray(data["scene_point_count"], dtype=np.int64)
            scene_guides = np.asarray(data["guide_star_index"], dtype=np.int32)
            scene_roll = np.asarray(data["roll_degree"], dtype=np.int16)

            for local_scene_idx in range(int(scene_count_chunk.shape[0])):
                start = int(scene_start[local_scene_idx])
                count = int(scene_count_chunk[local_scene_idx])
                end = start + count

                scene_ids = point_star_id[start:end]
                scene_real_ids = np.unique(scene_ids[scene_ids >= 0]).astype(np.int32, copy=False)

                real_start[global_scene_idx] = int(flat_cursor)
                real_count[global_scene_idx] = int(scene_real_ids.shape[0])
                if scene_real_ids.size > 0:
                    real_star_ids_parts.append(scene_real_ids)
                    flat_cursor += int(scene_real_ids.shape[0])

                    old_counts = appear_count[scene_real_ids]
                    for old in old_counts.tolist():
                        freq[int(old)] -= 1
                        freq[int(old) + 1] += 1
                    appear_count[scene_real_ids] = old_counts + 1

                    total_appearances += int(scene_real_ids.shape[0])
                    max_count = max(max_count, int(np.max(old_counts + 1)))

                    if scope_index_of is not None and scope_appear_count is not None and scope_freq is not None:
                        scope_scene_ids = scope_index_of[scene_real_ids]
                        scope_scene_ids = scope_scene_ids[scope_scene_ids >= 0]
                        if scope_scene_ids.size > 0:
                            scoped_old_counts = scope_appear_count[scope_scene_ids]
                            for old in scoped_old_counts.tolist():
                                scope_freq[int(old)] -= 1
                                scope_freq[int(old) + 1] += 1
                            scope_appear_count[scope_scene_ids] = scoped_old_counts + 1
                            scope_total_appearances += int(scope_scene_ids.shape[0])
                            scope_max_count = max(scope_max_count, int(np.max(scoped_old_counts + 1)))

                while min_count < freq.shape[0] and freq[min_count] <= 0:
                    min_count += 1
                if scope_freq is not None:
                    while scope_min_count < scope_freq.shape[0] and scope_freq[scope_min_count] <= 0:
                        scope_min_count += 1

                g = int(scene_guides[local_scene_idx])
                guide_star[global_scene_idx] = g
                roll_deg[global_scene_idx] = int(scene_roll[local_scene_idx])
                bv = vectors[g].astype(np.float32, copy=False)
                boresight_xyz[global_scene_idx] = bv
                ra, dec = _vector_to_radec_deg(bv)
                boresight_ra[global_scene_idx] = float(ra)
                boresight_dec[global_scene_idx] = float(dec)

                if scope_appear_count is not None:
                    cumulative_unique[global_scene_idx] = int(np.sum(scope_appear_count > 0))
                    cumulative_total[global_scene_idx] = int(scope_total_appearances)
                    cumulative_min[global_scene_idx] = int(scope_min_count)
                    cumulative_mean[global_scene_idx] = float(
                        scope_total_appearances / max(scope_star_count, 1)
                    )
                    cumulative_max[global_scene_idx] = int(scope_max_count)
                else:
                    cumulative_unique[global_scene_idx] = int(np.sum(appear_count > 0))
                    cumulative_total[global_scene_idx] = int(total_appearances)
                    cumulative_min[global_scene_idx] = int(min_count)
                    cumulative_mean[global_scene_idx] = float(total_appearances / max(n_stars, 1))
                    cumulative_max[global_scene_idx] = int(max_count)

                global_scene_idx += 1

    real_star_ids = (
        np.concatenate(real_star_ids_parts, axis=0).astype(np.int32, copy=False)
        if real_star_ids_parts
        else np.empty((0,), dtype=np.int32)
    )

    scene_meta = {
        "scene_boresight_xyz": boresight_xyz,
        "scene_center_ra_deg": boresight_ra,
        "scene_center_dec_deg": boresight_dec,
        "scene_roll_degree": roll_deg,
        "scene_guide_star_index": guide_star,
        "scene_real_star_start": real_start,
        "scene_real_star_count": real_count,
        "scene_real_star_id": real_star_ids,
        "cumulative_unique_seen_stars": cumulative_unique,
        "cumulative_total_appearances": cumulative_total,
        "cumulative_min_appear_count": cumulative_min,
        "cumulative_mean_appear_count": cumulative_mean,
        "cumulative_max_appear_count": cumulative_max,
        "final_appear_count": appear_count,
    }

    scoped = appear_count
    if scope_mask is not None:
        scoped = appear_count[scope_mask]

    coverage = {
        "stars": int(n_stars),
        "coverage_scope_stars": int(scope_star_count),
        "scene_count": int(total_scenes),
        "appear_count_min": int(np.min(scoped)) if scoped.size else 0,
        "appear_count_mean": float(np.mean(scoped)) if scoped.size else 0.0,
        "appear_count_max": int(np.max(scoped)) if scoped.size else 0,
        "unique_seen_stars": int(np.sum(scoped > 0)),
        "total_appearances": int(np.sum(scoped)),
    }
    return scene_meta, coverage


def _ra_in_window(ra_deg: float, ra_min: float, ra_max: float) -> bool:
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


def _dec_in_window(dec_deg: float, dec_min: float, dec_max: float) -> bool:
    lo = max(-90.0, min(float(dec_min), float(dec_max)))
    hi = min(90.0, max(float(dec_min), float(dec_max)))
    if (hi - lo) >= 180.0:
        return True
    dec = float(dec_deg)
    return bool(lo <= dec <= hi)


def _guide_has_full_fov_inside_window(
    *,
    boresight: np.ndarray,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    edge_samples: int,
) -> bool:
    for roll_deg in range(360):
        poly_ra, poly_dec = _build_fov_polygon_radec(
            boresight=boresight,
            roll_rad=math.radians(float(roll_deg)),
            fov_h_deg=FOV_HORIZONTAL_DEG,
            fov_v_deg=FOV_VERTICAL_DEG,
            edge_samples=edge_samples,
        )
        for ra, dec in zip(poly_ra.tolist(), poly_dec.tolist()):
            if not _ra_in_window(float(ra), ra_min, ra_max):
                return False
            if not _dec_in_window(float(dec), dec_min, dec_max):
                return False
    return True


def _select_full_fov_safe_guides(
    *,
    vectors: np.ndarray,
    candidate_indices: np.ndarray,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    edge_samples: int,
) -> np.ndarray:
    safe: list[int] = []
    for idx in candidate_indices.tolist():
        if _guide_has_full_fov_inside_window(
            boresight=vectors[int(idx)],
            ra_min=ra_min,
            ra_max=ra_max,
            dec_min=dec_min,
            dec_max=dec_max,
            edge_samples=edge_samples,
        ):
            safe.append(int(idx))
    return np.asarray(safe, dtype=np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic star dataset from Tetra star_table")
    p.add_argument("--guide-stars", type=int, default=None)
    p.add_argument(
        "--guide-indices",
        type=str,
        default=None,
        help="Explicit guide indices list (for example: '10,25,40').",
    )
    p.add_argument(
        "--guide-offset",
        type=int,
        default=0,
        help="Integer offset applied to each --guide-indices value.",
    )
    p.add_argument("--num-repeats", type=int, required=True)
    p.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tetra3" / "data" / "default_database.npz",
    )
    p.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--chunk-size-mb", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=(os.cpu_count() or 1))
    p.add_argument(
        "--magnitude-cutoff",
        type=float,
        default=DEFAULT_MAGNITUDE_CUTOFF,
        help="Include only stars with catalog magnitude <= this value (default: 8.0).",
    )
    p.add_argument(
        "--magnitude-perturb-mean",
        type=float,
        default=DEFAULT_MAG_PERTURB_MEAN,
        help=(
            "Mean of the normal perturbation added to synthetic point magnitudes. "
            "Default 0.0 keeps catalog magnitudes unchanged."
        ),
    )
    p.add_argument(
        "--magnitude-perturb-sigma",
        type=float,
        default=DEFAULT_MAG_PERTURB_SIGMA,
        help="Standard deviation of the normal perturbation added to synthetic point magnitudes.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for guide sampling and scene perturbations. Default: {DEFAULT_SEED}.",
    )
    p.add_argument(
        "--display-random-scene",
        action="store_true",
        help="Display one random generated scene (requires matplotlib)",
    )
    p.add_argument(
        "--instrument-coverage",
        action="store_true",
        help="Compute appear_count instrumentation and write scene metadata files for comparison.",
    )
    p.add_argument(
        "--timelapse",
        action="store_true",
        help=(
            "Timelapse validation mode: constrain generation to stars inside the RA window, "
            "and generate sky_plots frames (implies --instrument-coverage)."
        ),
    )
    p.add_argument(
        "--timelapse-ra-min",
        type=float,
        default=150.0,
        help="Minimum RA (deg) for timelapse generation and frame export.",
    )
    p.add_argument(
        "--timelapse-ra-max",
        type=float,
        default=180.0,
        help="Maximum RA (deg) for timelapse generation and frame export.",
    )
    p.add_argument(
        "--timelapse-dec-min",
        type=float,
        default=-90.0,
        help="Minimum declination (deg) for timelapse generation and frame export.",
    )
    p.add_argument(
        "--timelapse-dec-max",
        type=float,
        default=90.0,
        help="Maximum declination (deg) for timelapse generation and frame export.",
    )
    p.add_argument("--timelapse-plot-ra-min", type=float, default=None, help="RA axis minimum for timelapse plots.")
    p.add_argument("--timelapse-plot-ra-max", type=float, default=None, help="RA axis maximum for timelapse plots.")
    p.add_argument("--timelapse-plot-dec-min", type=float, default=None, help="Declination axis minimum for timelapse plots.")
    p.add_argument("--timelapse-plot-dec-max", type=float, default=None, help="Declination axis maximum for timelapse plots.")
    p.add_argument(
        "--every-nth-scene",
        type=int,
        default=1,
        help="Export one timelapse frame every N scenes.",
    )
    p.add_argument(
        "--timelapse-require-full-fov-inside",
        action="store_true",
        help=(
            "In timelapse mode, only allow guide stars whose full FOV footprint stays inside "
            "the analysis RA/Dec window for all 360 roll angles."
        ),
    )
    p.add_argument(
        "--timelapse-fov-edge-samples",
        type=int,
        default=24,
        help="Samples per FOV edge when validating full-footprint containment in timelapse mode.",
    )
    a = p.parse_args()

    if a.num_repeats <= 0 or a.chunk_size_mb <= 0 or a.num_workers <= 0:
        raise ValueError("Invalid args: num_repeats/chunk_size_mb/num_workers must be >0")
    if a.guide_indices is None and a.guide_stars is None:
        raise ValueError("Set --guide-stars or --guide-indices")
    if a.guide_stars is not None and int(a.guide_stars) < 0:
        raise ValueError("--guide-stars must be >= 0")
    if int(a.every_nth_scene) <= 0:
        raise ValueError("--every-nth-scene must be > 0")
    if int(a.timelapse_fov_edge_samples) < 4:
        raise ValueError("--timelapse-fov-edge-samples must be >= 4")
    if float(a.magnitude_perturb_sigma) < 0.0:
        raise ValueError("--magnitude-perturb-sigma must be >= 0")

    instrument_coverage = bool(a.instrument_coverage or a.timelapse)

    seed = int(a.seed) & ((1 << 63) - 1)

    db_path = a.database.expanduser().resolve()
    with np.load(db_path) as d:
        if "star_table" not in d:
            raise RuntimeError("Database must contain star_table")
        star_table = np.asarray(d["star_table"], dtype=np.float32)
        star_catalog_ids = np.asarray(d["star_catalog_IDs"]) if "star_catalog_IDs" in d else None

    if star_table.ndim != 2 or star_table.shape[1] < 6:
        raise RuntimeError("Invalid star_table format")

    v = star_table[:, 2:5].astype(np.float32)
    mags = star_table[:, 5].astype(np.float32)

    n_stars = int(v.shape[0])
    star_ra = (np.degrees(np.arctan2(v[:, 1].astype(np.float64), v[:, 0].astype(np.float64))) + 360.0) % 360.0
    star_dec = np.degrees(np.arcsin(np.clip(v[:, 2].astype(np.float64), -1.0, 1.0)))
    guide_candidate_mask = None
    if a.timelapse:
        analysis_star_mask = np.array(
            [
                _ra_in_window(float(ra), float(a.timelapse_ra_min), float(a.timelapse_ra_max))
                and _dec_in_window(float(dec), float(a.timelapse_dec_min), float(a.timelapse_dec_max))
                for ra, dec in zip(star_ra.tolist(), star_dec.tolist())
            ],
            dtype=bool,
        )
        allowed_star_mask = analysis_star_mask
        allowed_star_indices = np.flatnonzero(allowed_star_mask).astype(np.int32)
        if allowed_star_indices.size == 0:
            raise ValueError(
                "Timelapse RA/Dec window selected zero stars. "
                "Adjust --timelapse-ra-min/--timelapse-ra-max/--timelapse-dec-min/--timelapse-dec-max."
            )
        if a.timelapse_require_full_fov_inside:
            _log("Filtering timelapse guide candidates to keep the full FOV inside the analysis area...")
            guide_candidate_indices = _select_full_fov_safe_guides(
                vectors=v.astype(np.float64, copy=False),
                candidate_indices=allowed_star_indices,
                ra_min=float(a.timelapse_ra_min),
                ra_max=float(a.timelapse_ra_max),
                dec_min=float(a.timelapse_dec_min),
                dec_max=float(a.timelapse_dec_max),
                edge_samples=int(a.timelapse_fov_edge_samples),
            )
            guide_candidate_mask = np.zeros(n_stars, dtype=bool)
            guide_candidate_mask[guide_candidate_indices.astype(np.int64)] = True
            if guide_candidate_indices.size == 0:
                raise ValueError(
                    "No guide stars remain after requiring the full FOV inside the timelapse window. "
                    "Expand the RA/Dec window or disable --timelapse-require-full-fov-inside."
                )
        else:
            guide_candidate_mask = allowed_star_mask.copy()
            guide_candidate_indices = allowed_star_indices
    else:
        allowed_star_mask = np.ones(n_stars, dtype=bool)
        allowed_star_indices = np.arange(n_stars, dtype=np.int32)
        guide_candidate_mask = allowed_star_mask
        guide_candidate_indices = allowed_star_indices

    rng = np.random.default_rng(seed)
    if a.guide_indices is not None:
        explicit_guides = _choose_explicit_guides(
            n_stars=n_stars,
            raw_indices=str(a.guide_indices),
            guide_offset=int(a.guide_offset),
        )
        if a.timelapse:
            outside = explicit_guides[~guide_candidate_mask[explicit_guides.astype(np.int64)]]
            if outside.size > 0:
                if a.timelapse_require_full_fov_inside:
                    raise ValueError(
                        "Timelapse mode only accepts guide stars whose full FOV fits inside the RA/Dec window; "
                        f"invalid indices: {outside.tolist()}"
                    )
                raise ValueError(
                    "Timelapse mode only accepts guide stars inside the RA window; "
                    f"outside indices: {outside.tolist()}"
                )
        extra_random_count = int(a.guide_stars) if a.guide_stars is not None else 0
        if extra_random_count <= 0:
            guide = explicit_guides
        else:
            free_mask = guide_candidate_mask.copy()
            free_mask[explicit_guides.astype(np.int64)] = False
            candidates = np.flatnonzero(free_mask).astype(np.int32)
            if extra_random_count >= int(candidates.shape[0]):
                random_guides = candidates
            else:
                random_guides = np.sort(
                    rng.choice(candidates, size=extra_random_count, replace=False).astype(np.int32)
                )
            guide = np.concatenate((explicit_guides.astype(np.int32), random_guides.astype(np.int32)), axis=0)
    elif a.guide_stars <= 0 or a.guide_stars >= int(guide_candidate_indices.shape[0]):
        guide = guide_candidate_indices
    else:
        guide = np.sort(rng.choice(guide_candidate_indices, size=a.guide_stars, replace=False).astype(np.int32))

    if guide.size == 0:
        raise ValueError("No guide stars selected. Check timelapse RA window and guide arguments.")

    runs_root = a.runs_root.expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    run_idx = 1 + max(
        [
            int(x.name[3:])
            for x in runs_root.iterdir()
            if x.is_dir() and x.name.startswith("run") and x.name[3:].isdigit()
        ],
        default=0,
    )

    run_dir = runs_root / f"run{run_idx}"
    dataset_dir = runs_root / f"run{run_idx}" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=False)

    start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _log(f"Run folder created: {run_dir}")
    _log(f"Resolution fixed: {WIDTH}x{HEIGHT}")
    if a.timelapse:
        _log(
            "Timelapse mode: constraining generation to RA/Dec window "
            f"RA[{float(a.timelapse_ra_min):.3f}, {float(a.timelapse_ra_max):.3f}] "
            f"Dec[{float(a.timelapse_dec_min):.3f}, {float(a.timelapse_dec_max):.3f}] "
            f"with {int(allowed_star_indices.shape[0])} eligible catalog stars"
        )
        if a.timelapse_require_full_fov_inside:
            _log(
                "Timelapse guide selection: requiring full FOV containment for all rolls, "
                f"leaving {int(guide_candidate_indices.shape[0])} valid guide centers"
            )
    if a.guide_indices is not None:
        random_extra = int(a.guide_stars) if a.guide_stars is not None else 0
        _log(
            "Guide selection mode: explicit indices "
            f"+ {max(random_extra, 0)} random guides, offset {int(a.guide_offset)}"
        )

    fx, fy, fd = map(math.radians, (FOV_HORIZONTAL_DEG, FOV_VERTICAL_DEG, FOV_DIAGONAL_DEG))
    tx, ty, dcos = math.tan(fx / 2.0), math.tan(fy / 2.0), math.cos(fd / 2.0)
    _log(
        f"FOV fixed: diag={FOV_DIAGONAL_DEG:.6f} deg, "
        f"h={FOV_HORIZONTAL_DEG:.6f} deg, v={FOV_VERTICAL_DEG:.6f} deg"
    )
    _log(
        "Catalog base: stars from Tetra catalog inside FOV only, "
        f"magnitude <= {float(a.magnitude_cutoff):.2f}, "
        "with synthetic false/dropout/position perturbations enabled and "
        f"magnitude perturbation N({float(a.magnitude_perturb_mean):.3f}, "
        f"{float(a.magnitude_perturb_sigma):.3f})"
    )

    tasks = [(int(g), int(r), int(seed)) for g in guide.tolist() for r in range(a.num_repeats)]
    expected_scene_count = int(guide.shape[0] * a.num_repeats * 360)

    parts = {k: [] for k in ALL_KEYS}
    bytes_in_buffer = 0
    chunk_idx = 0
    chunk_target = int(a.chunk_size_mb * 1024 * 1024)

    chunks: list[dict[str, object]] = []
    totals = {
        "generated_scene_count": 0,
        "total_points": 0,
        "total_real_points": 0,
        "total_false_points": 0,
        "total_dropout_count": 0,
    }

    def flush() -> None:
        nonlocal bytes_in_buffer, chunk_idx
        if not parts["scene_point_count"]:
            return

        data = {k: np.concatenate(parts[k], axis=0).astype(DTYPE[k], copy=False) for k in ALL_KEYS}
        c = data["scene_point_count"].astype(np.int64)

        if c.shape[0] > 1:
            data["scene_point_start"] = np.concatenate(([0], np.cumsum(c[:-1], dtype=np.int64)))
        else:
            data["scene_point_start"] = np.zeros(c.shape[0], np.int64)

        chunk_idx += 1
        out = dataset_dir / f"dataset{chunk_idx}.npz"
        np.savez_compressed(out, **data)
        sz = int(out.stat().st_size)

        chunks.append(
            {
                "chunk_index": chunk_idx,
                "file": str(out.relative_to(run_dir)),
                "scene_count": int(c.shape[0]),
                "point_count": int(data["point_yx"].shape[0]),
                "size_bytes": sz,
            }
        )

        for k in ALL_KEYS:
            parts[k].clear()

        bytes_in_buffer = 0
        _log(f"Chunk written: {out.name} ({sz} bytes)")

    def consume_result(res: dict[str, np.ndarray]) -> None:
        nonlocal bytes_in_buffer
        for k in ALL_KEYS:
            parts[k].append(res[k])
            bytes_in_buffer += int(res[k].nbytes)

        totals["generated_scene_count"] += int(res["scene_total_point_count"].shape[0])
        totals["total_points"] += int(res["point_yx"].shape[0])
        totals["total_real_points"] += int(np.sum(res["scene_real_star_count"]))
        totals["total_false_points"] += int(np.sum(res["scene_false_stars_count"]))
        totals["total_dropout_count"] += int(np.sum(res["scene_dropout_count"]))

        if bytes_in_buffer >= chunk_target:
            flush()

    initargs = (
        v,
        mags,
        allowed_star_mask.astype(bool, copy=False),
        tx,
        ty,
        dcos,
        float(a.magnitude_cutoff),
        float(a.magnitude_perturb_mean),
        float(a.magnitude_perturb_sigma),
    )
    if int(a.num_workers) <= 1:
        _init_worker(*initargs)
        for task in tasks:
            consume_result(_worker(task))
    else:
        with ProcessPoolExecutor(
            max_workers=a.num_workers,
            initializer=_init_worker,
            initargs=initargs,
        ) as pool:
            for res in pool.map(_worker, tasks, chunksize=1):
                consume_result(res)

    flush()
    end_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    scene_meta: dict[str, np.ndarray] | None = None
    coverage_stats: dict[str, Any] | None = None
    if instrument_coverage:
        _log("Computing coverage instrumentation from generated chunks...")
        scene_meta, coverage_stats = _build_scene_metadata_from_chunks(
            run_dir=run_dir,
            chunks=chunks,
            vectors=v.astype(np.float64),
            coverage_mask=allowed_star_mask if a.timelapse else None,
        )
        np.savez_compressed(run_dir / "scene_metadata.npz", **scene_meta)
        np.savez_compressed(
            run_dir / "coverage_stats.npz",
            final_appear_count=np.asarray(scene_meta["final_appear_count"], dtype=np.int32),
        )
        (run_dir / "coverage_summary.json").write_text(
            json.dumps(coverage_stats, indent=2) + "\n",
            encoding="utf-8",
        )

    counts = {
        "expected_scene_count": int(expected_scene_count),
        "generated_scene_count": int(totals["generated_scene_count"]),
        "chunks_count": int(len(chunks)),
        "total_points": int(totals["total_points"]),
        "total_real_points": int(totals["total_real_points"]),
        "total_false_points": int(totals["total_false_points"]),
        "total_dropout_count": int(totals["total_dropout_count"]),
    }
    if coverage_stats is not None:
        counts.update(
            {
                "coverage_scope_stars": int(coverage_stats["coverage_scope_stars"]),
                "appear_count_min": int(coverage_stats["appear_count_min"]),
                "appear_count_mean": float(coverage_stats["appear_count_mean"]),
                "appear_count_max": int(coverage_stats["appear_count_max"]),
                "unique_seen_stars": int(coverage_stats["unique_seen_stars"]),
                "total_appearances": int(coverage_stats["total_appearances"]),
            }
        )

    params = {
        "guide_stars": int(guide.shape[0]),
        "num_repeats": int(a.num_repeats),
        "fov_diagonal_deg": float(FOV_DIAGONAL_DEG),
        "fov_horizontal_deg": float(FOV_HORIZONTAL_DEG),
        "fov_vertical_deg": float(FOV_VERTICAL_DEG),
        "magnitude_cutoff": float(a.magnitude_cutoff),
        "magnitude_perturb_mean": float(a.magnitude_perturb_mean),
        "magnitude_perturb_sigma": float(a.magnitude_perturb_sigma),
        "resolution": [WIDTH, HEIGHT],
        "chunk_size_mb": int(a.chunk_size_mb),
        "num_workers": int(a.num_workers),
        "seed": int(seed),
        "instrument_coverage": bool(instrument_coverage),
        "timelapse": bool(a.timelapse),
        "timelapse_ra_window_deg": [float(a.timelapse_ra_min), float(a.timelapse_ra_max)],
        "timelapse_dec_window_deg": [float(a.timelapse_dec_min), float(a.timelapse_dec_max)],
        "timelapse_require_full_fov_inside": bool(a.timelapse_require_full_fov_inside),
        "timelapse_fov_edge_samples": int(a.timelapse_fov_edge_samples),
        "timelapse_plot_ra_window_deg": [
            float(a.timelapse_plot_ra_min) if a.timelapse_plot_ra_min is not None else None,
            float(a.timelapse_plot_ra_max) if a.timelapse_plot_ra_max is not None else None,
        ],
        "timelapse_plot_dec_window_deg": [
            float(a.timelapse_plot_dec_min) if a.timelapse_plot_dec_min is not None else None,
            float(a.timelapse_plot_dec_max) if a.timelapse_plot_dec_max is not None else None,
        ],
        "eligible_stars_in_window": int(allowed_star_indices.shape[0]),
        "eligible_stars_in_ra_window": int(allowed_star_indices.shape[0]),
        "eligible_guides_full_fov_inside_window": int(guide_candidate_indices.shape[0]),
        "every_nth_scene": int(a.every_nth_scene),
    }

    manifest = {
        "run": {
            "name": f"run{run_idx}",
            "run_index": int(run_idx),
            "created_at_utc": start_utc,
            "finished_at_utc": end_utc,
            "database_path": str(db_path),
            "database_star_count": n_stars,
            "parameters": params,
        },
        "guide_star_indices": guide.tolist(),
        "guide_star_catalog_ids": (
            star_catalog_ids[guide].tolist() if star_catalog_ids is not None else None
        ),
        "counts": counts,
        "chunks": chunks,
    }
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        dur = (datetime.fromisoformat(end_utc) - datetime.fromisoformat(start_utc)).total_seconds()
    except Exception:
        dur = 0.0

    sp = {
        "guide_stars": int(guide.shape[0]),
        "num_repeats": int(a.num_repeats),
        "fov_diagonal_degrees": float(FOV_DIAGONAL_DEG),
        "fov_horizontal_degrees": float(FOV_HORIZONTAL_DEG),
        "fov_vertical_degrees": float(FOV_VERTICAL_DEG),
        "magnitude_cutoff": float(a.magnitude_cutoff),
        "magnitude_perturb_mean": float(a.magnitude_perturb_mean),
        "magnitude_perturb_sigma": float(a.magnitude_perturb_sigma),
        "resolution": f"{WIDTH}x{HEIGHT}",
        "chunk_size_mb": int(a.chunk_size_mb),
        "num_workers": int(a.num_workers),
        "seed": int(seed),
    }

    sc = {
        "total_stars_in_database": n_stars,
        "selected_guide_stars": int(guide.shape[0]),
        **counts,
    }

    lines = [
        f"run: run{run_idx}",
        f"started_at_utc: {start_utc}",
        f"ended_at_utc: {end_utc}",
        f"duration_seconds: {dur:.2f}",
        "",
        "parameters:",
    ]
    lines += [f"  {k}: {v}" for k, v in sp.items()]
    lines += ["", "counts:"]
    lines += [f"  {k}: {v}" for k, v in sc.items()]

    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _log("Run finished successfully")
    print(f"Run complete: run{run_idx}")
    print(f"Run directory: {run_dir}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Scenes generated: {totals['generated_scene_count']}")
    print(f"Chunks written: {len(chunks)}")
    if coverage_stats is not None:
        print(
            "Appear count stats: "
            f"min={int(coverage_stats['appear_count_min'])} "
            f"mean={float(coverage_stats['appear_count_mean']):.3f} "
            f"max={int(coverage_stats['appear_count_max'])}"
        )

    if a.timelapse:
        try:
            from visualize_sky_fov import generate_run_sky_plots  # type: ignore

            exported = generate_run_sky_plots(
                run_dir=run_dir,
                database_path=db_path,
                out_dir=run_dir / "sky_plots",
                ra_min=float(a.timelapse_ra_min),
                ra_max=float(a.timelapse_ra_max),
                dec_min=float(a.timelapse_dec_min),
                dec_max=float(a.timelapse_dec_max),
                plot_ra_min=(float(a.timelapse_plot_ra_min) if a.timelapse_plot_ra_min is not None else None),
                plot_ra_max=(float(a.timelapse_plot_ra_max) if a.timelapse_plot_ra_max is not None else None),
                plot_dec_min=(float(a.timelapse_plot_dec_min) if a.timelapse_plot_dec_min is not None else None),
                plot_dec_max=(float(a.timelapse_plot_dec_max) if a.timelapse_plot_dec_max is not None else None),
                focus_ra_min=float(a.timelapse_ra_min),
                focus_ra_max=float(a.timelapse_ra_max),
                focus_dec_min=float(a.timelapse_dec_min),
                focus_dec_max=float(a.timelapse_dec_max),
                every_nth_scene=int(a.every_nth_scene),
            )
            _log(f"Timelapse frames generated: {exported}")
        except Exception as exc:
            _log(f"Timelapse generation skipped due to error: {exc}")

    if a.display_random_scene:
        _display_random_scene(run_dir=run_dir, chunks=chunks, seed=seed)


if __name__ == "__main__":
    main()
