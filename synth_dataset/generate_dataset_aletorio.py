#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


WIDTH, HEIGHT = 1280, 960
FOV_HORIZONTAL_DEG, FOV_VERTICAL_DEG, FOV_DIAGONAL_DEG = 17.2, 13.0, 21.0
DEFAULT_MAGNITUDE_CUTOFF = 8.0
DEFAULT_MAG_PERTURB_MEAN = 0.0
DEFAULT_MAG_PERTURB_SIGMA = 0.0
DEFAULT_SEED = 577215227560855758
BAND_TOPUP_PROBABILITY = 0.75
BAND_TOPUP_MAX_OFFSET_DEG = 8.0
BAND_TOPUP_MAX_TRIES = 16

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


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def _vector_to_radec_deg(v: np.ndarray) -> tuple[float, float]:
    x, y, z = float(v[0]), float(v[1]), float(np.clip(v[2], -1.0, 1.0))
    ra = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    dec = math.degrees(math.asin(z))
    return ra, dec


def _camera_basis(boresight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(up, boresight))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    j0 = np.cross(up, boresight)
    n = np.linalg.norm(j0)
    if n < 1e-8:
        j0 = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float32), boresight)
        n = np.linalg.norm(j0)

    j0 = (j0 / n).astype(np.float32)
    k0 = np.cross(boresight, j0)
    k0 = (k0 / np.linalg.norm(k0)).astype(np.float32)
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

    j0, k0 = _camera_basis(boresight.astype(np.float32, copy=False))
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


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(3,)).astype(np.float64)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _random_unit_vector_in_window(
    rng: np.random.Generator,
    *,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> np.ndarray:
    lo_ra = min(float(ra_min), float(ra_max))
    hi_ra = max(float(ra_min), float(ra_max))
    span_abs = hi_ra - lo_ra
    if span_abs >= 360.0:
        ra_deg = float(rng.uniform(0.0, 360.0))
    else:
        if abs(span_abs) < 1e-9:
            ra_deg = float(lo_ra) % 360.0
        else:
            ra_deg = float(rng.uniform(lo_ra, hi_ra)) % 360.0

    lo_dec = max(-90.0, min(float(dec_min), float(dec_max)))
    hi_dec = min(90.0, max(float(dec_min), float(dec_max)))
    if (hi_dec - lo_dec) >= 180.0:
        z = float(rng.uniform(-1.0, 1.0))
    else:
        z0 = math.sin(math.radians(lo_dec))
        z1 = math.sin(math.radians(hi_dec))
        z = float(rng.uniform(min(z0, z1), max(z0, z1)))

    ra = math.radians(ra_deg)
    rho = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array([rho * math.cos(ra), rho * math.sin(ra), z], dtype=np.float32)


def _random_unit_vector_near_target(
    rng: np.random.Generator,
    target: np.ndarray,
    *,
    max_offset_deg: float,
) -> np.ndarray:
    target_n = target.astype(np.float64, copy=False)
    target_n = target_n / max(float(np.linalg.norm(target_n)), 1e-12)
    j0, k0 = _camera_basis(target_n.astype(np.float32, copy=False))

    offset_rad = math.radians(float(rng.uniform(0.0, max(0.0, float(max_offset_deg)))))
    az = float(rng.uniform(0.0, 2.0 * math.pi))
    tangent = math.cos(az) * j0.astype(np.float64, copy=False) + math.sin(az) * k0.astype(np.float64, copy=False)
    v = math.cos(offset_rad) * target_n + math.sin(offset_rad) * tangent
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return v.astype(np.float32, copy=False)


def _sample_targeted_boresight(
    rng: np.random.Generator,
    target: np.ndarray,
    *,
    max_offset_deg: float,
    timelapse: bool,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    require_full_fov_inside: bool,
    fov_edge_samples: int,
) -> np.ndarray | None:
    for _ in range(BAND_TOPUP_MAX_TRIES):
        boresight = _random_unit_vector_near_target(rng, target, max_offset_deg=max_offset_deg)
        if not timelapse:
            return boresight
        ra, dec = _vector_to_radec_deg(boresight.astype(np.float64, copy=False))
        if _ra_in_window(float(ra), float(ra_min), float(ra_max)) and _dec_in_window(float(dec), float(dec_min), float(dec_max)):
            return boresight
    return None


def _load_baseline_reference(run_dir: Path) -> tuple[int | None, float | None]:
    run = run_dir.expanduser().resolve()
    if not run.exists():
        raise FileNotFoundError(f"Baseline run not found: {run}")

    scene_count: int | None = None
    appear_mean: float | None = None

    coverage_json = run / "coverage_summary.json"
    if coverage_json.exists():
        try:
            data = json.loads(coverage_json.read_text(encoding="utf-8"))
            scene_count = int(data.get("scene_count")) if data.get("scene_count") is not None else scene_count
            appear_mean = float(data.get("appear_count_mean")) if data.get("appear_count_mean") is not None else appear_mean
        except Exception:
            pass

    manifest_path = run / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            counts = manifest.get("counts", {})
            if scene_count is None and isinstance(counts, dict) and counts.get("generated_scene_count") is not None:
                scene_count = int(counts["generated_scene_count"])
            if appear_mean is None and isinstance(counts, dict) and counts.get("appear_count_mean") is not None:
                appear_mean = float(counts["appear_count_mean"])
        except Exception:
            pass

    return scene_count, appear_mean


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


def _pose_has_full_fov_inside_window(
    *,
    boresight: np.ndarray,
    roll_deg: int,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    edge_samples: int,
) -> bool:
    poly_ra, poly_dec = _build_fov_polygon_radec(
        boresight=boresight.astype(np.float64, copy=False),
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


def _generate_scene_from_pose(
    *,
    vectors: np.ndarray,
    mags: np.ndarray,
    allowed_mask: np.ndarray,
    boresight: np.ndarray,
    roll_deg: int,
    seed: int,
    tx: float,
    ty: float,
    dcos: float,
    mag_cutoff: float,
    mag_perturb_mean: float,
    mag_perturb_sigma: float,
) -> dict[str, Any] | None:
    cand = np.flatnonzero(((vectors @ boresight) >= dcos) & (mags <= mag_cutoff) & allowed_mask).astype(np.int32)
    if cand.size == 0:
        return None

    cv = vectors[cand]
    cm = mags[cand]
    j0, k0 = _camera_basis(boresight)

    r = math.radians(float(roll_deg))
    c, si = math.cos(r), math.sin(r)
    j = (c * j0 + si * k0).astype(np.float32)
    k = (-si * j0 + c * k0).astype(np.float32)

    i = cv @ boresight
    jc = cv @ j
    kc = cv @ k
    with np.errstate(divide="ignore", invalid="ignore"):
        qj = jc / i
        qk = kc / i

    idx = np.flatnonzero((i > 0.0) & (np.abs(qj) <= tx) & (np.abs(qk) <= ty))
    if idx.size == 0:
        return None

    real_ids = cand[idx]
    real_mag = cm[idx].astype(np.float32)
    focal = WIDTH / (2.0 * tx)
    real = np.column_stack(
        (
            HEIGHT / 2.0 - qk[idx] * focal,
            WIDTH / 2.0 - qj[idx] * focal,
        )
    ).astype(np.float32)

    pre_n = int(real.shape[0])
    lim = int(math.floor(0.8 * float(pre_n)))
    rng = np.random.default_rng(int(seed))

    n_false = min(int(rng.integers(0, 6)), lim)
    if n_false > 0:
        fake = np.column_stack(
            (
                rng.uniform(0.0, float(HEIGHT), n_false),
                rng.uniform(0.0, float(WIDTH), n_false),
            )
        ).astype(np.float32)
        lo = float(np.min(real_mag))
        hi = float(np.max(real_mag))
        fake_mag = rng.uniform(lo, hi if lo != hi else lo + 0.2, n_false).astype(np.float32)
    else:
        fake = np.empty((0, 2), np.float32)
        fake_mag = np.empty((0,), np.float32)

    n_drop = min(int(rng.integers(0, 6)), lim, pre_n)
    if n_drop > 0:
        keep = np.ones(pre_n, dtype=bool)
        keep[rng.choice(pre_n, size=n_drop, replace=False)] = False
        real = real[keep]
        real_ids = real_ids[keep]
        real_mag = real_mag[keep]

    if real.shape[0] == 0:
        return None

    # Guide star for compatibility: brightest real star present in final scene.
    guide_idx = int(real_ids[int(np.argmin(real_mag))])

    point_yx = np.concatenate((real, fake), axis=0)
    point_star_id = np.concatenate((real_ids.astype(np.int32), np.full(n_false, -1, np.int32)), axis=0)
    point_is_false = np.concatenate((np.zeros(real.shape[0], np.int8), -np.ones(n_false, np.int8)), axis=0)
    point_mag = np.concatenate((real_mag, fake_mag), axis=0).astype(np.float32)

    amp = rng.uniform(0.25, 1.0, point_yx.shape[0]).astype(np.float32)
    point_yx += rng.normal(0.0, 1.0, point_yx.shape).astype(np.float32) * amp[:, None]
    np.clip(point_yx[:, 0], 0.0, float(HEIGHT) - 1e-3, out=point_yx[:, 0])
    np.clip(point_yx[:, 1], 0.0, float(WIDTH) - 1e-3, out=point_yx[:, 1])

    if float(mag_perturb_sigma) == 0.0:
        point_mag += np.float32(mag_perturb_mean)
    else:
        dmag = rng.normal(float(mag_perturb_mean), float(mag_perturb_sigma), point_mag.shape[0]).astype(np.float32)
        point_mag += dmag

    order = np.argsort(point_mag, kind="stable")
    point_yx = point_yx[order]
    point_star_id = point_star_id[order]
    point_is_false = point_is_false[order]
    point_mag = point_mag[order]

    scene_real_unique = np.unique(point_star_id[point_star_id >= 0]).astype(np.int32, copy=False)
    ra, dec = _vector_to_radec_deg(boresight.astype(np.float64, copy=False))

    return {
        "point_yx": point_yx.astype(np.float32, copy=False),
        "point_star_id": point_star_id.astype(np.int32, copy=False),
        "point_is_false_star": point_is_false.astype(np.int8, copy=False),
        "point_magnitude": point_mag.astype(np.float32, copy=False),
        "scene_point_count": np.asarray([int(point_yx.shape[0])], dtype=np.int32),
        "guide_star_index": np.asarray([guide_idx], dtype=np.int32),
        "pre_dropout_real_star_count": np.asarray([pre_n], dtype=np.int32),
        "scene_dropout_count": np.asarray([n_drop], dtype=np.int32),
        "scene_false_stars_count": np.asarray([n_false], dtype=np.int32),
        "scene_real_star_count": np.asarray([int(np.sum(point_is_false == 0))], dtype=np.int32),
        "scene_total_point_count": np.asarray([int(point_yx.shape[0])], dtype=np.int32),
        "scene_seed": np.asarray([int(seed)], dtype=np.int64),
        "roll_degree": np.asarray([int(roll_deg)], dtype=np.int16),
        "scene_real_unique_star_ids": scene_real_unique,
        "scene_boresight_xyz": boresight.astype(np.float32, copy=False),
        "scene_center_ra_deg": float(ra),
        "scene_center_dec_deg": float(dec),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Professor approach: synth dataset by random boresight + roll.")
    p.add_argument(
        "--stop-mode",
        type=str,
        choices=("scene_budget", "appear_target", "appear_mean_target", "appear_band_target"),
        default="scene_budget",
        help=(
            "Stop by fixed number of scenes, when every scoped star reaches a minimum "
            "appear target, when the scoped appear_count mean reaches a target, or "
            "when all scoped stars are inside a target band around the baseline mean."
        ),
    )
    p.add_argument("--scene-budget", type=int, default=None, help="Target number of accepted scenes (scene_budget mode).")
    p.add_argument(
        "--max-accepted-scenes",
        type=int,
        default=None,
        help=(
            "Hard cap on accepted scenes for any stop mode. "
            "If reached first, the run stops even if the target condition is not fully met."
        ),
    )
    p.add_argument(
        "--appear-target",
        type=int,
        default=None,
        help="Target appear_count reference for appear_target / appear_mean_target modes.",
    )
    p.add_argument(
        "--appear-band-margin",
        type=float,
        default=500.0,
        help=(
            "Margin used by appear_band_target mode. The run stops when scoped counts satisfy: "
            "min > (target_mean - margin), max < (target_mean + margin), and mean inside the same band."
        ),
    )
    p.add_argument(
        "--appear-cap",
        type=int,
        default=None,
        help="Cap for scene_budget mode. Candidate scene is discarded if any real star already reached this count.",
    )
    p.add_argument(
        "--appear-cap-margin",
        type=float,
        default=0.0,
        help=(
            "If > 0 and baseline mean is available, scene_budget cap becomes "
            "round(baseline_appear_mean + margin)."
        ),
    )
    p.add_argument(
        "--baseline-run",
        type=Path,
        default=None,
        help="Optional baseline run directory to auto-load scene_budget and mean appear_count references.",
    )
    p.add_argument("--max-attempts", type=int, default=None, help="Hard cap on sampled candidate scenes.")
    p.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tetra3" / "data" / "default_database.npz",
    )
    p.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--chunk-size-mb", type=int, default=200)
    p.add_argument(
        "--magnitude-cutoff",
        type=float,
        default=DEFAULT_MAGNITUDE_CUTOFF,
        help="Include only stars with catalog magnitude <= this value.",
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
        help=f"Random seed used for boresights, rolls and perturbations. Default: {DEFAULT_SEED}.",
    )
    p.add_argument(
        "--timelapse",
        action="store_true",
        help=(
            "Timelapse validation mode: constrain boresight/star generation to the RA window "
            "and generate sky_plots frames after run."
        ),
    )
    p.add_argument("--timelapse-ra-min", type=float, default=150.0)
    p.add_argument("--timelapse-ra-max", type=float, default=180.0)
    p.add_argument("--timelapse-dec-min", type=float, default=-90.0)
    p.add_argument("--timelapse-dec-max", type=float, default=90.0)
    p.add_argument("--timelapse-plot-ra-min", type=float, default=None)
    p.add_argument("--timelapse-plot-ra-max", type=float, default=None)
    p.add_argument("--timelapse-plot-dec-min", type=float, default=None)
    p.add_argument("--timelapse-plot-dec-max", type=float, default=None)
    p.add_argument("--every-nth-scene", type=int, default=1)
    p.add_argument(
        "--timelapse-require-full-fov-inside",
        action="store_true",
        help=(
            "In timelapse mode, only accept boresights whose full FOV footprint stays inside "
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

    if int(a.chunk_size_mb) <= 0:
        raise ValueError("--chunk-size-mb must be > 0")
    if int(a.every_nth_scene) <= 0:
        raise ValueError("--every-nth-scene must be > 0")
    if int(a.timelapse_fov_edge_samples) < 4:
        raise ValueError("--timelapse-fov-edge-samples must be >= 4")
    if a.max_accepted_scenes is not None and int(a.max_accepted_scenes) <= 0:
        raise ValueError("--max-accepted-scenes must be > 0 when provided")

    baseline_scene_count: int | None = None
    baseline_appear_mean: float | None = None
    if a.baseline_run is not None:
        baseline_scene_count, baseline_appear_mean = _load_baseline_reference(a.baseline_run)
        _log(
            "Baseline reference loaded: "
            f"scene_count={baseline_scene_count} appear_mean={baseline_appear_mean}"
        )

    stop_mode = str(a.stop_mode)
    scene_budget = int(a.scene_budget) if a.scene_budget is not None else baseline_scene_count
    appear_target = int(a.appear_target) if a.appear_target is not None else (
        int(round(float(baseline_appear_mean))) if baseline_appear_mean is not None else None
    )
    if a.appear_target is not None:
        band_target_mean = float(a.appear_target)
    elif baseline_appear_mean is not None:
        band_target_mean = float(baseline_appear_mean)
    else:
        band_target_mean = None
    band_margin = float(a.appear_band_margin)
    band_lower = (float(band_target_mean) - band_margin) if band_target_mean is not None else None
    band_upper = (float(band_target_mean) + band_margin) if band_target_mean is not None else None
    band_required_min = (int(math.floor(float(band_lower))) + 1) if band_lower is not None else None
    band_allowed_max = (int(math.ceil(float(band_upper))) - 1) if band_upper is not None else None
    if a.appear_cap is not None:
        appear_cap = int(a.appear_cap)
    elif float(a.appear_cap_margin) > 0.0 and baseline_appear_mean is not None:
        appear_cap = int(round(float(baseline_appear_mean) + float(a.appear_cap_margin)))
    else:
        appear_cap = None

    if stop_mode == "scene_budget" and (scene_budget is None or scene_budget <= 0):
        raise ValueError("scene_budget mode requires --scene-budget or a baseline run with generated_scene_count")
    if stop_mode in {"appear_target", "appear_mean_target"} and (appear_target is None or appear_target <= 0):
        raise ValueError(
            f"{stop_mode} mode requires --appear-target or a baseline run with appear_count_mean"
        )
    if stop_mode == "appear_band_target" and band_target_mean is None:
        raise ValueError("appear_band_target mode requires --appear-target or a baseline run with appear_count_mean")
    if a.appear_cap is not None and int(a.appear_cap) <= 0:
        raise ValueError("--appear-cap must be > 0 when provided")
    if float(a.appear_cap_margin) < 0.0:
        raise ValueError("--appear-cap-margin must be >= 0")
    if stop_mode == "appear_band_target" and band_margin <= 0.0:
        raise ValueError("--appear-band-margin must be > 0 for appear_band_target mode")
    if float(a.magnitude_perturb_sigma) < 0.0:
        raise ValueError("--magnitude-perturb-sigma must be >= 0")
    if stop_mode == "appear_band_target" and (
        band_required_min is None
        or band_allowed_max is None
        or band_allowed_max < band_required_min
    ):
        raise ValueError("Invalid band: baseline mean +/- margin produced an empty integer interval")

    if a.seed is None:
        seed = int.from_bytes(os.urandom(8), "little", signed=False) & ((1 << 63) - 1)
    else:
        seed = int(a.seed) & ((1 << 63) - 1)
    rng = np.random.default_rng(seed)

    db_path = a.database.expanduser().resolve()
    with np.load(db_path, allow_pickle=False) as d:
        if "star_table" not in d:
            raise RuntimeError("Database must contain star_table")
        star_table = np.asarray(d["star_table"], dtype=np.float32)
        star_catalog_ids = np.asarray(d["star_catalog_IDs"]) if "star_catalog_IDs" in d else None

    if star_table.ndim != 2 or star_table.shape[1] < 6:
        raise RuntimeError("Invalid star_table format")

    vectors = star_table[:, 2:5].astype(np.float32)
    mags = star_table[:, 5].astype(np.float32)
    n_stars = int(vectors.shape[0])
    star_ra = (np.degrees(np.arctan2(vectors[:, 1].astype(np.float64), vectors[:, 0].astype(np.float64))) + 360.0) % 360.0
    star_dec = np.degrees(np.arcsin(np.clip(vectors[:, 2].astype(np.float64), -1.0, 1.0)))
    if a.timelapse:
        allowed_star_mask = np.array(
            [
                _ra_in_window(float(ra), float(a.timelapse_ra_min), float(a.timelapse_ra_max))
                and _dec_in_window(float(dec), float(a.timelapse_dec_min), float(a.timelapse_dec_max))
                for ra, dec in zip(star_ra.tolist(), star_dec.tolist())
            ],
            dtype=bool,
        )
    else:
        allowed_star_mask = np.ones(n_stars, dtype=bool)
    allowed_star_indices = np.flatnonzero(allowed_star_mask).astype(np.int32)
    scope_star_count = int(allowed_star_indices.shape[0])
    if allowed_star_indices.size == 0:
        raise ValueError(
            "RA/Dec window selected zero stars. "
            "Adjust --timelapse-ra-min/--timelapse-ra-max/--timelapse-dec-min/--timelapse-dec-max."
        )

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
    dataset_dir = run_dir / "dataset"
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
            _log("Timelapse pose validation: requiring each sampled FOV rectangle to stay fully inside the analysis area")
    _log(f"Stop mode: {stop_mode}")
    if stop_mode == "scene_budget":
        _log(f"Scene budget target: {int(scene_budget)}")
        _log(f"Appear cap for acceptance: {appear_cap}")
    elif stop_mode == "appear_target":
        _log(f"Appear target (min per scoped star): {int(appear_target)}")
    elif stop_mode == "appear_mean_target":
        _log(f"Appear target (mean over scoped stars): {int(appear_target)}")
    else:
        _log(
            "Appear band target over scoped stars: "
            f"mean in [{float(band_lower):.3f}, {float(band_upper):.3f}] "
            f"with integer bounds min>={int(band_required_min)} max<={int(band_allowed_max)}"
        )
    if a.max_accepted_scenes is not None:
        _log(f"Hard cap on accepted scenes: {int(a.max_accepted_scenes)}")

    fx, fy, fd = map(math.radians, (FOV_HORIZONTAL_DEG, FOV_VERTICAL_DEG, FOV_DIAGONAL_DEG))
    tx, ty, dcos = math.tan(fx / 2.0), math.tan(fy / 2.0), math.cos(fd / 2.0)
    _log(
        f"FOV fixed: diag={FOV_DIAGONAL_DEG:.6f} deg, "
        f"h={FOV_HORIZONTAL_DEG:.6f} deg, v={FOV_VERTICAL_DEG:.6f} deg"
    )
    _log(
        "Synthetic magnitude perturbation: "
        f"N({float(a.magnitude_perturb_mean):.3f}, {float(a.magnitude_perturb_sigma):.3f})"
    )

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
        "attempted_scene_count": 0,
        "discarded_by_cap_count": 0,
        "discarded_empty_count": 0,
        "discarded_by_ra_count": 0,
    }

    # Coverage counters used both for comparison and stop criteria.
    appear_count = np.zeros(n_stars, dtype=np.int32)
    freq = np.zeros(1024, dtype=np.int64)
    freq[0] = n_stars
    min_count = 0
    max_count = 0
    total_appearances = 0

    # Per-scene metadata to reconstruct sky plots deterministically.
    meta_boresight: list[np.ndarray] = []
    meta_ra: list[float] = []
    meta_dec: list[float] = []
    meta_roll: list[int] = []
    meta_guide: list[int] = []
    meta_real_start: list[int] = []
    meta_real_count: list[int] = []
    meta_real_ids_parts: list[np.ndarray] = []
    flat_real_cursor = 0
    meta_cum_unique: list[int] = []
    meta_cum_total: list[int] = []
    meta_cum_min: list[int] = []
    meta_cum_mean: list[float] = []
    meta_cum_max: list[int] = []

    def _flush() -> None:
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
        size_bytes = int(out.stat().st_size)
        chunks.append(
            {
                "chunk_index": int(chunk_idx),
                "file": str(out.relative_to(run_dir)),
                "scene_count": int(c.shape[0]),
                "point_count": int(data["point_yx"].shape[0]),
                "size_bytes": int(size_bytes),
            }
        )
        for k in ALL_KEYS:
            parts[k].clear()
        bytes_in_buffer = 0
        _log(f"Chunk written: {out.name} ({size_bytes} bytes)")

    if a.max_attempts is not None and int(a.max_attempts) > 0:
        max_attempts = int(a.max_attempts)
    elif stop_mode == "scene_budget":
        max_attempts = max(int(scene_budget) * 100, 10000)
    elif stop_mode in {"appear_target", "appear_mean_target"}:
        max_attempts = max(int(scope_star_count * int(appear_target) * 8), 50000)
    else:
        max_attempts = max(int(scope_star_count * max(float(band_target_mean or 1.0), 1.0) * 12), 50000)

    _log(f"Max candidate attempts: {max_attempts}")
    accepted = 0
    attempts = 0

    while True:
        if a.max_accepted_scenes is not None and accepted >= int(a.max_accepted_scenes):
            _log("Max accepted scenes reached before/at stop condition; finishing run.")
            break
        if stop_mode == "scene_budget" and accepted >= int(scene_budget):
            break
        if stop_mode == "appear_target" and int(np.min(appear_count[allowed_star_indices])) >= int(appear_target):
            break
        if stop_mode == "appear_mean_target" and float(total_appearances / max(scope_star_count, 1)) >= float(appear_target):
            break
        if stop_mode == "appear_band_target":
            scoped_counts = appear_count[allowed_star_indices]
            scoped_mean = float(total_appearances / max(scope_star_count, 1))
            if (
                int(np.min(scoped_counts)) >= int(band_required_min)
                and int(np.max(scoped_counts)) <= int(band_allowed_max)
                and float(band_lower) <= scoped_mean <= float(band_upper)
            ):
                break
        if attempts >= max_attempts:
            raise RuntimeError(
                "Max attempts reached before stop condition. "
                "Try increasing --max-attempts or relaxing --appear-cap/--appear-target/--appear-band-margin."
            )

        attempts += 1
        totals["attempted_scene_count"] += 1

        boresight: np.ndarray | None = None
        if stop_mode == "appear_band_target" and float(rng.uniform(0.0, 1.0)) < BAND_TOPUP_PROBABILITY:
            scoped_counts = appear_count[allowed_star_indices]
            deficit_mask = scoped_counts < int(band_required_min)
            if np.any(deficit_mask):
                deficit_indices = np.flatnonzero(deficit_mask)
                candidates = allowed_star_indices[deficit_indices]
                deficits = (int(band_required_min) - scoped_counts[deficit_indices]).astype(np.float64)
                weights = deficits / max(float(np.sum(deficits)), 1e-12)
                target_star = int(rng.choice(candidates, p=weights))
                boresight = _sample_targeted_boresight(
                    rng,
                    vectors[target_star].astype(np.float64, copy=False),
                    max_offset_deg=float(BAND_TOPUP_MAX_OFFSET_DEG),
                    timelapse=bool(a.timelapse),
                    ra_min=float(a.timelapse_ra_min),
                    ra_max=float(a.timelapse_ra_max),
                    dec_min=float(a.timelapse_dec_min),
                    dec_max=float(a.timelapse_dec_max),
                    require_full_fov_inside=bool(a.timelapse_require_full_fov_inside),
                    fov_edge_samples=int(a.timelapse_fov_edge_samples),
                )
        if boresight is None:
            if a.timelapse:
                boresight = _random_unit_vector_in_window(
                    rng,
                    ra_min=float(a.timelapse_ra_min),
                    ra_max=float(a.timelapse_ra_max),
                    dec_min=float(a.timelapse_dec_min),
                    dec_max=float(a.timelapse_dec_max),
                )
            else:
                boresight = _random_unit_vector(rng)
        roll_deg = int(rng.integers(0, 360))
        scene_seed = int((seed + attempts * 2654435761) & ((1 << 63) - 1))

        if a.timelapse and a.timelapse_require_full_fov_inside:
            if not _pose_has_full_fov_inside_window(
                boresight=boresight,
                roll_deg=roll_deg,
                ra_min=float(a.timelapse_ra_min),
                ra_max=float(a.timelapse_ra_max),
                dec_min=float(a.timelapse_dec_min),
                dec_max=float(a.timelapse_dec_max),
                edge_samples=int(a.timelapse_fov_edge_samples),
            ):
                totals["discarded_by_ra_count"] += 1
                continue

        scene = _generate_scene_from_pose(
            vectors=vectors,
            mags=mags,
            allowed_mask=allowed_star_mask,
            boresight=boresight,
            roll_deg=roll_deg,
            seed=scene_seed,
            tx=tx,
            ty=ty,
            dcos=dcos,
            mag_cutoff=float(a.magnitude_cutoff),
            mag_perturb_mean=float(a.magnitude_perturb_mean),
            mag_perturb_sigma=float(a.magnitude_perturb_sigma),
        )
        if scene is None:
            totals["discarded_empty_count"] += 1
            continue

        scene_real_ids = np.asarray(scene["scene_real_unique_star_ids"], dtype=np.int32)
        if scene_real_ids.size > 0 and np.any(~allowed_star_mask[scene_real_ids]):
            totals["discarded_by_ra_count"] += 1
            continue
        if stop_mode == "scene_budget" and appear_cap is not None and scene_real_ids.size > 0:
            if np.any(appear_count[scene_real_ids] >= int(appear_cap)):
                totals["discarded_by_cap_count"] += 1
                continue
        if stop_mode == "appear_band_target" and scene_real_ids.size > 0:
            if np.any(appear_count[scene_real_ids] >= int(band_allowed_max)):
                totals["discarded_by_cap_count"] += 1
                continue

        if scene_real_ids.size > 0:
            old_counts = appear_count[scene_real_ids]
            for old in old_counts.tolist():
                old_i = int(old)
                if old_i + 1 >= freq.shape[0]:
                    freq = np.pad(freq, (0, max(1024, old_i + 2 - freq.shape[0])), mode="constant")
                freq[old_i] -= 1
                freq[old_i + 1] += 1
            appear_count[scene_real_ids] = old_counts + 1
            total_appearances += int(scene_real_ids.shape[0])
            max_count = max(max_count, int(np.max(old_counts + 1)))

        while min_count < freq.shape[0] and freq[min_count] <= 0:
            min_count += 1

        for k in ALL_KEYS:
            arr = np.asarray(scene[k], dtype=DTYPE[k])
            parts[k].append(arr)
            bytes_in_buffer += int(arr.nbytes)

        totals["generated_scene_count"] += 1
        totals["total_points"] += int(scene["point_yx"].shape[0])
        totals["total_real_points"] += int(scene["scene_real_star_count"][0])
        totals["total_false_points"] += int(scene["scene_false_stars_count"][0])
        totals["total_dropout_count"] += int(scene["scene_dropout_count"][0])

        meta_real_start.append(int(flat_real_cursor))
        meta_real_count.append(int(scene_real_ids.shape[0]))
        if scene_real_ids.size > 0:
            meta_real_ids_parts.append(scene_real_ids.astype(np.int32, copy=False))
            flat_real_cursor += int(scene_real_ids.shape[0])

        meta_boresight.append(np.asarray(scene["scene_boresight_xyz"], dtype=np.float32))
        meta_ra.append(float(scene["scene_center_ra_deg"]))
        meta_dec.append(float(scene["scene_center_dec_deg"]))
        meta_roll.append(int(scene["roll_degree"][0]))
        meta_guide.append(int(scene["guide_star_index"][0]))
        meta_cum_unique.append(int(np.sum(appear_count > 0)))
        meta_cum_total.append(int(total_appearances))
        meta_cum_min.append(int(min_count))
        meta_cum_mean.append(float(total_appearances / max(scope_star_count, 1)))
        meta_cum_max.append(int(max_count))

        accepted += 1

        if bytes_in_buffer >= chunk_target:
            _flush()

        if accepted % 5000 == 0:
            _log(
                f"Accepted scenes: {accepted} | attempts: {attempts} | "
                f"appear min/mean/max: "
                f"{int(np.min(appear_count[allowed_star_indices]))}/"
                f"{float(total_appearances / max(scope_star_count, 1)):.3f}/"
                f"{int(max_count)}"
            )

    _flush()
    end_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    real_star_ids = (
        np.concatenate(meta_real_ids_parts, axis=0).astype(np.int32, copy=False)
        if meta_real_ids_parts
        else np.empty((0,), dtype=np.int32)
    )
    scene_meta = {
        "scene_boresight_xyz": np.asarray(meta_boresight, dtype=np.float32),
        "scene_center_ra_deg": np.asarray(meta_ra, dtype=np.float32),
        "scene_center_dec_deg": np.asarray(meta_dec, dtype=np.float32),
        "scene_roll_degree": np.asarray(meta_roll, dtype=np.int16),
        "scene_guide_star_index": np.asarray(meta_guide, dtype=np.int32),
        "scene_real_star_start": np.asarray(meta_real_start, dtype=np.int64),
        "scene_real_star_count": np.asarray(meta_real_count, dtype=np.int32),
        "scene_real_star_id": real_star_ids,
        "cumulative_unique_seen_stars": np.asarray(meta_cum_unique, dtype=np.int32),
        "cumulative_total_appearances": np.asarray(meta_cum_total, dtype=np.int64),
        "cumulative_min_appear_count": np.asarray(meta_cum_min, dtype=np.int32),
        "cumulative_mean_appear_count": np.asarray(meta_cum_mean, dtype=np.float32),
        "cumulative_max_appear_count": np.asarray(meta_cum_max, dtype=np.int32),
        "final_appear_count": appear_count.astype(np.int32, copy=False),
    }
    np.savez_compressed(run_dir / "scene_metadata.npz", **scene_meta)
    np.savez_compressed(run_dir / "coverage_stats.npz", final_appear_count=appear_count.astype(np.int32, copy=False))

    scoped = appear_count[allowed_star_indices]
    coverage = {
        "stars": int(n_stars),
        "coverage_scope_stars": int(allowed_star_indices.shape[0]),
        "scene_count": int(totals["generated_scene_count"]),
        "appear_count_min": int(np.min(scoped)) if scoped.size else 0,
        "appear_count_mean": float(np.mean(scoped)) if scoped.size else 0.0,
        "appear_count_max": int(np.max(scoped)) if scoped.size else 0,
        "unique_seen_stars": int(np.sum(scoped > 0)),
        "total_appearances": int(np.sum(scoped)),
    }
    (run_dir / "coverage_summary.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

    counts = {
        "generated_scene_count": int(totals["generated_scene_count"]),
        "chunks_count": int(len(chunks)),
        "total_points": int(totals["total_points"]),
        "total_real_points": int(totals["total_real_points"]),
        "total_false_points": int(totals["total_false_points"]),
        "total_dropout_count": int(totals["total_dropout_count"]),
        "attempted_scene_count": int(totals["attempted_scene_count"]),
        "discarded_by_cap_count": int(totals["discarded_by_cap_count"]),
        "discarded_empty_count": int(totals["discarded_empty_count"]),
        "discarded_by_ra_count": int(totals["discarded_by_ra_count"]),
        "appear_count_min": int(coverage["appear_count_min"]),
        "appear_count_mean": float(coverage["appear_count_mean"]),
        "appear_count_max": int(coverage["appear_count_max"]),
        "unique_seen_stars": int(coverage["unique_seen_stars"]),
        "total_appearances": int(coverage["total_appearances"]),
    }

    params = {
        "stop_mode": stop_mode,
        "scene_budget": int(scene_budget) if scene_budget is not None else None,
        "max_accepted_scenes": int(a.max_accepted_scenes) if a.max_accepted_scenes is not None else None,
        "appear_target": int(appear_target) if appear_target is not None else None,
        "appear_target_semantics": (
            "min_per_scoped_star"
            if stop_mode == "appear_target"
            else "mean_over_scoped_stars"
            if stop_mode == "appear_mean_target"
            else "band_over_scoped_stars"
            if stop_mode == "appear_band_target"
            else None
        ),
        "appear_band_target_mean": float(band_target_mean) if band_target_mean is not None else None,
        "appear_band_margin": float(a.appear_band_margin),
        "appear_band_lower": float(band_lower) if band_lower is not None else None,
        "appear_band_upper": float(band_upper) if band_upper is not None else None,
        "appear_band_required_min_count": int(band_required_min) if band_required_min is not None else None,
        "appear_band_allowed_max_count": int(band_allowed_max) if band_allowed_max is not None else None,
        "appear_cap": int(appear_cap) if appear_cap is not None else None,
        "appear_cap_margin": float(a.appear_cap_margin),
        "baseline_run": str(a.baseline_run.resolve()) if a.baseline_run is not None else None,
        "fov_diagonal_deg": float(FOV_DIAGONAL_DEG),
        "fov_horizontal_deg": float(FOV_HORIZONTAL_DEG),
        "fov_vertical_deg": float(FOV_VERTICAL_DEG),
        "magnitude_cutoff": float(a.magnitude_cutoff),
        "magnitude_perturb_mean": float(a.magnitude_perturb_mean),
        "magnitude_perturb_sigma": float(a.magnitude_perturb_sigma),
        "resolution": [WIDTH, HEIGHT],
        "chunk_size_mb": int(a.chunk_size_mb),
        "num_workers": 1,
        "seed": int(seed),
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
        "every_nth_scene": int(a.every_nth_scene),
    }

    manifest = {
        "run": {
            "name": f"run{run_idx}",
            "run_index": int(run_idx),
            "created_at_utc": start_utc,
            "finished_at_utc": end_utc,
            "database_path": str(db_path),
            "database_star_count": int(n_stars),
            "parameters": params,
        },
        "guide_star_indices": None,
        "guide_star_catalog_ids": None,
        "counts": counts,
        "chunks": chunks,
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        duration_seconds = (datetime.fromisoformat(end_utc) - datetime.fromisoformat(start_utc)).total_seconds()
    except Exception:
        duration_seconds = 0.0

    summary_lines = [
        f"run: run{run_idx}",
        f"started_at_utc: {start_utc}",
        f"ended_at_utc: {end_utc}",
        f"duration_seconds: {duration_seconds:.2f}",
        "",
        "parameters:",
    ]
    summary_lines += [f"  {k}: {v}" for k, v in params.items()]
    summary_lines += ["", "counts:"]
    summary_lines += [f"  {k}: {v}" for k, v in counts.items()]
    (run_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    _log("Run finished successfully")
    print(f"Run complete: run{run_idx}")
    print(f"Run directory: {run_dir}")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Scenes generated: {totals['generated_scene_count']}")
    print(f"Chunks written: {len(chunks)}")
    print(
        "Appear count stats: "
        f"min={int(coverage['appear_count_min'])} "
        f"mean={float(coverage['appear_count_mean']):.3f} "
        f"max={int(coverage['appear_count_max'])}"
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


if __name__ == "__main__":
    main()
