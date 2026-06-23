#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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

DEFAULT_DB = Path(__file__).resolve().parents[1] / "tetra3" / "data" / "default_database.npz"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_ra_deg(value: float) -> float:
    return float(value) % 360.0


def ra_in_window(ra_deg: float, ra_min: float, ra_max: float) -> bool:
    ra = normalize_ra_deg(ra_deg)
    left = normalize_ra_deg(ra_min)
    right = normalize_ra_deg(ra_max)
    if left <= right:
        return left <= ra <= right
    return ra >= left or ra <= right


def dec_in_window(dec_deg: float, dec_min: float, dec_max: float) -> bool:
    low = min(float(dec_min), float(dec_max))
    high = max(float(dec_min), float(dec_max))
    return low <= float(dec_deg) <= high


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a subset dataset from an existing run by keeping scenes that contain "
            "at least one real star from a target RA/Dec area."
        )
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ra-min", type=float, required=True)
    parser.add_argument("--ra-max", type=float, required=True)
    parser.add_argument("--dec-min", type=float, required=True)
    parser.add_argument("--dec-max", type=float, required=True)
    parser.add_argument("--expected-star-count", type=int, default=None)
    parser.add_argument("--chunk-size-mb", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset_manifest.json not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_database_area(
    database_path: Path,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    with np.load(database_path, allow_pickle=False) as data:
        if "star_table" not in data:
            raise RuntimeError(f"Database missing star_table: {database_path}")
        star_table = np.asarray(data["star_table"], dtype=np.float32)
        catalog_ids = np.asarray(data["star_catalog_IDs"]) if "star_catalog_IDs" in data else None

    vectors = np.asarray(star_table[:, 2:5], dtype=np.float64)
    ra = (np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0)))

    target_mask = np.array(
        [
            ra_in_window(float(ra_i), float(ra_min), float(ra_max))
            and dec_in_window(float(dec_i), float(dec_min), float(dec_max))
            for ra_i, dec_i in zip(ra.tolist(), dec.tolist())
        ],
        dtype=bool,
    )
    target_ids = np.flatnonzero(target_mask).astype(np.int32, copy=False)
    return target_ids, catalog_ids, int(star_table.shape[0])


def chunk_paths(run_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    for item in manifest.get("chunks", []):
        rel = str(item["file"]).replace("\\", "/")
        p = (run_dir / rel).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Chunk file not found: {p}")
        out.append(p)
    if not out:
        raise RuntimeError("No chunks found in source manifest")
    return out


def scene_target_indices_for_chunk(
    point_star_id: np.ndarray,
    scene_point_start: np.ndarray,
    scene_point_count: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    point_star_id = np.asarray(point_star_id, dtype=np.int32)
    point_hits = np.zeros(point_star_id.shape[0], dtype=np.int8)
    real_mask = point_star_id >= 0
    if np.any(real_mask):
        point_hits[real_mask] = np.asarray(target_mask[point_star_id[real_mask]], dtype=np.int8)

    prefix = np.zeros(point_hits.shape[0] + 1, dtype=np.int64)
    np.cumsum(point_hits, out=prefix[1:])

    scene_start = np.asarray(scene_point_start, dtype=np.int64)
    scene_end = scene_start + np.asarray(scene_point_count, dtype=np.int64)
    scene_hit_count = prefix[scene_end] - prefix[scene_start]
    return np.flatnonzero(scene_hit_count > 0).astype(np.int64, copy=False)


def write_summary(
    output_dir: Path,
    *,
    start_utc: str,
    end_utc: str,
    source_run: Path,
    source_scene_count: int,
    target_star_count: int,
    selected_scene_count: int,
    coverage_summary: dict[str, Any],
    counts: dict[str, int],
    area: dict[str, float],
) -> None:
    try:
        duration = (
            datetime.fromisoformat(end_utc) - datetime.fromisoformat(start_utc)
        ).total_seconds()
    except Exception:
        duration = 0.0

    lines = [
        f"run: {output_dir.name}",
        f"started_at_utc: {start_utc}",
        f"ended_at_utc: {end_utc}",
        f"duration_seconds: {duration:.2f}",
        "",
        "source:",
        f"  run_dir: {source_run}",
        f"  scene_count: {source_scene_count}",
        "",
        "target_area:",
        f"  ra_min: {area['ra_min']}",
        f"  ra_max: {area['ra_max']}",
        f"  dec_min: {area['dec_min']}",
        f"  dec_max: {area['dec_max']}",
        f"  target_star_count: {target_star_count}",
        "",
        "selection:",
        "  mode: scene_contains_any_target_star",
        f"  selected_scene_count: {selected_scene_count}",
        "",
        "counts:",
        f"  chunks_count: {counts['chunks_count']}",
        f"  total_points: {counts['total_points']}",
        f"  total_real_points: {counts['total_real_points']}",
        f"  total_false_points: {counts['total_false_points']}",
        f"  total_dropout_count: {counts['total_dropout_count']}",
        "",
        "coverage_summary:",
        f"  appear_count_min: {coverage_summary['appear_count_min']}",
        f"  appear_count_mean: {coverage_summary['appear_count_mean']}",
        f"  appear_count_max: {coverage_summary['appear_count_max']}",
        f"  unique_seen_stars: {coverage_summary['unique_seen_stars']}",
        f"  total_appearances: {coverage_summary['total_appearances']}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    source_run = args.source_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    database_path = args.database.expanduser().resolve()

    manifest = load_manifest(source_run)
    source_chunk_paths = chunk_paths(source_run, manifest)

    target_star_ids, catalog_ids, database_star_count = load_database_area(
        database_path=database_path,
        ra_min=float(args.ra_min),
        ra_max=float(args.ra_max),
        dec_min=float(args.dec_min),
        dec_max=float(args.dec_max),
    )
    target_star_count = int(target_star_ids.shape[0])
    if target_star_count <= 0:
        raise RuntimeError("Target area selected zero stars")
    if args.expected_star_count is not None and target_star_count != int(args.expected_star_count):
        raise RuntimeError(
            f"Expected {int(args.expected_star_count)} stars in target area, got {target_star_count}"
        )

    target_mask = np.zeros(database_star_count, dtype=bool)
    target_mask[target_star_ids.astype(np.int64)] = True

    scene_meta_path = source_run / "scene_metadata.npz"
    if not scene_meta_path.exists():
        raise FileNotFoundError(f"scene_metadata.npz not found: {scene_meta_path}")
    with np.load(scene_meta_path, allow_pickle=False) as scene_meta:
        source_boresight_xyz = np.asarray(scene_meta["scene_boresight_xyz"], dtype=np.float32)
        source_center_ra = np.asarray(scene_meta["scene_center_ra_deg"], dtype=np.float32)
        source_center_dec = np.asarray(scene_meta["scene_center_dec_deg"], dtype=np.float32)
        source_roll_degree = np.asarray(scene_meta["scene_roll_degree"], dtype=np.int16)
        source_guide_star_index = np.asarray(scene_meta["scene_guide_star_index"], dtype=np.int32)

    selected_local_by_chunk: list[np.ndarray] = []
    selected_global_parts: list[np.ndarray] = []
    chunk_scene_offsets: list[int] = []
    scene_offset = 0

    log(f"Source run: {source_run}")
    log(
        "Target area: "
        f"RA[{float(args.ra_min):.3f}, {float(args.ra_max):.3f}] "
        f"Dec[{float(args.dec_min):.3f}, {float(args.dec_max):.3f}] "
        f"-> {target_star_count} stars"
    )

    for chunk_idx, chunk_path in enumerate(source_chunk_paths):
        chunk_scene_offsets.append(scene_offset)
        with np.load(chunk_path, allow_pickle=False) as data:
            scene_point_start = np.asarray(data["scene_point_start"], dtype=np.int64)
            scene_point_count = np.asarray(data["scene_point_count"], dtype=np.int64)
            point_star_id = np.asarray(data["point_star_id"], dtype=np.int32)
            selected_local = scene_target_indices_for_chunk(
                point_star_id=point_star_id,
                scene_point_start=scene_point_start,
                scene_point_count=scene_point_count,
                target_mask=target_mask,
            )

        selected_local_by_chunk.append(selected_local)
        if selected_local.size > 0:
            selected_global_parts.append((selected_local + scene_offset).astype(np.int64, copy=False))
        scene_offset += int(scene_point_count.shape[0])
        log(
            f"Chunk {chunk_idx + 1}/{len(source_chunk_paths)} scanned: "
            f"{int(selected_local.size)} selected scenes"
        )

    selected_global_scene_idx = (
        np.concatenate(selected_global_parts, axis=0).astype(np.int64, copy=False)
        if selected_global_parts
        else np.empty((0,), dtype=np.int64)
    )
    selected_scene_count = int(selected_global_scene_idx.shape[0])
    source_scene_count = int(scene_offset)

    log(f"Selected scenes containing target stars: {selected_scene_count}/{source_scene_count}")
    if selected_scene_count <= 0:
        raise RuntimeError("No scenes contain target stars in the requested area")

    if args.dry_run:
        print(f"target_star_count={target_star_count}")
        print(f"selected_scene_count={selected_scene_count}")
        return 0

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=False)

    chunk_target_bytes = int(args.chunk_size_mb) * 1024 * 1024
    chunk_idx_out = 0
    bytes_in_buffer = 0
    parts = {key: [] for key in ALL_KEYS}
    chunks_out: list[dict[str, object]] = []

    totals = {
        "generated_scene_count": 0,
        "total_points": 0,
        "total_real_points": 0,
        "total_false_points": 0,
        "total_dropout_count": 0,
    }

    target_index_of = np.full(database_star_count, -1, dtype=np.int32)
    target_index_of[target_star_ids.astype(np.int64)] = np.arange(target_star_count, dtype=np.int32)
    target_appear_count = np.zeros(target_star_count, dtype=np.int32)

    subset_boresight_xyz = np.zeros((selected_scene_count, 3), dtype=np.float32)
    subset_center_ra = np.zeros(selected_scene_count, dtype=np.float32)
    subset_center_dec = np.zeros(selected_scene_count, dtype=np.float32)
    subset_roll_degree = np.zeros(selected_scene_count, dtype=np.int16)
    subset_guide_star_index = np.zeros(selected_scene_count, dtype=np.int32)

    subset_real_start = np.zeros(selected_scene_count, dtype=np.int64)
    subset_real_count = np.zeros(selected_scene_count, dtype=np.int32)
    subset_real_ids_parts: list[np.ndarray] = []
    subset_real_cursor = 0

    cumulative_unique = np.zeros(selected_scene_count, dtype=np.int32)
    cumulative_total = np.zeros(selected_scene_count, dtype=np.int64)
    cumulative_min = np.zeros(selected_scene_count, dtype=np.int32)
    cumulative_mean = np.zeros(selected_scene_count, dtype=np.float32)
    cumulative_max = np.zeros(selected_scene_count, dtype=np.int32)
    cumulative_total_appearances = 0

    def flush() -> None:
        nonlocal chunk_idx_out, bytes_in_buffer
        if not parts["scene_point_count"]:
            return

        data = {
            key: np.concatenate(parts[key], axis=0).astype(DTYPE[key], copy=False)
            for key in ALL_KEYS
        }
        scene_counts = np.asarray(data["scene_point_count"], dtype=np.int64)
        if scene_counts.shape[0] > 1:
            data["scene_point_start"] = np.concatenate(
                ([0], np.cumsum(scene_counts[:-1], dtype=np.int64))
            )
        else:
            data["scene_point_start"] = np.zeros(scene_counts.shape[0], dtype=np.int64)

        chunk_idx_out += 1
        out_path = dataset_dir / f"dataset{chunk_idx_out}.npz"
        np.savez_compressed(out_path, **data)
        size_bytes = int(out_path.stat().st_size)
        chunks_out.append(
            {
                "chunk_index": int(chunk_idx_out),
                "file": str(out_path.relative_to(output_dir)),
                "scene_count": int(scene_counts.shape[0]),
                "point_count": int(data["point_yx"].shape[0]),
                "size_bytes": size_bytes,
            }
        )
        for key in ALL_KEYS:
            parts[key].clear()
        bytes_in_buffer = 0
        log(f"Chunk written: {out_path.name} ({size_bytes} bytes)")

    subset_scene_idx = 0
    start_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for chunk_idx, chunk_path in enumerate(source_chunk_paths):
        selected_local = selected_local_by_chunk[chunk_idx]
        if selected_local.size == 0:
            continue

        with np.load(chunk_path, allow_pickle=False) as data:
            point_yx = np.asarray(data["point_yx"], dtype=np.float32)
            point_star_id = np.asarray(data["point_star_id"], dtype=np.int32)
            point_is_false_star = np.where(
                np.asarray(data["point_is_false_star"], dtype=bool),
                -1,
                0,
            ).astype(np.int8, copy=False)
            point_magnitude = np.asarray(data["point_magnitude"], dtype=np.float32)
            scene_point_start = np.asarray(data["scene_point_start"], dtype=np.int64)
            scene_point_count = np.asarray(data["scene_point_count"], dtype=np.int64)
            guide_star_index = np.asarray(data["guide_star_index"], dtype=np.int32)
            pre_dropout_real_star_count = np.asarray(data["pre_dropout_real_star_count"], dtype=np.int32)
            scene_dropout_count = np.asarray(data["scene_dropout_count"], dtype=np.int32)
            scene_false_stars_count = np.asarray(data["scene_false_stars_count"], dtype=np.int32)
            scene_real_star_count = np.asarray(data["scene_real_star_count"], dtype=np.int32)
            scene_total_point_count = np.asarray(data["scene_total_point_count"], dtype=np.int32)
            scene_seed = np.asarray(data["scene_seed"], dtype=np.int64)
            roll_degree = np.asarray(data["roll_degree"], dtype=np.int16)

            for local_scene_idx in selected_local.tolist():
                start = int(scene_point_start[local_scene_idx])
                count = int(scene_point_count[local_scene_idx])
                end = start + count

                scene_point_star_id = point_star_id[start:end]
                scene_real_ids = np.unique(scene_point_star_id[scene_point_star_id >= 0]).astype(
                    np.int32,
                    copy=False,
                )
                target_scene_ids = (
                    scene_real_ids[target_mask[scene_real_ids.astype(np.int64)]]
                    if scene_real_ids.size > 0
                    else np.empty((0,), dtype=np.int32)
                )

                parts["point_yx"].append(point_yx[start:end])
                parts["point_star_id"].append(scene_point_star_id)
                parts["point_is_false_star"].append(point_is_false_star[start:end])
                parts["point_magnitude"].append(point_magnitude[start:end])
                parts["scene_point_count"].append(
                    np.asarray([scene_point_count[local_scene_idx]], dtype=np.int32)
                )
                parts["guide_star_index"].append(
                    np.asarray([guide_star_index[local_scene_idx]], dtype=np.int32)
                )
                parts["pre_dropout_real_star_count"].append(
                    np.asarray([pre_dropout_real_star_count[local_scene_idx]], dtype=np.int32)
                )
                parts["scene_dropout_count"].append(
                    np.asarray([scene_dropout_count[local_scene_idx]], dtype=np.int32)
                )
                parts["scene_false_stars_count"].append(
                    np.asarray([scene_false_stars_count[local_scene_idx]], dtype=np.int32)
                )
                parts["scene_real_star_count"].append(
                    np.asarray([scene_real_star_count[local_scene_idx]], dtype=np.int32)
                )
                parts["scene_total_point_count"].append(
                    np.asarray([scene_total_point_count[local_scene_idx]], dtype=np.int32)
                )
                parts["scene_seed"].append(np.asarray([scene_seed[local_scene_idx]], dtype=np.int64))
                parts["roll_degree"].append(
                    np.asarray([roll_degree[local_scene_idx]], dtype=np.int16)
                )

                bytes_in_buffer += int(point_yx[start:end].nbytes)
                bytes_in_buffer += int(scene_point_star_id.nbytes)
                bytes_in_buffer += int(point_is_false_star[start:end].nbytes)
                bytes_in_buffer += int(point_magnitude[start:end].nbytes)
                bytes_in_buffer += 8 * (4 + 4 + 4 + 4 + 4 + 4) + 8 + 2

                totals["generated_scene_count"] += 1
                totals["total_points"] += count
                totals["total_real_points"] += int(scene_real_star_count[local_scene_idx])
                totals["total_false_points"] += int(scene_false_stars_count[local_scene_idx])
                totals["total_dropout_count"] += int(scene_dropout_count[local_scene_idx])

                global_scene_idx = int(chunk_scene_offsets[chunk_idx] + local_scene_idx)
                subset_boresight_xyz[subset_scene_idx] = source_boresight_xyz[global_scene_idx]
                subset_center_ra[subset_scene_idx] = source_center_ra[global_scene_idx]
                subset_center_dec[subset_scene_idx] = source_center_dec[global_scene_idx]
                subset_roll_degree[subset_scene_idx] = source_roll_degree[global_scene_idx]
                subset_guide_star_index[subset_scene_idx] = source_guide_star_index[global_scene_idx]

                subset_real_start[subset_scene_idx] = int(subset_real_cursor)
                subset_real_count[subset_scene_idx] = int(target_scene_ids.shape[0])
                if target_scene_ids.size > 0:
                    subset_real_ids_parts.append(target_scene_ids)
                    subset_real_cursor += int(target_scene_ids.shape[0])
                    target_positions = target_index_of[target_scene_ids.astype(np.int64)]
                    target_appear_count[target_positions] += 1
                    cumulative_total_appearances += int(target_scene_ids.shape[0])

                cumulative_unique[subset_scene_idx] = int(np.sum(target_appear_count > 0))
                cumulative_total[subset_scene_idx] = int(cumulative_total_appearances)
                cumulative_min[subset_scene_idx] = int(np.min(target_appear_count))
                cumulative_mean[subset_scene_idx] = float(
                    cumulative_total_appearances / max(target_star_count, 1)
                )
                cumulative_max[subset_scene_idx] = int(np.max(target_appear_count))

                subset_scene_idx += 1

                if bytes_in_buffer >= chunk_target_bytes:
                    flush()

        log(
            f"Copied chunk {chunk_idx + 1}/{len(source_chunk_paths)}: "
            f"{int(selected_local.size)} scenes"
        )

    flush()
    end_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    subset_real_ids = (
        np.concatenate(subset_real_ids_parts, axis=0).astype(np.int32, copy=False)
        if subset_real_ids_parts
        else np.empty((0,), dtype=np.int32)
    )

    scene_meta = {
        "scene_boresight_xyz": subset_boresight_xyz,
        "scene_center_ra_deg": subset_center_ra,
        "scene_center_dec_deg": subset_center_dec,
        "scene_roll_degree": subset_roll_degree,
        "scene_guide_star_index": subset_guide_star_index,
        "scene_real_star_start": subset_real_start,
        "scene_real_star_count": subset_real_count,
        "scene_real_star_id": subset_real_ids,
        "cumulative_unique_seen_stars": cumulative_unique,
        "cumulative_total_appearances": cumulative_total,
        "cumulative_min_appear_count": cumulative_min,
        "cumulative_mean_appear_count": cumulative_mean,
        "cumulative_max_appear_count": cumulative_max,
        "final_appear_count": target_appear_count.astype(np.int32, copy=False),
    }
    np.savez_compressed(output_dir / "scene_metadata.npz", **scene_meta)
    np.savez_compressed(output_dir / "coverage_stats.npz", final_appear_count=target_appear_count)

    coverage_summary = {
        "stars": int(target_star_count),
        "coverage_scope_stars": int(target_star_count),
        "scene_count": int(selected_scene_count),
        "appear_count_min": int(np.min(target_appear_count)),
        "appear_count_mean": float(np.mean(target_appear_count)),
        "appear_count_max": int(np.max(target_appear_count)),
        "unique_seen_stars": int(np.sum(target_appear_count > 0)),
        "total_appearances": int(np.sum(target_appear_count)),
    }
    (output_dir / "coverage_summary.json").write_text(
        json.dumps(coverage_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    source_run_meta = manifest.get("run", {}) if isinstance(manifest.get("run"), dict) else {}
    source_params = (
        source_run_meta.get("parameters", {})
        if isinstance(source_run_meta.get("parameters"), dict)
        else {}
    )
    source_counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}

    run_name = output_dir.name
    run_index = int(run_name[3:]) if run_name.startswith("run") and run_name[3:].isdigit() else 0
    params = {
        "source_run": str(source_run),
        "source_run_name": str(source_run_meta.get("name", source_run.name)),
        "source_scene_count": int(source_counts.get("generated_scene_count", source_scene_count)),
        "selection_mode": "scene_contains_any_target_star",
        "ra_window_deg": [float(args.ra_min), float(args.ra_max)],
        "dec_window_deg": [float(args.dec_min), float(args.dec_max)],
        "target_star_count": int(target_star_count),
        "chunk_size_mb": int(args.chunk_size_mb),
        "database_path": str(database_path),
        "database_star_count": int(database_star_count),
        "timelapse": False,
        "every_nth_scene": 1,
        "source_fov_diagonal_deg": float(source_params.get("fov_diagonal_deg", 21.0)),
        "source_fov_horizontal_deg": float(source_params.get("fov_horizontal_deg", 17.2)),
        "source_fov_vertical_deg": float(source_params.get("fov_vertical_deg", 13.0)),
        "source_resolution": source_params.get("resolution", [1280, 960]),
    }
    counts = {
        "generated_scene_count": int(selected_scene_count),
        "chunks_count": int(len(chunks_out)),
        "total_points": int(totals["total_points"]),
        "total_real_points": int(totals["total_real_points"]),
        "total_false_points": int(totals["total_false_points"]),
        "total_dropout_count": int(totals["total_dropout_count"]),
        "coverage_scope_stars": int(target_star_count),
        "appear_count_min": int(coverage_summary["appear_count_min"]),
        "appear_count_mean": float(coverage_summary["appear_count_mean"]),
        "appear_count_max": int(coverage_summary["appear_count_max"]),
        "unique_seen_stars": int(coverage_summary["unique_seen_stars"]),
        "total_appearances": int(coverage_summary["total_appearances"]),
    }
    manifest_out = {
        "run": {
            "name": run_name,
            "run_index": int(run_index),
            "created_at_utc": start_utc,
            "finished_at_utc": end_utc,
            "database_path": str(database_path),
            "database_star_count": int(database_star_count),
            "parameters": params,
        },
        "guide_star_indices": None,
        "guide_star_catalog_ids": None,
        "class_star_ids": target_star_ids.astype(np.int32).tolist(),
        "class_catalog_ids": (
            catalog_ids[target_star_ids.astype(np.int64)].tolist() if catalog_ids is not None else None
        ),
        "counts": counts,
        "chunks": chunks_out,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest_out, indent=2) + "\n",
        encoding="utf-8",
    )

    write_summary(
        output_dir=output_dir,
        start_utc=start_utc,
        end_utc=end_utc,
        source_run=source_run,
        source_scene_count=source_scene_count,
        target_star_count=target_star_count,
        selected_scene_count=selected_scene_count,
        coverage_summary=coverage_summary,
        counts={
            "chunks_count": int(len(chunks_out)),
            "total_points": int(totals["total_points"]),
            "total_real_points": int(totals["total_real_points"]),
            "total_false_points": int(totals["total_false_points"]),
            "total_dropout_count": int(totals["total_dropout_count"]),
        },
        area={
            "ra_min": float(args.ra_min),
            "ra_max": float(args.ra_max),
            "dec_min": float(args.dec_min),
            "dec_max": float(args.dec_max),
        },
    )

    log(f"Subset dataset written: {output_dir}")
    log(f"Selected scenes: {selected_scene_count}")
    log(
        "Coverage over target stars: "
        f"min={coverage_summary['appear_count_min']} "
        f"mean={coverage_summary['appear_count_mean']:.3f} "
        f"max={coverage_summary['appear_count_max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
