#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DB = Path(__file__).resolve().parents[1] / "tetra3" / "data" / "default_database.npz"


@dataclass(frozen=True)
class ChunkInfo:
    path: Path
    scene_count: int
    scene_offset: int


def log(message: str) -> None:
    print(message, flush=True)


def circular_diff_deg(a: float, b: float) -> float:
    diff = abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
    return float(diff)


def angular_sep_deg(ra0: float, dec0: float, ra1: float, dec1: float) -> float:
    ra0r = math.radians(float(ra0))
    dec0r = math.radians(float(dec0))
    ra1r = math.radians(float(ra1))
    dec1r = math.radians(float(dec1))
    cos_d = (
        math.sin(dec0r) * math.sin(dec1r)
        + math.cos(dec0r) * math.cos(dec1r) * math.cos(ra0r - ra1r)
    )
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.degrees(math.acos(cos_d))


def parse_solution_txt(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"ra_deg", "dec_deg", "roll_deg", "fov_horizontal_deg", "fov_vertical_deg"}:
            out[key] = float(value)
    missing = [k for k in ("ra_deg", "dec_deg", "roll_deg") if k not in out]
    if missing:
        raise RuntimeError(f"Missing keys in solution txt: {missing}")
    return out


def parse_matched_catalog_ids(path: Path) -> list[int]:
    ids: list[int] = []
    seen_data = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("columns:"):
            seen_data = True
            continue
        if not seen_data:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        ids.append(int(parts[1]))
    if not ids:
        raise RuntimeError(f"No matched catalog IDs found in {path}")
    return ids


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset_manifest.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(run_dir: Path, manifest: dict[str, Any]) -> list[ChunkInfo]:
    chunks: list[ChunkInfo] = []
    offset = 0
    for item in manifest.get("chunks", []):
        rel = item.get("file")
        scene_count = int(item.get("scene_count", 0))
        if not rel or scene_count <= 0:
            continue
        path = (run_dir / str(rel)).resolve()
        chunks.append(ChunkInfo(path=path, scene_count=scene_count, scene_offset=offset))
        offset += scene_count
    if not chunks:
        raise RuntimeError("No chunks found in manifest")
    return chunks


def load_catalog_lookup(database_path: Path) -> np.ndarray:
    with np.load(database_path, allow_pickle=False) as data:
        if "star_catalog_IDs" not in data:
            raise RuntimeError(f"Database missing star_catalog_IDs: {database_path}")
        return np.asarray(data["star_catalog_IDs"])


def find_chunk_for_scene(chunks: list[ChunkInfo], global_scene_idx: int) -> tuple[ChunkInfo, int]:
    for chunk in chunks:
        if global_scene_idx < chunk.scene_offset + chunk.scene_count:
            return chunk, int(global_scene_idx - chunk.scene_offset)
    raise IndexError(f"Scene index out of range: {global_scene_idx}")


def load_scene_meta(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "scene_metadata.npz"
    if not path.exists():
        raise FileNotFoundError(f"scene_metadata.npz not found: {path}")
    with np.load(path, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def best_scene_by_overlap(
    *,
    run_dir: Path,
    matched_catalog_ids: list[int],
    tetra_ra_deg: float,
    tetra_dec_deg: float,
    tetra_roll_deg: float,
    database_path: Path,
) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    scene_meta = load_scene_meta(run_dir)
    catalog_lookup = load_catalog_lookup(database_path)

    real_start = np.asarray(scene_meta["scene_real_star_start"], dtype=np.int64)
    real_count = np.asarray(scene_meta["scene_real_star_count"], dtype=np.int32)
    real_ids_flat = np.asarray(scene_meta["scene_real_star_id"], dtype=np.int32)
    scene_ra = np.asarray(scene_meta["scene_center_ra_deg"], dtype=np.float32)
    scene_dec = np.asarray(scene_meta["scene_center_dec_deg"], dtype=np.float32)
    scene_roll = np.asarray(scene_meta["scene_roll_degree"], dtype=np.float32)

    matched_set = set(int(x) for x in matched_catalog_ids)
    best: dict[str, Any] | None = None
    best_key: tuple[int, float, float, int] | None = None

    for scene_idx in range(int(real_start.shape[0])):
        start = int(real_start[scene_idx])
        count = int(real_count[scene_idx])
        real_ids = real_ids_flat[start : start + count]
        if real_ids.size == 0:
            continue
        scene_catalog_ids = [int(catalog_lookup[int(star_id)]) for star_id in real_ids.tolist()]
        overlap_ids = sorted(set(scene_catalog_ids) & matched_set)
        overlap = int(len(overlap_ids))
        if overlap <= 0:
            continue
        ang = float(
            angular_sep_deg(
                float(scene_ra[scene_idx]),
                float(scene_dec[scene_idx]),
                float(tetra_ra_deg),
                float(tetra_dec_deg),
            )
        )
        roll_diff = float(circular_diff_deg(float(scene_roll[scene_idx]), float(tetra_roll_deg)))
        # Higher overlap is better; smaller angular/roll deltas are better.
        key = (overlap, -ang, -roll_diff, -scene_idx)
        if best is None or key > best_key:
            best = {
                "scene_idx": int(scene_idx),
                "scene_center_ra_deg": float(scene_ra[scene_idx]),
                "scene_center_dec_deg": float(scene_dec[scene_idx]),
                "scene_roll_degree": float(scene_roll[scene_idx]),
                "overlap_count": overlap,
                "overlap_catalog_ids": overlap_ids,
                "angular_sep_deg": ang,
                "roll_diff_deg": roll_diff,
            }
            best_key = key

    if best is None:
        raise RuntimeError("No scene with overlap against tetra matched catalog IDs was found")

    best["matched_catalog_ids"] = [int(x) for x in matched_catalog_ids]
    best["manifest_run_name"] = str((manifest.get("run") or {}).get("name", run_dir.name))
    return best


def render_scene_with_catalog_ids(
    *,
    run_dir: Path,
    scene_idx: int,
    overlap_catalog_ids: set[int],
    database_path: Path,
    output_png: Path,
    summary: dict[str, Any],
) -> None:
    manifest = load_manifest(run_dir)
    chunks = load_chunks(run_dir, manifest)
    catalog_lookup = load_catalog_lookup(database_path)
    chunk, local_scene_idx = find_chunk_for_scene(chunks, int(scene_idx))

    with np.load(chunk.path, allow_pickle=False) as data:
        start = int(data["scene_point_start"][local_scene_idx])
        count = int(data["scene_point_count"][local_scene_idx])
        end = start + count

        points = np.asarray(data["point_yx"][start:end], dtype=np.float32)
        mags = np.asarray(data["point_magnitude"][start:end], dtype=np.float32)
        false_mask = np.asarray(data["point_is_false_star"][start:end], dtype=bool)
        point_star_id = np.asarray(data["point_star_id"][start:end], dtype=np.int32)
        guide_star_idx = int(data["guide_star_index"][local_scene_idx])
        roll_deg = int(data["roll_degree"][local_scene_idx])
        false_count = int(data["scene_false_stars_count"][local_scene_idx])
        real_count = int(data["scene_real_star_count"][local_scene_idx])
        total_count = int(data["scene_total_point_count"][local_scene_idx])
        dropout_count = int(data["scene_dropout_count"][local_scene_idx])
        pre_dropout_real_count = int(data["pre_dropout_real_star_count"][local_scene_idx])

    width = int((manifest.get("run") or {}).get("parameters", {}).get("resolution", [1280, 960])[0])
    height = int((manifest.get("run") or {}).get("parameters", {}).get("resolution", [1280, 960])[1])
    image = Image.new("RGB", (width, height + 72), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 12)
        font_small = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    min_mag = float(np.min(mags)) if mags.size else 0.0
    max_mag = float(np.max(mags)) if mags.size else 1.0
    mag_span = max(max_mag - min_mag, 1e-6)

    # Header
    draw.rectangle((0, 0, width, 72), fill=(12, 12, 12))
    h1 = (
        f"best_scene_idx={int(scene_idx)} chunk={chunk.path.name} local_scene={local_scene_idx} "
        f"guide={guide_star_idx} roll={roll_deg}"
    )
    h2 = (
        f"overlap={summary['overlap_count']} ang_sep={summary['angular_sep_deg']:.3f}deg "
        f"roll_diff={summary['roll_diff_deg']:.3f}deg real={real_count} false={false_count} "
        f"total={total_count} dropout={dropout_count}/{pre_dropout_real_count}"
    )
    draw.text((10, 10), h1, fill=(235, 235, 235), font=font)
    draw.text((10, 34), h2, fill=(255, 196, 96), font=font_small)

    for idx in np.argsort(mags)[::-1]:
        y = float(points[idx, 0]) + 72.0
        x = float(points[idx, 1])
        is_false = bool(false_mask[idx])
        star_id = int(point_star_id[idx])
        norm = (max_mag - float(mags[idx])) / mag_span if mags.size else 0.0
        radius = float(np.clip(1.0 + norm * 2.8, 1.0, 4.2))

        if is_false or star_id < 0:
            color = (180, 70, 70)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=1)
            continue

        catalog_id = int(catalog_lookup[star_id])
        is_overlap = catalog_id in overlap_catalog_ids
        fill = (255, 220, 120) if is_overlap else (250, 250, 250)
        outline = (60, 255, 60) if is_overlap else (180, 180, 180)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=1)

        label = str(catalog_id)
        tx = min(max(2.0, x + 5.0), float(width - 40))
        ty = min(max(74.0, y + 5.0), float(height + 72 - 15))
        draw.text(
            (tx, ty),
            label,
            fill=(255, 255, 210) if is_overlap else (220, 220, 220),
            font=font_small,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and render the dataset scene that best matches a tetra solution."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--solution-txt", type=Path, required=True)
    parser.add_argument("--stars-txt", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    solution_txt = args.solution_txt.expanduser().resolve()
    stars_txt = args.stars_txt.expanduser().resolve()
    database_path = args.database.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (dataset_dir / "best_scene_match").resolve()
    )

    tetra = parse_solution_txt(solution_txt)
    matched_catalog_ids = parse_matched_catalog_ids(stars_txt)
    best = best_scene_by_overlap(
        run_dir=dataset_dir,
        matched_catalog_ids=matched_catalog_ids,
        tetra_ra_deg=float(tetra["ra_deg"]),
        tetra_dec_deg=float(tetra["dec_deg"]),
        tetra_roll_deg=float(tetra["roll_deg"]),
        database_path=database_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "best_scene_match.json"
    out_txt = output_dir / "best_scene_match.txt"
    out_png = output_dir / "best_scene_match_with_catalog_ids.png"

    out_json.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    out_txt.write_text(
        "\n".join(
            [
                f"scene_idx: {best['scene_idx']}",
                f"scene_center_ra_deg: {best['scene_center_ra_deg']}",
                f"scene_center_dec_deg: {best['scene_center_dec_deg']}",
                f"scene_roll_degree: {best['scene_roll_degree']}",
                f"overlap_count: {best['overlap_count']}",
                f"angular_sep_deg: {best['angular_sep_deg']}",
                f"roll_diff_deg: {best['roll_diff_deg']}",
                f"overlap_catalog_ids: {best['overlap_catalog_ids']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    render_scene_with_catalog_ids(
        run_dir=dataset_dir,
        scene_idx=int(best["scene_idx"]),
        overlap_catalog_ids=set(int(x) for x in best["overlap_catalog_ids"]),
        database_path=database_path,
        output_png=out_png,
        summary=best,
    )

    log(f"best_scene_idx={best['scene_idx']}")
    log(f"overlap_count={best['overlap_count']}")
    log(f"angular_sep_deg={best['angular_sep_deg']:.3f}")
    log(f"roll_diff_deg={best['roll_diff_deg']:.3f}")
    log(f"overlap_catalog_ids={best['overlap_catalog_ids']}")
    log(f"written_json={out_json}")
    log(f"written_png={out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
