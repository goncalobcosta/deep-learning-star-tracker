from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tetra4.synth_dataset.find_best_scene_match import (
    DEFAULT_DB,
    find_chunk_for_scene,
    load_catalog_lookup,
    load_chunks,
    load_manifest,
)
from tetra4.tetra3 import get_centroids_from_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a semantic overlay between the real-image tetra result and the best matching synthetic scene."
        )
    )
    parser.add_argument("--image", type=Path, required=True, help="Real TIFF image.")
    parser.add_argument(
        "--stars-txt",
        type=Path,
        required=True,
        help="Tetra matched-stars TXT with catalog IDs and centroid coordinates.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Subset dataset run directory.")
    parser.add_argument(
        "--best-scene-json",
        type=Path,
        required=True,
        help="best_scene_match.json generated for the dataset run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--match-tol-px",
        type=float,
        default=1.5,
        help="Tolerance for treating an extracted real centroid as tetra-matched.",
    )
    parser.add_argument(
        "--real-top-n",
        type=int,
        default=None,
        help="If set, keep only the top-N brightest extracted real centroids (tetra uses 30 in the default database).",
    )
    parser.add_argument(
        "--scene-top-n",
        type=int,
        default=None,
        help="If set, keep only the top-N brightest real stars from the synthetic scene.",
    )
    return parser.parse_args()


def load_star_rows(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    seen_header = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("columns:"):
            seen_header = True
            continue
        if not seen_header:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "star_index": int(parts[0]),
                "catalog_id": int(parts[1]),
                "instrumental_mag": float(parts[2]),
                "catalog_mag": float(parts[3]),
                "centroid_y": float(parts[4]),
                "centroid_x": float(parts[5]),
            }
        )
    if not rows:
        raise RuntimeError(f"No matched rows parsed from {path}")
    return rows


def load_real_extracted_centroids(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        centroids, _moments = get_centroids_from_image(img, return_moments=True)
    point_yx = np.asarray(centroids, dtype=np.float32)
    if point_yx.ndim != 2 or point_yx.shape[1] != 2:
        raise RuntimeError(f"Unexpected centroid array shape from {image_path}: {point_yx.shape}")
    return point_yx


def classify_real_centroids(
    extracted_yx: np.ndarray, matched_rows: list[dict[str, float | int]], tol_px: float
) -> tuple[np.ndarray, list[dict[str, float | int]], np.ndarray]:
    matched_yx = np.asarray(
        [[float(r["centroid_y"]), float(r["centroid_x"])] for r in matched_rows], dtype=np.float32
    )
    if matched_yx.size == 0:
        return extracted_yx, matched_rows, np.empty((0, 2), dtype=np.float32)

    diffs = extracted_yx[:, None, :] - matched_yx[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    is_matched = np.any(dists <= float(tol_px), axis=1)
    junk = extracted_yx[~is_matched]
    return extracted_yx, matched_rows, junk


def load_scene_points(
    dataset_dir: Path,
    scene_idx: int,
    database_path: Path,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    manifest = load_manifest(dataset_dir)
    chunks = load_chunks(dataset_dir, manifest)
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

    keep = (~false_mask) & (point_star_id >= 0)
    points = points[keep]
    mags = mags[keep]
    point_star_id = point_star_id[keep]
    total_real_count = int(point_star_id.shape[0])

    if top_n > 0 and point_star_id.shape[0] > top_n:
        # Smaller magnitude means brighter; keep only the brightest catalog stars,
        # mirroring how tetra keeps the top-N brightest stars in the FOV.
        order = np.argsort(mags, kind="stable")[:top_n]
        points = points[order]
        point_star_id = point_star_id[order]

    catalog_ids = np.asarray([int(catalog_lookup[int(star_id)]) for star_id in point_star_id], dtype=np.int64)
    return points, point_star_id, catalog_ids, total_real_count


def estimate_similarity_transform(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    if src_xy.shape != dst_xy.shape or src_xy.shape[0] < 2:
        raise RuntimeError("Need at least two shared points to estimate a 2D similarity transform.")

    src_mean = src_xy.mean(axis=0)
    dst_mean = dst_xy.mean(axis=0)
    src0 = src_xy - src_mean
    dst0 = dst_xy - dst_mean

    cov = src0.T @ dst0
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = u @ vt

    src_var = float(np.sum(src0 * src0))
    scale = float(np.sum(s) / max(src_var, 1e-12))
    t = dst_mean - scale * (src_mean @ r)
    return r, scale, t


def apply_similarity(points_xy: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
    return scale * (points_xy @ r) + t


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, color: tuple[int, int, int], width: int = 1) -> None:
    draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
    draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)


def draw_legend(draw: ImageDraw.ImageDraw, width: int, font: ImageFont.ImageFont, font_small: ImageFont.ImageFont) -> None:
    x0 = 12
    y0 = 12
    box_w = 405
    box_h = 134
    draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=10, fill=(8, 10, 16))
    draw.text((x0 + 12, y0 + 8), "Real vs synthetic aligned overlay", fill=(240, 240, 240), font=font)

    items = [
        ((245, 245, 245), "dot", "Real-image extracted centroid"),
        ((255, 80, 80), "ring", "Real centroid not matched by tetra (junk)"),
        ((70, 255, 110), "ring", "Tetra-identified star in real image"),
        ((70, 210, 255), "cross", "Star from best synthetic scene"),
        ((255, 220, 90), "cross", "Catalog ID present in both scene and tetra"),
    ]

    row_y = y0 + 34
    for color, kind, text in items:
        cx = x0 + 18
        cy = row_y + 8
        if kind == "dot":
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color, outline=color)
        elif kind == "ring":
            draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=color, width=2)
        else:
            draw_cross(draw, cx, cy, 6, color, width=2)
        draw.text((x0 + 36, row_y), text, fill=(228, 228, 228), font=font_small)
        row_y += 20


def make_overlay(
    *,
    image_size: tuple[int, int],
    real_centroids_yx: np.ndarray,
    junk_yx: np.ndarray,
    matched_rows: list[dict[str, float | int]],
    scene_points_yx: np.ndarray,
    scene_catalog_ids: np.ndarray,
    overlap_catalog_ids: set[int],
    r: np.ndarray,
    scale: float,
    t: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    width, height = image_size
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
        font_small = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw_legend(draw, width, font, font_small)

    # Real extracted centroids.
    for y, x in real_centroids_yx.tolist():
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(245, 245, 245), outline=(245, 245, 245))

    # Synthetic scene points, aligned to the real image coordinate system.
    scene_xy = np.stack([scene_points_yx[:, 1], scene_points_yx[:, 0]], axis=1)
    aligned_xy = apply_similarity(scene_xy, r, scale, t)

    overlap_lookup: dict[int, tuple[float, float]] = {}
    for (ax, ay), catalog_id in zip(aligned_xy.tolist(), scene_catalog_ids.tolist()):
        color = (255, 220, 90) if int(catalog_id) in overlap_catalog_ids else (70, 210, 255)
        draw_cross(draw, ax, ay, 4.5 if int(catalog_id) in overlap_catalog_ids else 3.0, color, width=2 if int(catalog_id) in overlap_catalog_ids else 1)
        if int(catalog_id) in overlap_catalog_ids:
            overlap_lookup[int(catalog_id)] = (float(ax), float(ay))

    # Junk real centroids.
    for y, x in junk_yx.tolist():
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=(255, 64, 64), width=2)

    # Tetra-matched real stars and correspondence to the synthetic overlap.
    residuals: list[float] = []
    for row in matched_rows:
        catalog_id = int(row["catalog_id"])
        x = float(row["centroid_x"])
        y = float(row["centroid_y"])

        if catalog_id in overlap_lookup:
            sx, sy = overlap_lookup[catalog_id]
            draw.line((sx, sy, x, y), fill=(255, 220, 90), width=1)
            residuals.append(float(math.hypot(sx - x, sy - y)))
            draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), outline=(255, 220, 90), width=1)

        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(70, 255, 110), width=2)
        label_x = min(max(2.0, x + 6.0), float(width - 46))
        label_y = min(max(2.0, y + 6.0), float(height - 16))
        draw.text(
            (label_x, label_y),
            str(catalog_id),
            fill=(220, 255, 210),
            font=font_small,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return {
        "overlay_png": str(output_path),
        "matched_real_count": int(len(matched_rows)),
        "junk_real_count": int(junk_yx.shape[0]),
        "synthetic_scene_real_count": int(scene_catalog_ids.shape[0]),
        "shared_catalog_id_count": int(len(overlap_lookup)),
        "alignment_rmse_px": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
    }


def main() -> int:
    args = parse_args()

    image_path = args.image.expanduser().resolve()
    stars_txt = args.stars_txt.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    best_scene_json = args.best_scene_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    database_path = args.database.expanduser().resolve()

    matched_rows = load_star_rows(stars_txt)
    real_extracted_yx = load_real_extracted_centroids(image_path)
    if args.real_top_n is not None and args.real_top_n > 0:
        real_extracted_yx = real_extracted_yx[: int(args.real_top_n)]
    _real_all, matched_rows, junk_yx = classify_real_centroids(
        real_extracted_yx, matched_rows, float(args.match_tol_px)
    )

    best = json.loads(best_scene_json.read_text(encoding="utf-8"))
    if args.scene_top_n is not None and args.scene_top_n > 0:
        top_n = int(args.scene_top_n)
    else:
        top_n = int(real_extracted_yx.shape[0])
    scene_points_yx, _scene_star_ids, scene_catalog_ids, scene_total_real_count = load_scene_points(
        dataset_dir, int(best["scene_idx"]), database_path, top_n=top_n
    )
    overlap_catalog_ids = set(int(x) for x in best.get("overlap_catalog_ids", []))

    matched_real_xy_by_id = {
        int(row["catalog_id"]): np.asarray([float(row["centroid_x"]), float(row["centroid_y"])], dtype=np.float64)
        for row in matched_rows
    }
    scene_xy_by_id = {
        int(catalog_id): np.asarray([float(point[1]), float(point[0])], dtype=np.float64)
        for point, catalog_id in zip(scene_points_yx.tolist(), scene_catalog_ids.tolist())
        if int(catalog_id) in overlap_catalog_ids
    }

    shared_ids = sorted(overlap_catalog_ids & set(matched_real_xy_by_id) & set(scene_xy_by_id))
    if len(shared_ids) < 2:
        raise RuntimeError("Not enough shared catalog IDs between real tetra solution and synthetic scene.")

    src_xy = np.asarray([scene_xy_by_id[cid] for cid in shared_ids], dtype=np.float64)
    dst_xy = np.asarray([matched_real_xy_by_id[cid] for cid in shared_ids], dtype=np.float64)
    r, scale, t = estimate_similarity_transform(src_xy, dst_xy)

    with Image.open(image_path) as img:
        image_size = img.size

    output_png = output_dir / "best_scene_vs_solver_semantic_overlay.png"
    summary = make_overlay(
        image_size=image_size,
        real_centroids_yx=real_extracted_yx,
        junk_yx=junk_yx,
        matched_rows=matched_rows,
        scene_points_yx=scene_points_yx,
        scene_catalog_ids=scene_catalog_ids,
        overlap_catalog_ids=overlap_catalog_ids,
        r=r,
        scale=scale,
        t=t,
        output_path=output_png,
    )

    summary.update(
        {
            "scene_idx": int(best["scene_idx"]),
            "scene_center_ra_deg": float(best["scene_center_ra_deg"]),
            "scene_center_dec_deg": float(best["scene_center_dec_deg"]),
            "scene_roll_degree": float(best["scene_roll_degree"]),
            "real_top_n_applied": None if args.real_top_n is None else int(args.real_top_n),
            "scene_top_n_applied": None if args.scene_top_n is None else int(args.scene_top_n),
            "real_centroid_count_used_as_top_n": int(top_n),
            "scene_total_real_star_count_before_top_n": int(scene_total_real_count),
            "scene_real_star_count_after_top_n": int(scene_catalog_ids.shape[0]),
            "transform_scale": float(scale),
            "transform_rotation_matrix": np.asarray(r, dtype=np.float64).tolist(),
            "transform_translation_xy": np.asarray(t, dtype=np.float64).tolist(),
            "shared_catalog_ids": shared_ids,
        }
    )
    summary_path = output_dir / "best_scene_vs_solver_semantic_overlay.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"overlay_png={output_png}")
    print(f"summary_json={summary_path}")
    print(f"shared_catalog_ids={len(shared_ids)}")
    print(f"alignment_rmse_px={summary.get('alignment_rmse_px')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
