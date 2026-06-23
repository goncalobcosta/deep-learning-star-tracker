#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TETRA4_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = TETRA4_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tetra4.tetra3 import Tetra3, get_centroids_from_image

np.math = math  # Compatibility with newer NumPy versions.

OUT_ROOT = Path(__file__).resolve().parent
REAL_IMAGE = TETRA4_ROOT / "imgs_extras" / "imgs_teste" / "1000ms_18-50" / "1000ms_18-50-26-712529.tiff"
SCENE_IMAGE = (
    TETRA4_ROOT
    / "synth_dataset"
    / "runs"
    / "1000ms_18-50_subset_run1"
    / "run1"
    / "best_scene_match"
    / "preview_raw"
    / "scene_00057961_guide000062_roll119.png"
)
BEST_SCENE_JSON = (
    TETRA4_ROOT
    / "synth_dataset"
    / "runs"
    / "1000ms_18-50_subset_run1"
    / "run1"
    / "best_scene_match"
    / "best_scene_match.json"
)

SOLVE_PARAMS = {
    "distortion": 0,
    "return_matches": True,
    "return_visual": True,
}


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    return value


def circular_diff_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def angular_sep_deg(ra0: float | None, dec0: float | None, ra1: float | None, dec1: float | None) -> float | None:
    if ra0 is None or dec0 is None or ra1 is None or dec1 is None:
        return None
    ra0r = math.radians(float(ra0))
    dec0r = math.radians(float(dec0))
    ra1r = math.radians(float(ra1))
    dec1r = math.radians(float(dec1))
    cos_d = (
        math.sin(dec0r) * math.sin(dec1r)
        + math.cos(dec0r) * math.cos(dec1r) * math.cos(ra0r - ra1r)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_d))))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def get_solution_value(solution: dict[str, Any], key: str) -> float | int | None:
    value = solution.get(key)
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_catalog_ids(solution: dict[str, Any]) -> list[int]:
    raw = solution.get("matched_catID") or []
    ids: list[int] = []
    for item in raw:
        arr = np.asarray(item)
        if arr.ndim == 0:
            ids.append(int(arr.item()))
        else:
            ids.append(int(arr.reshape(-1)[0]))
    return ids


def load_centroid_count(image_path: Path) -> int:
    with Image.open(image_path) as img:
        centroids = get_centroids_from_image(img)
    return int(np.asarray(centroids).shape[0])


def solve_one(label: str, image_path: Path, out_dir: Path, solver: Tetra3) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = OUT_ROOT / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, inputs_dir / f"{label}{image_path.suffix.lower()}")

    with Image.open(image_path) as img:
        solution = solver.solve_from_image(img, **SOLVE_PARAMS)

    visual = solution.pop("visual", None)
    if visual is not None:
        visual_path = out_dir / "tetra3_visual.png"
        visual.save(visual_path)
    else:
        visual_path = None

    centroid_count = load_centroid_count(image_path)
    catalog_ids = get_catalog_ids(solution)
    solution_payload = {
        "label": label,
        "image": str(image_path.resolve()),
        "image_size": list(Image.open(image_path).size),
        "centroid_count": centroid_count,
        "solver_params": SOLVE_PARAMS,
        "solution": as_jsonable(solution),
        "visual_png": str(visual_path.resolve()) if visual_path else None,
    }

    (out_dir / "tetra3_solution.json").write_text(
        json.dumps(solution_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_solution_txt(out_dir / "tetra3_solution.txt", solution_payload)
    write_matched_stars_csv(out_dir / "tetra3_matched_stars.csv", solution)

    return {
        "label": label,
        "image": image_path,
        "out_dir": out_dir,
        "solution": solution,
        "centroid_count": centroid_count,
        "catalog_ids": catalog_ids,
        "visual_path": visual_path,
    }


def write_solution_txt(path: Path, payload: dict[str, Any]) -> None:
    sol = payload["solution"]
    lines = [
        f"label: {payload['label']}",
        f"image: {payload['image']}",
        f"image_size: {payload['image_size'][0]}x{payload['image_size'][1]}",
        f"centroid_count: {payload['centroid_count']}",
        "solver_params: distortion=0, return_matches=True, return_visual=True",
        f"ra_deg: {sol.get('RA')}",
        f"dec_deg: {sol.get('Dec')}",
        f"roll_deg: {sol.get('Roll')}",
        f"fov_horizontal_deg: {sol.get('FOV')}",
        f"distortion: {sol.get('distortion')}",
        f"rmse_arcsec: {sol.get('RMSE')}",
        f"matches: {sol.get('Matches')}",
        f"prob_false_positive: {sol.get('Prob')}",
        f"t_solve_ms: {sol.get('T_solve')}",
        f"t_extract_ms: {sol.get('T_extract')}",
        f"visual_png: {payload['visual_png']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matched_stars_csv(path: Path, solution: dict[str, Any]) -> None:
    cat_ids = get_catalog_ids(solution)
    centroids = np.asarray(solution.get("matched_centroids") or [], dtype=np.float64)
    stars = np.asarray(solution.get("matched_stars") or [], dtype=np.float64)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "match_index",
                "catalog_id",
                "catalog_ra_deg",
                "catalog_dec_deg",
                "catalog_mag",
                "centroid_y",
                "centroid_x",
            ]
        )
        for i, catalog_id in enumerate(cat_ids):
            star = stars[i] if i < len(stars) else [None, None, None]
            centroid = centroids[i] if i < len(centroids) else [None, None]
            writer.writerow(
                [
                    i,
                    catalog_id,
                    star[0],
                    star[1],
                    star[2],
                    centroid[0],
                    centroid[1],
                ]
            )


def write_overlap_csv(path: Path, real: dict[str, Any], scene: dict[str, Any]) -> None:
    real_ids = real["catalog_ids"]
    scene_ids = scene["catalog_ids"]
    real_centroids = np.asarray(real["solution"].get("matched_centroids") or [], dtype=np.float64)
    scene_centroids = np.asarray(scene["solution"].get("matched_centroids") or [], dtype=np.float64)

    real_lookup = {catalog_id: real_centroids[i].tolist() for i, catalog_id in enumerate(real_ids)}
    scene_lookup = {catalog_id: scene_centroids[i].tolist() for i, catalog_id in enumerate(scene_ids)}
    all_ids = sorted(set(real_ids) | set(scene_ids))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "catalog_id",
                "in_real",
                "in_scene",
                "real_centroid_y",
                "real_centroid_x",
                "scene_centroid_y",
                "scene_centroid_x",
            ]
        )
        for catalog_id in all_ids:
            r = real_lookup.get(catalog_id, [None, None])
            s = scene_lookup.get(catalog_id, [None, None])
            writer.writerow([catalog_id, catalog_id in real_lookup, catalog_id in scene_lookup, r[0], r[1], s[0], s[1]])


def make_comparison(real: dict[str, Any], scene: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    real_sol = real["solution"]
    scene_sol = scene["solution"]
    real_ids = set(real["catalog_ids"])
    scene_ids = set(scene["catalog_ids"])
    shared = sorted(real_ids & scene_ids)
    real_only = sorted(real_ids - scene_ids)
    scene_only = sorted(scene_ids - real_ids)

    metadata = json.loads(BEST_SCENE_JSON.read_text(encoding="utf-8")) if BEST_SCENE_JSON.exists() else {}
    comparison = {
        "inputs": {
            "real_image": str(real["image"].resolve()),
            "scene_image": str(scene["image"].resolve()),
            "best_scene_json": str(BEST_SCENE_JSON.resolve()) if BEST_SCENE_JSON.exists() else None,
        },
        "solver_params": SOLVE_PARAMS,
        "real": metrics_for(real),
        "scene": metrics_for(scene),
        "real_vs_scene": {
            "angular_sep_deg": angular_sep_deg(real_sol.get("RA"), real_sol.get("Dec"), scene_sol.get("RA"), scene_sol.get("Dec")),
            "roll_diff_deg": circular_diff_deg(real_sol.get("Roll"), scene_sol.get("Roll")),
            "fov_horizontal_diff_deg": (
                maybe_float(scene_sol.get("FOV")) - maybe_float(real_sol.get("FOV"))
                if scene_sol.get("FOV") is not None and real_sol.get("FOV") is not None
                else None
            ),
            "match_count_diff": (
                int(scene_sol.get("Matches")) - int(real_sol.get("Matches"))
                if scene_sol.get("Matches") is not None and real_sol.get("Matches") is not None
                else None
            ),
            "shared_catalog_id_count": len(shared),
            "real_only_catalog_id_count": len(real_only),
            "scene_only_catalog_id_count": len(scene_only),
            "shared_catalog_ids": shared,
            "real_only_catalog_ids": real_only,
            "scene_only_catalog_ids": scene_only,
        },
        "scene_metadata_comparison": scene_metadata_comparison(scene_sol, metadata),
    }

    (out_dir / "comparison_summary.json").write_text(
        json.dumps(as_jsonable(comparison), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_comparison_txt(out_dir / "comparison_summary.txt", comparison)
    write_overlap_csv(out_dir / "matched_catalog_overlap.csv", real, scene)
    make_board(real, scene, out_dir / "real_vs_scene_tetra3_board.png")
    return comparison


def metrics_for(result: dict[str, Any]) -> dict[str, Any]:
    sol = result["solution"]
    return {
        "centroid_count": result["centroid_count"],
        "ra_deg": get_solution_value(sol, "RA"),
        "dec_deg": get_solution_value(sol, "Dec"),
        "roll_deg": get_solution_value(sol, "Roll"),
        "fov_horizontal_deg": get_solution_value(sol, "FOV"),
        "distortion": get_solution_value(sol, "distortion"),
        "rmse_arcsec": get_solution_value(sol, "RMSE"),
        "matches": get_solution_value(sol, "Matches"),
        "prob_false_positive": get_solution_value(sol, "Prob"),
        "t_solve_ms": get_solution_value(sol, "T_solve"),
        "t_extract_ms": get_solution_value(sol, "T_extract"),
    }


def scene_metadata_comparison(scene_solution: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    scene_ra = metadata.get("scene_center_ra_deg")
    scene_dec = metadata.get("scene_center_dec_deg")
    scene_roll = metadata.get("scene_roll_degree")
    return {
        "scene_idx": metadata.get("scene_idx"),
        "metadata_ra_deg": scene_ra,
        "metadata_dec_deg": scene_dec,
        "metadata_roll_deg": scene_roll,
        "tetra3_vs_metadata_angular_sep_deg": angular_sep_deg(
            scene_solution.get("RA"), scene_solution.get("Dec"), scene_ra, scene_dec
        ),
        "tetra3_vs_metadata_roll_diff_deg": circular_diff_deg(scene_solution.get("Roll"), scene_roll),
        "metadata_overlap_count_from_previous_real_match": metadata.get("overlap_count"),
        "metadata_overlap_catalog_ids_from_previous_real_match": metadata.get("overlap_catalog_ids"),
    }


def write_comparison_txt(path: Path, comparison: dict[str, Any]) -> None:
    real = comparison["real"]
    scene = comparison["scene"]
    rel = comparison["real_vs_scene"]
    meta = comparison["scene_metadata_comparison"]
    lines = [
        "Tetra3 real image vs simulated scene",
        "",
        "Solver params: distortion=0, return_matches=True, return_visual=True",
        "",
        "Real image:",
        f"  RA={real['ra_deg']} Dec={real['dec_deg']} Roll={real['roll_deg']}",
        f"  FOV={real['fov_horizontal_deg']} distortion={real['distortion']}",
        f"  Matches={real['matches']} RMSE={real['rmse_arcsec']} arcsec Prob={real['prob_false_positive']}",
        "",
        "Simulated scene:",
        f"  RA={scene['ra_deg']} Dec={scene['dec_deg']} Roll={scene['roll_deg']}",
        f"  FOV={scene['fov_horizontal_deg']} distortion={scene['distortion']}",
        f"  Matches={scene['matches']} RMSE={scene['rmse_arcsec']} arcsec Prob={scene['prob_false_positive']}",
        "",
        "Real vs scene:",
        f"  Angular separation={rel['angular_sep_deg']} deg",
        f"  Roll difference={rel['roll_diff_deg']} deg",
        f"  FOV horizontal difference={rel['fov_horizontal_diff_deg']} deg",
        f"  Shared catalog IDs={rel['shared_catalog_id_count']}",
        f"  Real-only catalog IDs={rel['real_only_catalog_id_count']}",
        f"  Scene-only catalog IDs={rel['scene_only_catalog_id_count']}",
    ]
    if meta:
        lines.extend(
            [
                "",
                "Scene Tetra3 vs scene metadata:",
                f"  scene_idx={meta['scene_idx']}",
                f"  Angular separation={meta['tetra3_vs_metadata_angular_sep_deg']} deg",
                f"  Roll difference={meta['tetra3_vs_metadata_roll_diff_deg']} deg",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_panel_image(path: Path, max_size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", max_size, (12, 14, 18))
    x = (max_size[0] - img.width) // 2
    y = (max_size[1] - img.height) // 2
    panel.paste(img, (x, y))
    return panel


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, max_width: int, fnt: ImageFont.ImageFont) -> int:
    x, y = xy
    line = ""
    line_height = 18
    for word in text.split():
        candidate = word if not line else f"{line} {word}"
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            line = candidate
        else:
            draw.text((x, y), line, fill=(232, 235, 240), font=fnt)
            y += line_height
            line = word
    if line:
        draw.text((x, y), line, fill=(232, 235, 240), font=fnt)
        y += line_height
    return y


def make_board(real: dict[str, Any], scene: dict[str, Any], output_path: Path) -> None:
    panel_size = (640, 480)
    header_h = 86
    gap = 16
    margin = 18
    width = panel_size[0] * 2 + margin * 2 + gap
    height = header_h * 2 + panel_size[1] * 2 + margin * 3 + gap
    board = Image.new("RGB", (width, height), (6, 8, 11))
    draw = ImageDraw.Draw(board)
    title_font = font(18)
    small_font = font(13)

    panels = [
        ("Real image", real["image"], real),
        ("Simulated scene", scene["image"], scene),
        ("Real tetra3 visual", real["visual_path"], real),
        ("Scene tetra3 visual", scene["visual_path"], scene),
    ]
    positions = [
        (margin, margin),
        (margin + panel_size[0] + gap, margin),
        (margin, margin + header_h + panel_size[1] + gap),
        (margin + panel_size[0] + gap, margin + header_h + panel_size[1] + gap),
    ]

    for (title, image_path, result), (x, y) in zip(panels, positions):
        sol = result["solution"]
        draw.text((x, y), title, fill=(255, 255, 255), font=title_font)
        summary = (
            f"RA {sol.get('RA'):.6f} Dec {sol.get('Dec'):.6f} Roll {sol.get('Roll'):.3f} "
            f"FOV {sol.get('FOV'):.3f} Matches {sol.get('Matches')} RMSE {sol.get('RMSE'):.2f} arcsec"
        )
        draw_wrapped_text(draw, (x, y + 24), summary, panel_size[0], small_font)
        panel = load_panel_image(Path(image_path), panel_size)
        board.paste(panel, (x, y + header_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)


def main() -> int:
    for required in (REAL_IMAGE, SCENE_IMAGE):
        if not required.exists():
            raise FileNotFoundError(required)

    solver = Tetra3()
    real = solve_one("real_image", REAL_IMAGE, OUT_ROOT / "real_image", solver)
    scene = solve_one("simulated_scene", SCENE_IMAGE, OUT_ROOT / "simulated_scene", solver)
    comparison = make_comparison(real, scene, OUT_ROOT / "comparison")

    readme = [
        "# Tetra3 Real Image vs Simulated Scene",
        "",
        "This folder contains a fresh Tetra3 run for the Jose Lino real image and the selected simulated scene.",
        "",
        "- Solver settings: `distortion=0`, `return_matches=True`, `return_visual=True`.",
        "- Main comparison: `comparison/comparison_summary.txt` and `comparison/comparison_summary.json`.",
        "- Side-by-side board: `comparison/real_vs_scene_tetra3_board.png`.",
        "- Matched catalog overlap: `comparison/matched_catalog_overlap.csv`.",
        "",
        f"Shared catalog IDs: {comparison['real_vs_scene']['shared_catalog_id_count']}",
    ]
    (OUT_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"Wrote validation outputs to: {OUT_ROOT}")
    print(f"Real matches: {comparison['real']['matches']}")
    print(f"Scene matches: {comparison['scene']['matches']}")
    print(f"Shared catalog IDs: {comparison['real_vs_scene']['shared_catalog_id_count']}")
    print(f"Angular separation deg: {comparison['real_vs_scene']['angular_sep_deg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
