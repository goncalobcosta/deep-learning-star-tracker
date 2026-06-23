from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tetra3
from tetra3 import tetra4_GNN

DEFAULT_IMAGE_ROOT = REPO_ROOT / "imgs_extras" / "imgs_teste"
DEFAULT_GNN_RUN = (
    REPO_ROOT
    / "GNN"
    / "runs"
    / "final_r3_synthd"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "real_image_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tetra3 classic vs tetra4_GNN for the real TIFF test folders."
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--gnn-run", type=Path, default=DEFAULT_GNN_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=None, help="Short label to include in the output CSV filename.")
    parser.add_argument("--device", default="auto", help="GNN device: auto, cpu, cuda, etc.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images.")
    parser.add_argument("--classic-timeout-ms", type=float, default=10000.0)
    parser.add_argument("--gnn-timeout-ms", type=float, default=10000.0)
    parser.add_argument("--gnn-topk", type=int, default=10)
    parser.add_argument("--gnn-brightest-stars", type=int, default=8)
    parser.add_argument("--gnn-anchor-stars", type=int, default=8)
    parser.add_argument("--gnn-pair-topk", type=int, default=10)
    parser.add_argument(
        "--gnn-verification-budget",
        type=int,
        default=65,
        help="Cap on complete GNN geometric verifications. Default: 65.",
    )
    parser.add_argument("--distortion", type=float, nargs=2, default=[-0.2, 0.1])
    return parser.parse_args()


def direct_tiffs(folder: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ]
    )


def collect_cases(image_root: Path) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []

    for name in ("img1_1000ms_18-50", "img2_with_hot_pixels", "img2_without_hot_pixels"):
        folder = image_root / name
        if folder.exists():
            for image in direct_tiffs(folder):
                cases.append((name, image))

    obs_root = image_root / "Observation_23_March_2026"
    if obs_root.exists():
        for folder in sorted(obs_root.iterdir()):
            if folder.is_dir() and folder.name.startswith("imgObs_"):
                for image in direct_tiffs(folder):
                    cases.append((f"Observation_23_March_2026/{folder.name}", image))

    return cases


def ok(result: dict[str, Any]) -> bool:
    return result.get("RA") is not None and result.get("Dec") is not None


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def diff_angle_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def get(result: dict[str, Any], key: str) -> Any:
    value = result.get(key)
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return value


def get_any(result: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = get(result, key)
        if value is not None:
            return value
    return None


def solve_one(
    solver: Any,
    image_path: Path,
    *,
    distortion: list[float],
    solve_timeout: float,
    is_gnn: bool,
    gnn_checkpoint: Path | None = None,
    device: str = "auto",
    gnn_topk: int = 10,
    gnn_brightest_stars: int = 8,
    gnn_anchor_stars: int = 8,
    gnn_pair_topk: int = 10,
    gnn_verification_budget: int | None = None,
) -> tuple[dict[str, Any], str | None, float]:
    start = perf_counter()
    try:
        with Image.open(image_path) as img:
            kwargs: dict[str, Any] = {
                "distortion": distortion,
                "return_matches": False,
                "return_visual": False,
                "solve_timeout": solve_timeout,
            }
            if is_gnn:
                kwargs.update(
                    {
                        "gnn_checkpoint": gnn_checkpoint,
                        "gnn_device": device,
                        "gnn_topk": gnn_topk,
                        "gnn_brightest_stars": gnn_brightest_stars,
                        "gnn_anchor_stars": gnn_anchor_stars,
                        "gnn_pair_topk": gnn_pair_topk,
                        "gnn_verification_budget": gnn_verification_budget,
                    }
                )
            result = solver.solve_from_image(img, **kwargs)
        if isinstance(result, tuple):
            result = result[0]
        return result, None, (perf_counter() - start) * 1000.0
    except Exception as exc:  # Keep the batch alive; record the failure.
        return {}, f"{type(exc).__name__}: {exc}", (perf_counter() - start) * 1000.0


def row_for(
    case_name: str,
    image_path: Path,
    classic: dict[str, Any],
    classic_error: str | None,
    classic_wall_ms: float,
    gnn: dict[str, Any],
    gnn_error: str | None,
    gnn_wall_ms: float,
) -> dict[str, Any]:
    classic_ra = fnum(classic.get("RA"))
    classic_dec = fnum(classic.get("Dec"))
    gnn_ra = fnum(gnn.get("RA"))
    gnn_dec = fnum(gnn.get("Dec"))
    classic_total = fnum(classic.get("T_extract")) or 0.0
    classic_total += fnum(classic.get("T_solve")) or 0.0
    gnn_total_no_load = (fnum(gnn.get("T_extract")) or 0.0) + (fnum(gnn.get("T_solve")) or 0.0)
    gnn_total_no_load -= fnum(gnn.get("T_gnn_load")) or 0.0

    return {
        "case": case_name,
        "image": str(image_path),
        "classic_ok": ok(classic),
        "gnn_ok": ok(gnn),
        "same_success": ok(classic) == ok(gnn),
        "classic_RA": classic_ra,
        "classic_Dec": classic_dec,
        "classic_Roll": get(classic, "Roll"),
        "classic_FOV": get(classic, "FOV"),
        "classic_distortion": get(classic, "distortion"),
        "classic_RMSE": get(classic, "RMSE"),
        "classic_Matches": get(classic, "Matches"),
        "classic_Prob": get(classic, "Prob"),
        "classic_T_extract_ms": get(classic, "T_extract"),
        "classic_T_solve_ms": get(classic, "T_solve"),
        "classic_T_total_ms": classic_total,
        "classic_wall_ms": classic_wall_ms,
        "classic_error": classic_error,
        "gnn_RA": gnn_ra,
        "gnn_Dec": gnn_dec,
        "gnn_Roll": get(gnn, "Roll"),
        "gnn_FOV": get(gnn, "FOV"),
        "gnn_distortion": get(gnn, "distortion"),
        "gnn_RMSE": get(gnn, "RMSE"),
        "gnn_Matches": get(gnn, "Matches"),
        "gnn_Prob": get(gnn, "Prob"),
        "gnn_model": get(gnn, "gnn_model"),
        "gnn_anchor_attempts": get(gnn, "gnn_anchor_attempts"),
        "gnn_anchor_hypotheses_max": get(gnn, "gnn_anchor_hypotheses_max"),
        "gnn_sep_rejections": get(gnn, "gnn_sep_rejections"),
        "gnn_min_sep_skips": get(gnn, "gnn_min_sep_skips"),
        "gnn_invalid_pair_skips": get(gnn, "gnn_invalid_pair_skips"),
        "gnn_geometric_verifications": get_any(
            gnn,
            "gnn_verification_tests_performed",
            "gnn_geometric_verifications",
        ),
        "gnn_verification_rejections": get(gnn, "gnn_verification_rejections"),
        "gnn_verification_budget": get(gnn, "gnn_verification_budget"),
        "gnn_budget_exhausted": get_any(gnn, "gnn_verification_budget_exhausted", "gnn_budget_exhausted"),
        "gnn_T_extract_ms": get(gnn, "T_extract"),
        "gnn_T_solve_ms": get(gnn, "T_solve"),
        "gnn_T_load_ms": get(gnn, "T_gnn_load"),
        "gnn_T_inference_ms": get(gnn, "T_gnn_inference"),
        "gnn_T_hypothesis_search_ms": get(gnn, "T_hypothesis_search"),
        "gnn_T_attitude_ms": get(gnn, "T_attitude"),
        "gnn_T_total_no_load_ms": gnn_total_no_load,
        "gnn_wall_ms": gnn_wall_ms,
        "gnn_error": gnn_error,
        "delta_RA_deg": diff_angle_deg(classic_ra, gnn_ra),
        "delta_Dec_deg": None if classic_dec is None or gnn_dec is None else abs(classic_dec - gnn_dec),
    }


def main() -> int:
    args = parse_args()
    image_root = args.image_root.resolve()
    gnn_run = args.gnn_run.resolve()
    checkpoint = gnn_run / "best_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {checkpoint}")

    cases = collect_cases(image_root)
    if args.limit is not None:
        cases = cases[: int(args.limit)]
    if not cases:
        raise RuntimeError(f"No direct TIFF test images found under {image_root}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_label = args.label or output_dir.parent.name or "run"
    safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(raw_label))
    output_csv = output_dir / f"tetra_vs_tetra4_gnn_{safe_label}_{stamp}.csv"

    print(f"images: {len(cases)}")
    print(f"gnn_checkpoint: {checkpoint}")
    print(f"output: {output_csv}")
    print("loading solvers...")
    classic_solver = tetra3.Tetra3()
    gnn_solver = tetra4_GNN.Tetra3(gnn_device=args.device)
    # Warm the GNN once so per-image timings can report the no-load path.
    gnn_solver._get_gnn_identifier(gnn_checkpoint=checkpoint, gnn_device=args.device)
    print("solvers ready")

    rows: list[dict[str, Any]] = []
    for idx, (case_name, image_path) in enumerate(cases, start=1):
        print(f"[{idx:03d}/{len(cases):03d}] {case_name} :: {image_path.name}")
        classic, classic_error, classic_wall = solve_one(
            classic_solver,
            image_path,
            distortion=list(args.distortion),
            solve_timeout=float(args.classic_timeout_ms),
            is_gnn=False,
        )
        gnn, gnn_error, gnn_wall = solve_one(
            gnn_solver,
            image_path,
            distortion=list(args.distortion),
            solve_timeout=float(args.gnn_timeout_ms),
            is_gnn=True,
            gnn_checkpoint=checkpoint,
            device=args.device,
            gnn_topk=int(args.gnn_topk),
            gnn_brightest_stars=int(args.gnn_brightest_stars),
            gnn_anchor_stars=int(args.gnn_anchor_stars),
            gnn_pair_topk=int(args.gnn_pair_topk),
            gnn_verification_budget=args.gnn_verification_budget,
        )
        row = row_for(case_name, image_path, classic, classic_error, classic_wall, gnn, gnn_error, gnn_wall)
        rows.append(row)
        print(
            "  classic="
            f"{'OK' if row['classic_ok'] else 'FAIL'} "
            f"gnn={'OK' if row['gnn_ok'] else 'FAIL'} "
            f"matches={row['classic_Matches']}/{row['gnn_Matches']} "
            f"t_ms={float(row['classic_T_total_ms']):.1f}/{float(row['gnn_T_total_no_load_ms']):.1f}"
        )

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    classic_ok_count = sum(1 for row in rows if row["classic_ok"])
    gnn_ok_count = sum(1 for row in rows if row["gnn_ok"])
    both_ok = sum(1 for row in rows if row["classic_ok"] and row["gnn_ok"])
    print()
    print(f"done: {output_csv}")
    print(f"classic OK: {classic_ok_count}/{len(rows)}")
    print(f"gnn OK: {gnn_ok_count}/{len(rows)}")
    print(f"both OK: {both_ok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
