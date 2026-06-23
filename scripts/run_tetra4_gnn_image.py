from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from tetra3.tetra4_GNN import Tetra3


DEFAULT_MODEL = "final_r3_synthd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tetra4_GNN on one image and report per-frame timings excluding GNN load."
    )
    parser.add_argument("image", type=Path, help="Path to the image to solve.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"GNN model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--device", default="auto", help="GNN device: auto, cpu, cuda, etc. Default: auto")
    parser.add_argument("--repeat", type=int, default=1, help="Number of measured repetitions. Default: 1")
    parser.add_argument("--gnn-topk", type=int, default=10, help="Top-k GNN candidates per centroid. Default: 10")
    parser.add_argument(
        "--gnn-anchor-stars",
        type=int,
        default=8,
        help="Number of GNN confidence-ranked centroids used as anchors. Default: 8",
    )
    parser.add_argument(
        "--gnn-pair-topk",
        type=int,
        default=10,
        help="Top-K identities used per anchor in pair search. Default: 10",
    )
    parser.add_argument(
        "--gnn-brightest-stars",
        type=int,
        default=8,
        help="Number of brightest centroids sent to the GNN. Default: 8",
    )
    parser.add_argument(
        "--gnn-verification-budget",
        type=int,
        default=65,
        help="Maximum complete geometric verifications. Default: 65",
    )
    parser.add_argument(
        "--distortion",
        type=float,
        nargs="*",
        default=[-0.2, 0.1],
        help="Distortion: omit values for None, one value for fixed k, two values for search range. Default: -0.2 0.1",
    )
    parser.add_argument("--matches", action="store_true", help="Return matched star data.")
    return parser.parse_args()


def parse_distortion(values: list[float]):
    if len(values) == 0:
        return None
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return [float(values[0]), float(values[1])]
    raise ValueError("--distortion expects 0, 1, or 2 values")


def ms(value) -> float:
    if value is None:
        return 0.0
    return float(value)


def no_load_timings(result: dict) -> dict[str, float | None]:
    t_load = ms(result.get("T_gnn_load"))
    t_solve_no_load = ms(result.get("T_solve")) - t_load
    t_identify = result.get("T_identify")
    return {
        "T_extract": result.get("T_extract"),
        "T_solve_no_gnn_load": t_solve_no_load,
        "T_gnn_inference": result.get("T_gnn_inference"),
        "T_hypothesis_search": result.get("T_hypothesis_search"),
        "T_attitude": result.get("T_attitude"),
        "T_total_no_gnn_load": ms(result.get("T_extract")) + t_solve_no_load,
        "T_identify_no_gnn_load": None if t_identify is None else ms(t_identify) - t_load,
    }


def print_result(rep: int, result: dict) -> None:
    timings = no_load_timings(result)
    print(
        f"rep {rep:02d}: "
        f"{'OK' if result.get('RA') is not None else 'FAIL'} | "
        f"RA={result.get('RA')} Dec={result.get('Dec')} "
        f"matches={result.get('Matches')} rmse={result.get('RMSE')}"
    )
    print(
        "  tempos sem load GNN: "
        f"total={timings['T_total_no_gnn_load']:.2f} ms | "
        f"extract={ms(timings['T_extract']):.2f} ms | "
        f"solve={timings['T_solve_no_gnn_load']:.2f} ms | "
        f"identify={ms(timings['T_identify_no_gnn_load']):.2f} ms | "
        f"gnn_inference={ms(timings['T_gnn_inference']):.2f} ms | "
        f"hypothesis_search={ms(timings['T_hypothesis_search']):.2f} ms | "
        f"attitude={ms(timings['T_attitude']):.2f} ms"
    )


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    distortion = parse_distortion(args.distortion)

    print(f"image: {image_path}")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print("loading database + GNN once...")
    solver = Tetra3(gnn_model_name=args.model, gnn_device=args.device)
    solver._get_gnn_identifier(args.model, gnn_device=args.device)
    print("load done; timings below exclude GNN load.")

    for rep in range(1, max(1, int(args.repeat)) + 1):
        with Image.open(image_path) as img:
            result = solver.solve_from_image(
                img,
                distortion=distortion,
                gnn_topk=args.gnn_topk,
                gnn_anchor_stars=args.gnn_anchor_stars,
                gnn_pair_topk=args.gnn_pair_topk,
                gnn_brightest_stars=args.gnn_brightest_stars,
                gnn_verification_budget=args.gnn_verification_budget,
                return_matches=args.matches,
            )
        print_result(rep, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
