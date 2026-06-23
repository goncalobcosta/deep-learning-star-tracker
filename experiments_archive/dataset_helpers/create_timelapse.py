#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MP4 timelapses from sky_plots PNG sequences."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "Run names or paths. Examples: run5 run6 "
            "runs_tmp_validation/run7 tetra4/synth_dataset/runs_tmp_validation/run8"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output frames per second. Default: 30.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("tetra4/synth_dataset/runs_tmp_validation"),
        help="Base directory used when a run name like 'run5' is provided.",
    )
    parser.add_argument(
        "--sky-dir-name",
        default="sky_plots",
        help="Name of the folder containing the PNG sequence. Default: sky_plots.",
    )
    parser.add_argument(
        "--output-name",
        default="sky_plots_timelapse.mp4",
        help="Output video filename written inside each run directory.",
    )
    return parser.parse_args()


def resolve_run_dir(run_arg: str, base_dir: Path) -> Path:
    candidate = Path(run_arg)
    if candidate.exists():
        return candidate
    return base_dir / run_arg


def create_timelapse(
    run_dir: Path,
    sky_dir_name: str,
    output_name: str,
    fps: int,
) -> None:
    input_dir = run_dir / sky_dir_name
    output_file = run_dir / output_name

    images = sorted(input_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"Sem PNGs em {input_dir}")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Nao foi possivel ler a primeira imagem: {images[0]}")

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o video: {output_file}")

    written_frames = 0
    try:
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"A ignorar imagem ilegivel: {img_path}")
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written_frames += 1
    finally:
        writer.release()

    print(f"Video criado em: {output_file}")
    print(
        f"Frames: {written_frames} | FPS: {fps} | Duracao: {written_frames / fps:.1f}s"
    )


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps tem de ser maior que 0")

    for run_arg in args.runs:
        run_dir = resolve_run_dir(run_arg, args.base_dir)
        print(f"\nA processar: {run_dir}")
        create_timelapse(run_dir, args.sky_dir_name, args.output_name, args.fps)


if __name__ == "__main__":
    main()
