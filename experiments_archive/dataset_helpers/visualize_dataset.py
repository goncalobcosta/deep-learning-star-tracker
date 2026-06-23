#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 960
HEADER_HEIGHT = 58


@dataclass(frozen=True)
class ChunkInfo:
    path: Path
    scene_count: int
    scene_offset: int


def _latest_run_dir(runs_root: Path) -> Path:
    run_dirs = [
        d
        for d in runs_root.iterdir()
        if d.is_dir() and d.name.startswith("run") and d.name[3:].isdigit()
    ]
    if not run_dirs:
        raise RuntimeError(f"No run directories found in: {runs_root}")
    return max(run_dirs, key=lambda d: int(d.name[3:]))


def _resolve_run_dir(args: argparse.Namespace, script_dir: Path) -> Path:
    runs_root = (script_dir / "runs").resolve()
    if args.run_dir is not None:
        return args.run_dir.expanduser().resolve()
    if args.dataset_run is not None:
        return (runs_root / args.dataset_run).resolve()
    return _latest_run_dir(runs_root)


def _load_chunks(run_dir: Path) -> list[ChunkInfo]:
    manifest_path = run_dir / "dataset_manifest.json"
    chunks: list[ChunkInfo] = []
    global_offset = 0

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_chunks = manifest.get("chunks", [])
        for item in manifest_chunks:
            rel = item.get("file")
            scene_count = int(item.get("scene_count", 0))
            if not rel or scene_count <= 0:
                continue
            chunk_path = (run_dir / str(rel)).resolve()
            if not chunk_path.exists():
                continue
            chunks.append(
                ChunkInfo(
                    path=chunk_path,
                    scene_count=scene_count,
                    scene_offset=global_offset,
                )
            )
            global_offset += scene_count
        if chunks:
            return chunks

    dataset_dir = run_dir / "dataset"
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {dataset_dir}")

    for chunk_path in sorted(dataset_dir.glob("dataset*.npz")):
        with np.load(chunk_path, allow_pickle=False) as data:
            scene_count = int(data["scene_point_count"].shape[0])
        chunks.append(
            ChunkInfo(
                path=chunk_path.resolve(),
                scene_count=scene_count,
                scene_offset=global_offset,
            )
        )
        global_offset += scene_count

    if not chunks:
        raise RuntimeError(f"No dataset chunk files found in: {dataset_dir}")
    return chunks


def _find_chunk_for_scene(chunks: list[ChunkInfo], global_scene_idx: int) -> tuple[ChunkInfo, int]:
    if global_scene_idx < 0:
        raise ValueError("scene index must be >= 0")
    for chunk in chunks:
        if global_scene_idx < (chunk.scene_offset + chunk.scene_count):
            return chunk, (global_scene_idx - chunk.scene_offset)
    raise IndexError(f"Scene index {global_scene_idx} is out of range")


def _render_scene(
    *,
    points_yx: np.ndarray,
    magnitudes: np.ndarray,
    is_false_star: np.ndarray,
    dropout_count: int,
    width: int,
    height: int,
    false_color: bool,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    if points_yx.shape[0] == 0:
        return img

    order = np.argsort(magnitudes)[::-1]
    points = points_yx[order]
    mags = magnitudes[order]
    false_mask = is_false_star[order]
    dropout_proxy_mask = np.zeros(points.shape[0], dtype=bool)
    if dropout_count > 0:
        # Dropped stars are not stored as points in the dataset.
        # For visualization only, mark an equivalent number of kept real stars.
        real_idx = np.flatnonzero(~false_mask)
        if real_idx.size > 0:
            n_drop = min(int(dropout_count), int(real_idx.size))
            dropout_proxy_mask[real_idx[:n_drop]] = True

    min_mag = float(np.min(mags))
    max_mag = float(np.max(mags))
    mag_span = max(max_mag - min_mag, 1e-6)

    for idx in range(points.shape[0]):
        y = float(points[idx, 0])
        x = float(points[idx, 1])
        mag = float(mags[idx])
        is_false = bool(false_mask[idx])

        norm = (max_mag - mag) / mag_span
        brightness = int(np.clip(110 + norm * 145, 0, 255))
        radius = float(np.clip(1.0 + norm * 2.6, 1.0, 4.0))

        if false_color and is_false:
            color = (35, min(255, brightness + 10), 35)  # false stars -> green
        elif dropout_proxy_mask[idx]:
            color = (min(255, brightness + 10), 35, 35)  # dropped real stars proxy -> red
        else:
            color = (brightness, brightness, brightness)  # kept real stars -> white

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
        )

    return img


def _draw_header(
    image: Image.Image,
    *,
    chunk_name: str,
    global_scene_idx: int,
    local_scene_idx: int,
    guide_star_idx: int,
    roll_deg: int,
    real_count: int,
    false_count: int,
    total_count: int,
    dropout_count: int,
    pre_dropout_real_count: int,
) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    w, _ = image.size

    draw.rectangle((0, 0, w, HEADER_HEIGHT), fill=(12, 12, 12))
    draw.line((0, HEADER_HEIGHT - 1, w, HEADER_HEIGHT - 1), fill=(55, 55, 55), width=1)

    line1 = (
        f"chunk={chunk_name} scene_global={global_scene_idx} scene_local={local_scene_idx} "
        f"guide={guide_star_idx} roll={roll_deg}deg"
    )
    line2 = (
        f"real={real_count} false={false_count} total={total_count} "
        f"dropout={dropout_count} pre_dropout_real={pre_dropout_real_count}"
    )
    draw.text((10, 10), line1, fill=(230, 230, 230), font=font)
    draw.text((10, 31), line2, fill=(255, 188, 96), font=font)


def _scene_indices(args: argparse.Namespace, total_scenes: int) -> list[int]:
    if args.scene_index:
        return [int(i) for i in args.scene_index]
    rng = np.random.default_rng(args.seed)
    count = max(int(args.num_random), 1)
    return [int(x) for x in rng.integers(0, total_scenes, size=count)]


def _scene_indices_by_guide(
    *,
    chunks: list[ChunkInfo],
    guide_star_indices: list[int],
    roll_degree: int | None,
    seed: int | None,
) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []

    for guide in guide_star_indices:
        candidates: list[int] = []
        for chunk in chunks:
            with np.load(chunk.path, allow_pickle=False) as data:
                guides = np.asarray(data["guide_star_index"], dtype=np.int64)
                if roll_degree is None:
                    local = np.flatnonzero(guides == int(guide))
                else:
                    rolls = np.asarray(data["roll_degree"], dtype=np.int64)
                    local = np.flatnonzero(
                        (guides == int(guide)) & (rolls == int(roll_degree))
                    )

            if local.size > 0:
                candidates.extend((local + chunk.scene_offset).astype(int).tolist())

        if not candidates:
            if roll_degree is None:
                raise RuntimeError(f"No scenes found for guide_star_index={guide}")
            raise RuntimeError(
                f"No scenes found for guide_star_index={guide} with roll_degree={roll_degree}"
            )

        chosen = int(candidates[int(rng.integers(0, len(candidates)))])
        selected.append(chosen)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export synthetic dataset scenes to PNG images."
    )
    parser.add_argument("--dataset-run", type=str, default=None, help="Run name, e.g. run6")
    parser.add_argument("--run-dir", type=Path, default=None, help="Absolute/relative path to a run dir")
    parser.add_argument(
        "--scene-index",
        type=int,
        nargs="+",
        default=None,
        help="Global scene indices across all chunks",
    )
    parser.add_argument(
        "--guide-star-index",
        type=int,
        nargs="+",
        default=None,
        help="Export one scene per guide star index (sampled from its 360 scenes).",
    )
    parser.add_argument(
        "--roll-degree",
        type=int,
        default=None,
        help="When used with --guide-star-index, filter by this roll [0..359].",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=2,
        help="How many random scenes to export when --scene-index is not set",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for random scene sampling")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: <run_dir>/preview_images)",
    )
    parser.add_argument(
        "--no-false-color",
        action="store_true",
        help="Render false stars in white instead of reddish color",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    run_dir = _resolve_run_dir(args, script_dir)
    chunks = _load_chunks(run_dir)
    total_scenes = sum(c.scene_count for c in chunks)

    if total_scenes <= 0:
        raise RuntimeError(f"No scenes found in run: {run_dir}")

    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / "preview_images").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.guide_star_index:
        target_scene_indices = _scene_indices_by_guide(
            chunks=chunks,
            guide_star_indices=[int(x) for x in args.guide_star_index],
            roll_degree=None if args.roll_degree is None else int(args.roll_degree),
            seed=args.seed,
        )
    else:
        target_scene_indices = _scene_indices(args, total_scenes)
    print(f"Run directory: {run_dir}")
    print(f"Total scenes: {total_scenes}")
    print(f"Exporting {len(target_scene_indices)} scene(s) to: {out_dir}")

    for global_scene_idx in target_scene_indices:
        chunk, local_scene_idx = _find_chunk_for_scene(chunks, global_scene_idx)

        with np.load(chunk.path, allow_pickle=False) as data:
            start = int(data["scene_point_start"][local_scene_idx])
            count = int(data["scene_point_count"][local_scene_idx])
            end = start + count

            points = np.asarray(data["point_yx"][start:end], dtype=np.float32)
            mags = np.asarray(data["point_magnitude"][start:end], dtype=np.float32)
            false_mask = np.asarray(data["point_is_false_star"][start:end], dtype=bool)
            guide_star_idx = int(data["guide_star_index"][local_scene_idx])
            roll_deg = int(data["roll_degree"][local_scene_idx])
            false_count = int(data["scene_false_stars_count"][local_scene_idx])
            real_count = int(data["scene_real_star_count"][local_scene_idx])
            total_count = int(data["scene_total_point_count"][local_scene_idx])
            dropout_count = int(data["scene_dropout_count"][local_scene_idx])
            pre_dropout_real_count = int(data["pre_dropout_real_star_count"][local_scene_idx])

        image = _render_scene(
            points_yx=points,
            magnitudes=mags,
            is_false_star=false_mask,
            dropout_count=dropout_count,
            width=int(args.width),
            height=int(args.height),
            false_color=not args.no_false_color,
        )
        _draw_header(
            image,
            chunk_name=chunk.path.name,
            global_scene_idx=global_scene_idx,
            local_scene_idx=local_scene_idx,
            guide_star_idx=guide_star_idx,
            roll_deg=roll_deg,
            real_count=real_count,
            false_count=false_count,
            total_count=total_count,
            dropout_count=dropout_count,
            pre_dropout_real_count=pre_dropout_real_count,
        )

        filename = (
            f"scene_{global_scene_idx:08d}"
            f"_guide{guide_star_idx:06d}"
            f"_roll{roll_deg:03d}.png"
        )
        out_path = out_dir / filename
        image.save(out_path)
        print(
            "Saved:",
            out_path,
            (
                f"(chunk={chunk.path.name}, local_scene={local_scene_idx}, "
                f"real={real_count}, false={false_count}, total={total_count}, "
                f"dropout={dropout_count}, pre_dropout_real={pre_dropout_real_count})"
            ),
        )


if __name__ == "__main__":
    main()
