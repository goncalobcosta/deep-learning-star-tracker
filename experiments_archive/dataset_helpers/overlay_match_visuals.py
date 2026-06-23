from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay a solver visualization and a best-scene match visualization."
    )
    parser.add_argument("--solver-image", required=True, help="Path to solver visual PNG.")
    parser.add_argument(
        "--scene-image", required=True, help="Path to best-scene match PNG."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where overlay/comparison PNGs will be written.",
    )
    parser.add_argument(
        "--scene-crop-top",
        type=int,
        default=None,
        help="Pixels to crop from the top of the scene image. If omitted, a top crop is auto-inferred when possible.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Brightness threshold used to treat pixels as foreground.",
    )
    return parser.parse_args()


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def align_images(
    solver: Image.Image, scene: Image.Image, scene_crop_top: int | None
) -> tuple[Image.Image, Image.Image, int]:
    crop_top = 0
    if scene_crop_top is not None:
        crop_top = max(0, int(scene_crop_top))
    elif scene.width == solver.width and scene.height > solver.height:
        crop_top = scene.height - solver.height

    if crop_top > 0:
        scene = scene.crop((0, crop_top, scene.width, scene.height))

    if scene.size != solver.size:
        raise ValueError(
            f"Images do not align after cropping: solver={solver.size}, scene={scene.size}."
        )
    return solver, scene, crop_top


def tint_foreground(image: Image.Image, rgb: tuple[int, int, int], threshold: int) -> Image.Image:
    gray = image.convert("L")
    src_rgba = image.convert("RGBA")
    out = Image.new("RGBA", image.size, (0, 0, 0, 255))

    src_px = src_rgba.load()
    gray_px = gray.load()
    out_px = out.load()

    for y in range(image.height):
        for x in range(image.width):
            alpha = src_px[x, y][3]
            brightness = gray_px[x, y]
            if alpha == 0 or brightness <= threshold:
                continue
            scale = brightness / 255.0
            out_px[x, y] = (
                int(rgb[0] * scale),
                int(rgb[1] * scale),
                int(rgb[2] * scale),
                210,
            )
    return out


def build_overlay(solver: Image.Image, scene: Image.Image, threshold: int) -> Image.Image:
    canvas = Image.new("RGBA", solver.size, (0, 0, 0, 255))
    solver_tinted = tint_foreground(solver, (255, 90, 90), threshold)
    scene_tinted = tint_foreground(scene, (90, 220, 255), threshold)
    canvas = Image.alpha_composite(canvas, solver_tinted)
    canvas = Image.alpha_composite(canvas, scene_tinted)
    return canvas


def add_panel_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    padding = 6
    draw.rectangle(
        (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=(235, 235, 235), font=font)


def build_comparison_panel(
    solver: Image.Image, scene: Image.Image, overlay: Image.Image, crop_top: int
) -> Image.Image:
    gap = 24
    label_band = 44
    width = solver.width * 3 + gap * 4
    height = solver.height + label_band + gap * 2
    panel = Image.new("RGBA", (width, height), (10, 12, 20, 255))

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(panel)

    x_positions = [gap, gap * 2 + solver.width, gap * 3 + solver.width * 2]
    y = gap + label_band

    for x, image in zip(x_positions, [solver, scene, overlay]):
        panel.alpha_composite(image, (x, y))

    add_panel_label(draw, "Solver visual", x_positions[0] + 10, gap + 8, font)
    scene_label = "Best scene match"
    if crop_top > 0:
        scene_label += f" (crop top {crop_top}px)"
    add_panel_label(draw, scene_label, x_positions[1] + 10, gap + 8, font)
    add_panel_label(draw, "Overlay", x_positions[2] + 10, gap + 8, font)
    return panel


def main() -> int:
    args = parse_args()
    solver_path = Path(args.solver_image)
    scene_path = Path(args.scene_image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    solver = load_rgba(solver_path)
    scene = load_rgba(scene_path)
    solver, scene, crop_top = align_images(solver, scene, args.scene_crop_top)

    overlay = build_overlay(solver, scene, args.threshold)
    panel = build_comparison_panel(solver, scene, overlay, crop_top)

    overlay_path = output_dir / "best_scene_vs_solver_overlay.png"
    panel_path = output_dir / "best_scene_vs_solver_comparison.png"
    overlay.save(overlay_path)
    panel.save(panel_path)

    print(f"overlay_path={overlay_path}")
    print(f"comparison_path={panel_path}")
    print(f"scene_crop_top={crop_top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
