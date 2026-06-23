#!/usr/bin/env python3
"""Generate learning-curve plots from GNN train_history.jsonl files.

The script intentionally uses Pillow instead of matplotlib because the current
local venv used for these plots has Pillow installed but not matplotlib.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - only hit in the wrong env
    raise SystemExit(
        "Pillow is required. Run with the repo venv, e.g.:\n"
        "  .venv/bin/python synth_dataset/help_scripts/plot_learning_curves.py"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "synth_dataset" / "plots" / "learning_curves"
DEFAULT_IMG1_DIR = DEFAULT_OUT_DIR / "data" / "img1_T0_l3_h128"
DEFAULT_IMG3_DIR = DEFAULT_OUT_DIR / "data" / "img3_T0_l3_h128"
DEFAULT_FULL_DIR = DEFAULT_OUT_DIR / "data" / "full_run5_T0_l3_h128"

W = 1600
H = 900

BG = (255, 255, 255)
TEXT = (18, 20, 23)
MUTED = (86, 96, 108)
AXIS = (35, 42, 50)
GRID = (222, 228, 235)
PANEL_BG = (250, 252, 255)
TRAIN = (28, 116, 185)
VAL = (231, 125, 42)
GREEN = (78, 153, 61)
TOP10 = (105, 116, 132)
SUBSET_TRAIN = (28, 116, 185)
SUBSET_VAL = (87, 164, 82)
FULL_TRAIN = (182, 73, 38)
FULL_VAL = (231, 125, 42)

SECONDARY_METRIC_LABELS = {
    "val_top1_real": "val_top1",
    "val_top5_real": "val_top5",
    "val_top10_real": "val_top10",
}


@dataclass(frozen=True)
class RunData:
    label: str
    history: list[dict]
    summary: dict
    raw_rows: int
    history_dir: Path


@dataclass(frozen=True)
class Panel:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = load_font(42, bold=True)
F_SUBTITLE = load_font(24)
F_AXIS = load_font(25, bold=True)
F_TICK = load_font(20)
F_LEGEND = load_font(23, bold=True)
F_SMALL = load_font(19)
F_PANEL = load_font(25, bold=True)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0])


def best_summary_metric(summary: dict, metric_name: str) -> float | None:
    if metric_name == "val_loss":
        value = summary.get("best_val_loss")
    elif metric_name == "val_top1_real":
        value = summary.get("best_val_top1_real")
    elif metric_name == "val_top5_real":
        value = summary.get("best_val_top5_real")
    elif metric_name == "val_top10_real":
        value = summary.get("best_val_top10_real", summary.get("best_val_top10"))
    else:
        value = summary.get("best_monitor_value")
    return None if value is None else float(value)


def history_metric(row: dict, metric_name: str) -> float:
    if metric_name in row:
        return float(row[metric_name])
    if metric_name == "val_top10_real" and "val_top10" in row:
        return float(row["val_top10"])
    raise KeyError(f"Missing metric {metric_name!r} in history row")


def load_run(history_dir: Path, label: str) -> RunData:
    history_path = history_dir / "train_history.jsonl"
    summary_path = history_dir / "train_summary.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    raw_history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    if not raw_history:
        raise ValueError(f"Empty history file: {history_path}")

    # Some runs have history appended twice after reruns. Keep the last row per
    # epoch so the curve never jumps backwards from epoch N to epoch 1.
    by_epoch: dict[int, dict] = {}
    for row in raw_history:
        by_epoch[int(row["epoch"])] = row
    history = [by_epoch[epoch] for epoch in sorted(by_epoch)]

    summary = json.loads(summary_path.read_text())
    return RunData(
        label=label,
        history=history,
        summary=summary,
        raw_rows=len(raw_history),
        history_dir=history_dir,
    )


def nice_range(values: Iterable[float]) -> tuple[float, float]:
    vals = list(values)
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return lo - 1.0, hi + 1.0
    padding = (hi - lo) * 0.2
    return max(0.0, lo - padding), hi + padding


def draw_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    tmp = Image.new("RGBA", (420, 60), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tmp)
    draw.text((0, 0), text, font=font, fill=fill)
    tmp = tmp.rotate(90, expand=True)
    image.paste(tmp, xy, tmp)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    panel: Panel,
    x_values: list[int],
    y_values: list[float],
) -> tuple:
    y_min, y_max = nice_range(y_values)
    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def x_map(epoch: int | float) -> float:
        return panel.left + ((float(epoch) - x_min) / (x_max - x_min)) * panel.width

    def y_map(loss: int | float) -> float:
        return panel.top + ((y_max - float(loss)) / (y_max - y_min)) * panel.height

    draw.rectangle((panel.left, panel.top, panel.right, panel.bottom), fill=PANEL_BG)

    for idx in range(6):
        y = y_min + (y_max - y_min) * idx / 5
        py = y_map(y)
        draw.line((panel.left, py, panel.right, py), fill=GRID, width=2)
        label = f"{y:.3f}" if y_max - y_min < 0.2 else f"{y:.2f}"
        draw.text(
            (panel.left - 20 - text_width(draw, label, F_TICK), py - 12),
            label,
            font=F_TICK,
            fill=MUTED,
        )

    tick_step = max(1, math.ceil((x_max - x_min) / 8))
    for epoch in range(int(math.ceil(x_min)), int(math.floor(x_max)) + 1, tick_step):
        px = x_map(epoch)
        draw.line((px, panel.top, px, panel.bottom), fill=GRID, width=1)
        label = str(epoch)
        draw.text((px - text_width(draw, label, F_TICK) / 2, panel.bottom + 18), label, font=F_TICK, fill=MUTED)

    draw.line((panel.left, panel.top, panel.left, panel.bottom), fill=AXIS, width=4)
    draw.line((panel.left, panel.bottom, panel.right, panel.bottom), fill=AXIS, width=4)
    return x_map, y_map, y_min, y_max


def draw_secondary_axis(
    draw: ImageDraw.ImageDraw,
    panel: Panel,
    values: list[float],
    *,
    color: tuple[int, int, int] = TOP10,
):
    y_min, y_max = nice_range(values)

    def y_map(value: int | float) -> float:
        return panel.top + ((y_max - float(value)) / (y_max - y_min)) * panel.height

    draw.line((panel.right, panel.top, panel.right, panel.bottom), fill=color, width=3)
    for idx in range(5):
        value = y_min + (y_max - y_min) * idx / 4
        py = y_map(value)
        draw.line((panel.right, py, panel.right + 8, py), fill=color, width=2)
        if y_max < 0.01 or (y_max - y_min) < 0.001:
            label = f"{value:.5f}"
        elif y_max < 0.1:
            label = f"{value:.4f}"
        else:
            label = f"{value:.3f}"
        draw.text((panel.right + 13, py - 11), label, font=F_TICK, fill=color)
    return y_map


def draw_series(
    draw: ImageDraw.ImageDraw,
    epochs: list[int],
    values: list[float],
    x_map,
    y_map,
    color: tuple[int, int, int],
    *,
    width: int = 6,
) -> None:
    points = [(x_map(epoch), y_map(value)) for epoch, value in zip(epochs, values)]
    if len(points) > 1:
        draw.line(points, fill=color, width=width)
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def draw_dashed_series(
    draw: ImageDraw.ImageDraw,
    epochs: list[int],
    values: list[float],
    x_map,
    y_map,
    color: tuple[int, int, int],
    *,
    width: int = 3,
    dash: int = 14,
    gap: int = 10,
) -> None:
    points = [(float(x_map(epoch)), float(y_map(value))) for epoch, value in zip(epochs, values)]
    for start, end in zip(points, points[1:]):
        x1, y1 = start
        x2, y2 = end
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist <= 0:
            continue
        step = dash + gap
        travelled = 0.0
        while travelled < dist:
            seg_start = travelled
            seg_end = min(travelled + dash, dist)
            sx = x1 + (x2 - x1) * (seg_start / dist)
            sy = y1 + (y2 - y1) * (seg_start / dist)
            ex = x1 + (x2 - x1) * (seg_end / dist)
            ey = y1 + (y2 - y1) * (seg_end / dist)
            draw.line((sx, sy, ex, ey), fill=color, width=width)
            travelled += step
    for x, y in points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    items: list[tuple[str, tuple[int, int, int]]],
    *,
    x: int,
    y: int,
    row_height: int = 38,
) -> None:
    for idx, (label, color) in enumerate(items):
        yy = y + idx * row_height
        draw.line((x, yy + 15, x + 62, yy + 15), fill=color, width=7)
        draw.text((x + 78, yy), label, font=F_LEGEND, fill=TEXT)


def draw_single_run(
    run: RunData,
    *,
    title: str,
    subtitle: str,
    out_file: Path,
    secondary_metric: str,
) -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    panel = Panel(left=150, top=225, right=1515, bottom=720)
    epochs = [int(row["epoch"]) for row in run.history]
    train_loss = [float(row["train_loss"]) for row in run.history]
    val_loss = [float(row["val_loss"]) for row in run.history]
    secondary_values = [history_metric(row, secondary_metric) for row in run.history]
    secondary_label = SECONDARY_METRIC_LABELS[secondary_metric]

    draw.text((78, 52), title, font=F_TITLE, fill=TEXT)
    draw.text((78, 112), subtitle, font=F_SUBTITLE, fill=MUTED)

    monitor_metric = str(run.summary.get("best_monitor_metric", "val_top10_real"))
    monitor_value = best_summary_metric(run.summary, monitor_metric)
    extra = [
        f"epochs={len(run.history)}",
        f"best_{monitor_metric}_epoch={run.summary.get('best_epoch', '-')}",
        f"best_{monitor_metric}={monitor_value:.6f}" if monitor_value is not None else f"best_{monitor_metric}=n/a",
        f"final_val_loss={val_loss[-1]:.3f}",
    ]
    if run.raw_rows != len(run.history):
        extra.append(f"histórico deduplicado: {run.raw_rows}->{len(run.history)}")
    draw.text((78, 148), " | ".join(extra), font=F_SMALL, fill=MUTED)

    x_map, y_map, _, _ = draw_axes(draw, panel, epochs, train_loss + val_loss)
    draw_series(draw, epochs, train_loss, x_map, y_map, TRAIN)
    draw_series(draw, epochs, val_loss, x_map, y_map, VAL)
    secondary_y_map = draw_secondary_axis(draw, panel, secondary_values)
    draw_dashed_series(draw, epochs, secondary_values, x_map, secondary_y_map, TOP10)

    best_epoch = run.summary.get("best_epoch")
    if best_epoch is not None:
        px = x_map(float(best_epoch))
        draw.line((px, panel.top, px, panel.bottom), fill=GREEN, width=3)
        draw.text((px + 10, panel.top + 10), f"best {monitor_metric} epoch {best_epoch}", font=F_SMALL, fill=GREEN)

    x_label = "Epoch"
    draw.text((panel.left + panel.width / 2 - text_width(draw, x_label, F_AXIS) / 2, 805), x_label, font=F_AXIS, fill=TEXT)
    draw_rotated_text(image, (42, 375), "CrossEntropy loss", F_AXIS, TEXT)
    draw_legend(
        draw,
        [("train_loss", TRAIN), ("val_loss", VAL), (f"{secondary_label} (eixo dir.)", TOP10)],
        x=1185,
        y=52,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)


def draw_comparison(subset: RunData, full: RunData, *, out_file: Path, secondary_metric: str) -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((78, 46), "Learning curve - TestID 0 baseline", font=F_TITLE, fill=TEXT)
    draw.text(
        (78, 106),
        "Comparação com escalas Y separadas: subset e full têm número de classes diferente",
        font=F_SUBTITLE,
        fill=MUTED,
    )

    panels = [
        (
            subset,
            Panel(left=150, top=210, right=1515, bottom=445),
            "subDataset img1 / Lino - 512 classes",
            SUBSET_TRAIN,
            SUBSET_VAL,
        ),
        (
            full,
            Panel(left=150, top=575, right=1515, bottom=810),
            "Full Dataset run5_expD_all - 8818 classes",
            FULL_TRAIN,
            FULL_VAL,
        ),
    ]

    for run, panel, label, train_color, val_color in panels:
        epochs = [int(row["epoch"]) for row in run.history]
        train_loss = [float(row["train_loss"]) for row in run.history]
        val_loss = [float(row["val_loss"]) for row in run.history]
        secondary_values = [history_metric(row, secondary_metric) for row in run.history]
        secondary_label = SECONDARY_METRIC_LABELS[secondary_metric]
        draw.text((panel.left, panel.top - 42), label, font=F_PANEL, fill=TEXT)
        x_map, y_map, _, _ = draw_axes(draw, panel, epochs, train_loss + val_loss)
        draw_series(draw, epochs, train_loss, x_map, y_map, train_color, width=5)
        draw_series(draw, epochs, val_loss, x_map, y_map, val_color, width=5)
        secondary_y_map = draw_secondary_axis(draw, panel, secondary_values)
        draw_dashed_series(draw, epochs, secondary_values, x_map, secondary_y_map, TOP10, width=2)
        draw.text(
            (panel.left, panel.bottom + 54),
            f"train: {train_loss[0]:.3f}->{train_loss[-1]:.3f} | val: {val_loss[0]:.3f}->{val_loss[-1]:.3f} | {secondary_label}: {secondary_values[0]:.4f}->{secondary_values[-1]:.4f}",
            font=F_SMALL,
            fill=MUTED,
        )

    draw_rotated_text(image, (42, 375), "CrossEntropy loss", F_AXIS, TEXT)
    draw.text((W // 2 - text_width(draw, "Epoch", F_AXIS) // 2, 852), "Epoch", font=F_AXIS, fill=TEXT)
    draw_legend(
        draw,
        [
            ("subset train", SUBSET_TRAIN),
            ("subset val", SUBSET_VAL),
            ("full train", FULL_TRAIN),
            ("full val", FULL_VAL),
            (f"{SECONDARY_METRIC_LABELS[secondary_metric]} (dir.)", TOP10),
        ],
        x=1165,
        y=46,
        row_height=34,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img1-dir", type=Path, default=DEFAULT_IMG1_DIR)
    parser.add_argument("--img3-dir", type=Path, default=DEFAULT_IMG3_DIR)
    parser.add_argument("--full-dir", type=Path, default=DEFAULT_FULL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--secondary-metric",
        choices=sorted(SECONDARY_METRIC_LABELS),
        default="val_top1_real",
        help="Validation metric to plot on the right axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img1 = load_run(args.img1_dir, "img1_T0_l3_h128")
    img3 = load_run(args.img3_dir, "img3_T0_l3_h128")
    full = load_run(args.full_dir, "full_run5_T0_l3_h128")

    single_img1 = args.out_dir / "learning_curve_T0_l3_h128_img1_subset.png"
    single_img3 = args.out_dir / "learning_curve_T0_l3_h128_img3_subset.png"
    single_full = args.out_dir / "learning_curve_T0_l3_h128_full_run5_expD.png"
    comparison = args.out_dir / "learning_curve_T0_l3_h128_subset_vs_full.png"

    draw_single_run(
        img1,
        title="Learning curve - TestID 0 baseline",
        subtitle="subDataset img1 / Lino - 512 estrelas | T0_l3_h128 | batch_size=512 | CrossEntropy",
        out_file=single_img1,
        secondary_metric=args.secondary_metric,
    )
    draw_single_run(
        img3,
        title="Learning curve - TestID 0 baseline",
        subtitle="subDataset img3 - 512 estrelas | T0_l3_h128 | batch_size=512 | CrossEntropy",
        out_file=single_img3,
        secondary_metric=args.secondary_metric,
    )
    draw_single_run(
        full,
        title="Learning curve - TestID 0 baseline",
        subtitle="Full Dataset run5_expD_all - 8818 estrelas | T0_l3_h128 | batch_size=2048 | CrossEntropy",
        out_file=single_full,
        secondary_metric=args.secondary_metric,
    )
    draw_comparison(img1, full, out_file=comparison, secondary_metric=args.secondary_metric)

    # Backwards-compatible aliases for the first version of these plots.
    aliases = {
        single_img1: args.out_dir / "T0_l3_h128_learning_curve_img1_subset.png",
        single_img3: args.out_dir / "T0_l3_h128_learning_curve_img3_subset.png",
        single_full: args.out_dir / "T0_l3_h128_learning_curve_full_run5_expD.png",
        comparison: args.out_dir / "T0_l3_h128_learning_curve_comparison.png",
    }
    for source, target in aliases.items():
        target.write_bytes(source.read_bytes())

    for path in [single_img1, single_img3, single_full, comparison, *aliases.values()]:
        print(path)


if __name__ == "__main__":
    main()
