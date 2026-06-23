#!/usr/bin/env python3
"""Minimal end-to-end GNN training pipeline for star identification.

Design goals:
- single-file training pipeline (paper-like message passing + node classification)
- uses synth_dataset run shards produced by synth_dataset/generate_dataset_baseline.py
  or synth_dataset/generate_dataset_aletorio.py
- closed-set split is by scene within each guide_star (via GNN/split/split.py)
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from datetime import datetime, timezone
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
import importlib.util

HIDDEN_DIM = 512
NUM_LAYERS = 5
HEADS = 4
DROPOUT = 0.2
REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_K_FORMULA = "min(max(3, round(0.25 * n)), 8)"
GRAPH_CONNECTIVITY_CHOICES = ("knn", "fully")
MODEL_BACKBONE_CHOICES = ("gatv2", "edge_mlp")
EARLY_STOP_MONITOR_CHOICES = (
    "val_loss",
    "val_top1_real",
    "val_top5_real",
    "val_top10_real",
)
NODE_FEATURE_MODE_CHOICES = (
    "magnitude_subtracted_rank",
    "none",
    "magnitude_rank",
    "magnitude_rank_1based",
    "magnitude",
    "magnitude_subtracted",
    "magnitude_norm_max",
    "magnitude_norm_median",
    "magnitude_subtracted_norm_max",
    "magnitude_subtracted_norm_median",
)
EDGE_FEATURE_MODE_CHOICES = (
    "distance_diagonal_dmag",
    "distance_diagonal_dmag_node",
    "distance_max",
    "distance_raw",
    "distance_diagonal",
    "distance_max_dmag",
    "distance_max_dmag_node",
    "distance_raw_dmag",
    "distance_raw_dmag_node",
)
QUAD_COMBINATION_MODE_CHOICES = ("all", "sample", "balanced_sample")


def ensure_pyg_available() -> None:
    if importlib.util.find_spec("torch_geometric") is not None:
        return
    raise ImportError(
        "torch_geometric is not installed. Install dependencies first, for example:\n"
        "  /usr/bin/python3 -m pip install torch-geometric\n"
        "and ensure compatible torch/torch-scatter/torch-sparse wheels are available."
    )


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lower_is_better_metric(metric_name: str) -> bool:
    return metric_name.endswith("_loss") or metric_name == "val_loss"


def best_history_value(
    history: Sequence[Dict[str, object]],
    metric_name: str,
) -> Tuple[int, float]:
    if not history:
        return 0, 0.0

    best_epoch = 0
    best_value = float("inf") if lower_is_better_metric(metric_name) else -float("inf")
    for row in history:
        if metric_name not in row:
            continue
        value = float(row[metric_name])
        improved = value < best_value if lower_is_better_metric(metric_name) else value > best_value
        if improved:
            best_value = value
            best_epoch = int(row["epoch"])

    if math.isinf(best_value):
        return 0, 0.0
    return best_epoch, float(best_value)


def choose_device(device_arg: str) -> torch.device:
    def mps_scatter_reduce_supported() -> bool:
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            return False
        try:
            src = torch.tensor([1.0, 2.0, 3.0, 4.0], device="mps")
            index = torch.tensor([0, 0, 1, 1], dtype=torch.long, device="mps")
            out = torch.zeros(2, device="mps")
            out.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
            torch.mps.synchronize()
            return True
        except Exception:
            return False

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # On macOS + PyG this training path is not stable on MPS; use CPU by default.
        if sys.platform != "darwin" and mps_scatter_reduce_supported():
            return torch.device("mps")
        return torch.device("cpu")

    requested = torch.device(device_arg)
    if requested.type == "mps" and not mps_scatter_reduce_supported():
        log("MPS requested but scatter_reduce is not supported for this setup; falling back to CPU.")
        return torch.device("cpu")
    return requested


def next_run_dir(root: Path, run_name: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if run_name:
        run_dir = (root / run_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    max_idx = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("run") and child.name[3:].isdigit():
            max_idx = max(max_idx, int(child.name[3:]))
    run_dir = root / f"run{max_idx + 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def resolve_dataset_dir(dataset_dir: Path | None, dataset_run: str | None) -> Path:
    if dataset_dir is not None:
        resolved = dataset_dir.expanduser().resolve()
    elif dataset_run:
        resolved = (Path("synth_dataset") / "runs" / dataset_run).resolve()
    else:
        runs_root = (Path("synth_dataset") / "runs").resolve()
        if not runs_root.exists():
            raise FileNotFoundError("No synth_dataset/runs directory found")
        runs = sorted(
            [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit()],
            key=lambda p: int(p.name[3:]),
        )
        if not runs:
            raise FileNotFoundError("No dataset runs found under synth_dataset/runs")
        resolved = runs[-1]
        log(f"No --dataset-dir/--dataset-run provided; using latest dataset run: {resolved.name}")

    manifest_path = resolved / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset_manifest.json not found: {manifest_path}")
    return resolved


def resolve_split_file(split_file: Path | None) -> Path:
    def is_closed_set_scene_split(path: Path) -> bool:
        try:
            with np.load(path, allow_pickle=False) as d:
                if "split_mode_code" in d.files:
                    mode_code = np.asarray(d["split_mode_code"], dtype=np.int8).reshape(-1)
                else:
                    mode_code = np.asarray([0], dtype=np.int8)
                if mode_code.size > 0 and int(mode_code[0]) == 1:
                    return True
                # Backward compatibility for older files: infer closed-set if guide sets overlap.
                tr = set(np.asarray(d["train_guides"], dtype=np.int64).tolist()) if "train_guides" in d.files else set()
                va = set(np.asarray(d["val_guides"], dtype=np.int64).tolist()) if "val_guides" in d.files else set()
                te = set(np.asarray(d["test_guides"], dtype=np.int64).tolist()) if "test_guides" in d.files else set()
                return bool(tr & va) or bool(tr & te) or bool(va & te)
        except Exception:
            return False

    if split_file is None:
        raise ValueError(
            "--split-file is required. Generate it first with: "
            "python3 -m GNN.split.split --dataset-run <run> --seed <seed>"
        )

    out = split_file.expanduser().resolve()
    if not out.exists():
        raise FileNotFoundError(f"Split file not found: {out}")
    if not is_closed_set_scene_split(out):
        raise RuntimeError(
            "Provided split file is not a closed-set scene split. "
            "Regenerate with: python3 -m GNN.split.split --dataset-run <run> --seed <seed>"
        )
    return out


def load_manifest(dataset_dir: Path) -> Dict[str, object]:
    with (dataset_dir / "dataset_manifest.json").open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    if "chunks" not in manifest:
        raise RuntimeError("Invalid dataset_manifest.json: missing 'chunks'")
    return manifest


def manifest_class_star_ids(manifest: Dict[str, object]) -> np.ndarray | None:
    raw = manifest.get("class_star_ids")
    if not isinstance(raw, list) or not raw:
        return None
    values = np.asarray(raw, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return None
    return np.unique(values).astype(np.int64, copy=False)


def chunk_paths(dataset_dir: Path, manifest: Dict[str, object]) -> List[Path]:
    chunks = manifest.get("chunks", [])
    paths: List[Path] = []
    for item in chunks:
        # Dataset manifests may be generated on Windows and later consumed on
        # Linux/HPC. Normalize separators before constructing a platform Path.
        rel = Path(str(item["file"]).replace("\\", "/"))
        p = (dataset_dir / rel).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Chunk file not found: {p}")
        paths.append(p)
    if not paths:
        raise RuntimeError("No chunks found in dataset manifest")
    return paths


def image_size_from_manifest(manifest: Dict[str, object]) -> Tuple[int, int]:
    run = manifest.get("run", {})
    params = run.get("parameters", {}) if isinstance(run, dict) else {}
    resolution = params.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        width = int(resolution[0])
        height = int(resolution[1])
        return width, height
    return 1280, 960


def database_path_from_manifest(manifest: Dict[str, object]) -> Path:
    run = manifest.get("run", {})
    params = run.get("parameters", {}) if isinstance(run, dict) else {}
    raw_path = params.get("database_path") or run.get("database_path") if isinstance(run, dict) else None
    if raw_path:
        path = Path(str(raw_path).replace("\\", "/"))
        if path.exists():
            return path
        if not path.is_absolute():
            candidate = (REPO_ROOT / path).resolve()
            if candidate.exists():
                return candidate
    return (REPO_ROOT / "tetra3" / "data" / "default_database.npz").resolve()


def build_star_mapping(
    paths: Sequence[Path],
    explicit_class_star_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Map real star IDs to dense class indices [0..C-1]."""
    if explicit_class_star_ids is not None and int(explicit_class_star_ids.size) > 0:
        class_to_star_id = np.unique(np.asarray(explicit_class_star_ids, dtype=np.int64)).astype(
            np.int64,
            copy=False,
        )
        star_id_to_class = {int(star_id): i for i, star_id in enumerate(class_to_star_id.tolist())}
        return class_to_star_id, star_id_to_class

    all_real: List[np.ndarray] = []
    for p in paths:
        with np.load(p, allow_pickle=False) as data:
            ids = np.asarray(data["point_star_id"], dtype=np.int64)
            real = ids[ids >= 0]
            if real.size > 0:
                all_real.append(real)
    if not all_real:
        raise RuntimeError("No real stars found in dataset chunks")
    class_to_star_id = np.unique(np.concatenate(all_real, axis=0)).astype(np.int64)
    star_id_to_class = {int(star_id): i for i, star_id in enumerate(class_to_star_id.tolist())}
    return class_to_star_id, star_id_to_class


def build_class_loss_features(
    *,
    database_path: Path,
    class_to_star_id: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with np.load(database_path, allow_pickle=False) as data:
        if "star_table" not in data:
            raise RuntimeError(f"Database {database_path} does not contain star_table")
        star_table = np.asarray(data["star_table"], dtype=np.float32)

    star_ids = np.asarray(class_to_star_id, dtype=np.int64)
    if star_ids.size == 0:
        raise RuntimeError("Cannot build class loss features without classes")
    if int(np.max(star_ids)) >= int(star_table.shape[0]) or int(np.min(star_ids)) < 0:
        raise RuntimeError("Class star IDs are not valid indices into database star_table")

    class_vectors = star_table[star_ids, 2:5].astype(np.float32, copy=False)
    norms = np.linalg.norm(class_vectors, axis=1, keepdims=True)
    class_vectors = class_vectors / np.maximum(norms, 1e-12)

    class_mag = star_table[star_ids, 5].astype(np.float32, copy=False)
    if class_mag.size <= 1:
        class_mag_rank = np.zeros_like(class_mag, dtype=np.float32)
    else:
        order = np.argsort(class_mag, kind="stable")
        class_mag_rank = np.empty_like(class_mag, dtype=np.float32)
        class_mag_rank[order] = np.arange(class_mag.size, dtype=np.float32) / float(class_mag.size - 1)

    return (
        torch.from_numpy(class_vectors.astype(np.float32, copy=False)).to(device),
        torch.from_numpy(class_mag_rank.astype(np.float32, copy=False)).to(device),
    )


def load_split_refs(split_npz: Path, split_name: str) -> List[Tuple[int, int]]:
    prefix = split_name.lower().strip()
    if prefix not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split name: {split_name}")

    with np.load(split_npz, allow_pickle=False) as d:
        chunk_idx = np.asarray(d[f"{prefix}_chunk_idx"], dtype=np.int64)
        scene_idx = np.asarray(d[f"{prefix}_scene_idx"], dtype=np.int64)
    if chunk_idx.shape != scene_idx.shape:
        raise RuntimeError(f"Malformed split file: {split_npz}")
    return [(int(c), int(s)) for c, s in zip(chunk_idx.tolist(), scene_idx.tolist())]


def collect_candidate_star_ids_from_refs(
    paths: Sequence[Path],
    refs: Sequence[Tuple[int, int]],
    top_n: int | None,
) -> np.ndarray:
    """Collect real star IDs that can appear as model targets for a split."""
    by_chunk: Dict[int, List[int]] = {}
    for chunk_idx, scene_idx in refs:
        by_chunk.setdefault(int(chunk_idx), []).append(int(scene_idx))

    out: List[np.ndarray] = []
    for chunk_idx, scene_indices in by_chunk.items():
        with np.load(paths[int(chunk_idx)], allow_pickle=False) as data:
            scene_start = np.asarray(data["scene_point_start"], dtype=np.int64)
            scene_count = np.asarray(data["scene_point_count"], dtype=np.int64)
            point_star_id = np.asarray(data["point_star_id"], dtype=np.int64)

            for scene_idx in scene_indices:
                start = int(scene_start[int(scene_idx)])
                count = int(scene_count[int(scene_idx)])
                limit = count if top_n is None or int(top_n) <= 0 else min(count, int(top_n))
                if limit <= 0:
                    continue
                ids = point_star_id[start : start + limit]
                ids = ids[ids >= 0]
                if ids.size:
                    out.append(ids.astype(np.int64, copy=False))

    if not out:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(out, axis=0)).astype(np.int64, copy=False)


class ChunkCache:
    def __init__(self, max_items: int = 2):
        self.max_items = max(1, int(max_items))
        self._cache: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()

    def get(self, chunk_idx: int, chunk_path: Path) -> Dict[str, np.ndarray]:
        chunk_idx = int(chunk_idx)
        if chunk_idx in self._cache:
            self._cache.move_to_end(chunk_idx)
            return self._cache[chunk_idx]

        with np.load(chunk_path, allow_pickle=False) as d:
            loaded = {
                "point_yx": np.asarray(d["point_yx"], dtype=np.float32),
                "point_star_id": np.asarray(d["point_star_id"], dtype=np.int64),
                "point_is_false_star": np.asarray(d["point_is_false_star"], dtype=bool),
                "point_magnitude": np.asarray(d["point_magnitude"], dtype=np.float32),
                "scene_point_start": np.asarray(d["scene_point_start"], dtype=np.int64),
                "scene_point_count": np.asarray(d["scene_point_count"], dtype=np.int64),
                "guide_star_index": np.asarray(d["guide_star_index"], dtype=np.int64),
                "roll_degree": np.asarray(d["roll_degree"], dtype=np.float32),
                "scene_seed": np.asarray(d["scene_seed"], dtype=np.int64),
            }

        self._cache[chunk_idx] = loaded
        self._cache.move_to_end(chunk_idx)
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return loaded


def build_knn_edges(point_yx: np.ndarray, k_neighbors: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    n = int(point_yx.shape[0])
    if n == 1:
        return np.array([[0], [0]], dtype=np.int64)

    k_eff = min(max(1, int(k_neighbors)), n - 1)
    tree = cKDTree(point_yx)
    _, idx = tree.query(point_yx, k=k_eff + 1)
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx[:, None]

    src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    dst = idx[:, 1 : k_eff + 1].reshape(-1).astype(np.int64)
    return np.stack((src, dst), axis=0)


def build_fully_connected_edges(point_yx: np.ndarray) -> np.ndarray:
    n = int(point_yx.shape[0])
    if n <= 1:
        return np.array([[0], [0]], dtype=np.int64)

    src, dst = np.where(~np.eye(n, dtype=bool))
    return np.stack((src.astype(np.int64), dst.astype(np.int64)), axis=0)


def quad_combo_by_index(source_top_n: int, combo_idx: int) -> Tuple[int, int, int, int]:
    source_top_n = int(source_top_n)
    combo_idx = int(combo_idx)
    combos = itertools.combinations(range(source_top_n), 4)
    return tuple(next(itertools.islice(combos, combo_idx, None)))


def choose_balanced_quad_combo(
    point_star_id: np.ndarray,
    source_top_n: int,
    star_as_input_count: Dict[int, int],
    rng: np.random.Generator,
    eligible_star_ids: set[int] | None = None,
    star_as_input_candidate_count: Dict[int, int] | None = None,
) -> Tuple[int, Tuple[int, int, int, int]]:
    """Choose one 4-star combo that keeps per-star selection rates balanced."""
    source_top_n = int(source_top_n)
    if source_top_n < 4:
        raise ValueError("source_top_n must be >= 4")

    source_star_ids = np.asarray(point_star_id[:source_top_n], dtype=np.int64)
    target_rate = 4.0 / float(source_top_n)
    eligible_positions = []
    for local_idx, star_id in enumerate(source_star_ids.tolist()):
        star_id = int(star_id)
        if star_id >= 0 and (eligible_star_ids is None or star_id in eligible_star_ids):
            eligible_positions.append(int(local_idx))

    if star_as_input_candidate_count is not None:
        best_score: Tuple[int, float, float, Tuple[float, ...]] | None = None
        best: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for combo_idx, combo in enumerate(itertools.combinations(range(source_top_n), 4)):
            combo = tuple(int(i) for i in combo)
            combo_set = set(combo)
            selected_ineligible = 0
            for local_idx in combo:
                star_id = int(source_star_ids[int(local_idx)])
                if star_id < 0 or (eligible_star_ids is not None and star_id not in eligible_star_ids):
                    selected_ineligible += 1

            deviations = []
            for local_idx in eligible_positions:
                star_id = int(source_star_ids[int(local_idx)])
                candidate_count = max(1, int(star_as_input_candidate_count.get(star_id, 0)))
                selected_count = int(star_as_input_count.get(star_id, 0))
                if local_idx in combo_set:
                    selected_count += 1
                rate = float(selected_count) / float(candidate_count)
                deviations.append(abs(rate - target_rate))

            sorted_deviations = tuple(sorted((float(x) for x in deviations), reverse=True))
            max_deviation = float(max(deviations)) if deviations else 0.0
            squared_deviation = float(sum(float(x) * float(x) for x in deviations))
            score = (int(selected_ineligible), max_deviation, squared_deviation, sorted_deviations)
            if best_score is None or score < best_score:
                best_score = score
                best = [(int(combo_idx), combo)]
            elif score == best_score:
                best.append((int(combo_idx), combo))

        if not best:
            raise RuntimeError("No valid 4-star combinations found")
        chosen_idx = int(rng.integers(0, len(best))) if len(best) > 1 else 0
        return best[chosen_idx]

    real_source_counts = [
        int(star_as_input_count.get(int(star_id), 0))
        for star_id in source_star_ids.tolist()
        if int(star_id) >= 0 and (eligible_star_ids is None or int(star_id) in eligible_star_ids)
    ]
    false_star_penalty = int(max(real_source_counts) + 1) if real_source_counts else 0

    best_score: Tuple[int, Tuple[int, ...]] | None = None
    best: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for combo_idx, combo in enumerate(itertools.combinations(range(source_top_n), 4)):
        counts = []
        for local_idx in combo:
            star_id = int(source_star_ids[int(local_idx)])
            if star_id >= 0 and (eligible_star_ids is None or star_id in eligible_star_ids):
                counts.append(int(star_as_input_count.get(star_id, 0)))
            else:
                counts.append(false_star_penalty)
        sorted_counts = tuple(sorted(counts, reverse=True))
        score = (int(sum(counts)), sorted_counts)
        if best_score is None or score < best_score:
            best_score = score
            best = [(int(combo_idx), tuple(int(i) for i in combo))]
        elif score == best_score:
            best.append((int(combo_idx), tuple(int(i) for i in combo)))

    if not best:
        raise RuntimeError("No valid 4-star combinations found")
    chosen_idx = int(rng.integers(0, len(best))) if len(best) > 1 else 0
    return best[chosen_idx]


def choose_graph_k_neighbors(point_count: int) -> int:
    point_count = max(int(point_count), 0)
    if point_count <= 0:
        return 1
    return int(min(max(3, round(0.25 * point_count)), 8))


def infer_model_max_neighbors(
    *,
    graph_connectivity: str,
    top_n_choices: Sequence[int],
    quad_combinations_top_n: int | None,
    sample_node_count: int,
) -> int:
    """Infer the fixed neighbour slots needed by EdgeMLPStarGNN."""
    if quad_combinations_top_n:
        return 3
    if graph_connectivity == "fully":
        if top_n_choices:
            return max(1, int(max(top_n_choices)) - 1)
        return max(1, int(sample_node_count) - 1)
    if top_n_choices:
        return max(1, max(choose_graph_k_neighbors(int(n)) for n in top_n_choices))
    return max(1, choose_graph_k_neighbors(int(sample_node_count)))


def normalize_point_magnitude(point_mag: np.ndarray) -> np.ndarray:
    point_mag = np.asarray(point_mag, dtype=np.float32)
    if point_mag.size == 0:
        return point_mag.astype(np.float32, copy=True)

    # Magnitudes include an arbitrary zero-point offset. Subtracting the
    # brightest star (lowest magnitude) keeps only relative brightness.
    min_mag = float(np.min(point_mag))
    return (point_mag - min_mag).astype(np.float32)


def parse_top_n_choices(raw: str | None) -> Tuple[int, ...]:
    if raw is None:
        return ()

    values: List[int] = []
    for token in raw.replace(",", " ").split():
        value = int(token)
        if value <= 0:
            raise ValueError("top-n choices must be positive integers")
        if value not in values:
            values.append(value)
    return tuple(values)


def effective_top_n_choices(point_count: int, top_n_choices: Sequence[int]) -> Tuple[int, ...]:
    point_count = max(int(point_count), 0)
    if point_count <= 0:
        return ()
    if not top_n_choices:
        return (point_count,)

    clipped: List[int] = []
    for value in top_n_choices:
        n = max(1, min(int(value), point_count))
        if n not in clipped:
            clipped.append(n)
    return tuple(clipped) if clipped else (point_count,)


def choose_scene_top_n(
    point_count: int,
    top_n_choices: Sequence[int],
    *,
    scene_seed: int,
    top_n_seed: int,
    sample_top_n: bool,
) -> int:
    choices = effective_top_n_choices(point_count, top_n_choices)
    if not choices:
        return 0
    if not sample_top_n or len(choices) == 1:
        return int(choices[-1])

    rng = np.random.default_rng(int(scene_seed) + int(top_n_seed) * 1000003)
    return int(choices[int(rng.integers(0, len(choices)))])


def safe_feature_scale(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or abs(value) < 1e-6:
        return 1.0
    return value


def magnitude_rank_feature(point_mag: np.ndarray) -> np.ndarray:
    point_mag = np.asarray(point_mag, dtype=np.float32)
    n_points = int(point_mag.shape[0])
    if n_points <= 1:
        return np.zeros((n_points,), dtype=np.float32)

    order = np.argsort(point_mag, kind="stable")
    ranks = np.empty((n_points,), dtype=np.float32)
    ranks[order] = np.arange(n_points, dtype=np.float32) / float(n_points - 1)
    return ranks


def magnitude_rank_1based_feature(point_mag: np.ndarray) -> np.ndarray:
    point_mag = np.asarray(point_mag, dtype=np.float32)
    n_points = int(point_mag.shape[0])
    if n_points <= 0:
        return np.zeros((0,), dtype=np.float32)

    order = np.argsort(point_mag, kind="stable")
    ranks = np.empty((n_points,), dtype=np.float32)
    ranks[order] = np.arange(1, n_points + 1, dtype=np.float32)
    return ranks


def build_node_features(
    point_mag: np.ndarray,
    node_feature_mode: str = "magnitude_subtracted_rank",
) -> Tuple[np.ndarray, np.ndarray]:
    point_mag = np.asarray(point_mag, dtype=np.float32)
    point_mag_subtracted = normalize_point_magnitude(point_mag)
    n_points = int(point_mag.shape[0])

    if node_feature_mode == "none":
        node_x = np.zeros((n_points, 1), dtype=np.float32)
    elif node_feature_mode == "magnitude_rank":
        rank_norm = magnitude_rank_feature(point_mag)
        node_x = rank_norm.reshape(-1, 1).astype(np.float32, copy=False)
    elif node_feature_mode == "magnitude_rank_1based":
        rank_1based = magnitude_rank_1based_feature(point_mag)
        node_x = rank_1based.reshape(-1, 1).astype(np.float32, copy=False)
    elif node_feature_mode == "magnitude":
        node_x = point_mag.reshape(-1, 1).astype(np.float32, copy=False)
    elif node_feature_mode == "magnitude_subtracted":
        node_x = point_mag_subtracted.reshape(-1, 1).astype(np.float32, copy=False)
    elif node_feature_mode == "magnitude_norm_max":
        scale = safe_feature_scale(float(np.max(np.abs(point_mag))) if point_mag.size else 1.0)
        node_x = (point_mag / scale).reshape(-1, 1).astype(np.float32, copy=False)
    elif node_feature_mode == "magnitude_norm_median":
        scale = safe_feature_scale(float(np.median(np.abs(point_mag))) if point_mag.size else 1.0)
        node_x = (point_mag / scale).reshape(-1, 1).astype(np.float32, copy=False)
    elif node_feature_mode == "magnitude_subtracted_norm_max":
        scale = safe_feature_scale(float(np.max(point_mag_subtracted)) if point_mag_subtracted.size else 1.0)
        node_x = (point_mag_subtracted / scale).reshape(-1, 1).astype(np.float32, copy=False)
    elif node_feature_mode == "magnitude_subtracted_norm_median":
        scale = safe_feature_scale(float(np.median(np.abs(point_mag_subtracted))) if point_mag_subtracted.size else 1.0)
        node_x = (point_mag_subtracted / scale).reshape(-1, 1).astype(np.float32, copy=False)
    elif node_feature_mode == "magnitude_subtracted_rank":
        rank_norm = magnitude_rank_feature(point_mag)
        node_x = np.stack((point_mag_subtracted, rank_norm.astype(np.float32)), axis=1)
    else:
        raise ValueError(f"Unsupported node_feature_mode: {node_feature_mode}")

    return node_x.astype(np.float32, copy=False), point_mag_subtracted


def edge_attr_from_geometry(
    point_yx: np.ndarray,
    point_mag: np.ndarray,
    edge_index: np.ndarray,
    width: int,
    height: int,
    edge_feature_mode: str = "distance_diagonal_dmag",
) -> np.ndarray:
    src, dst = edge_index
    delta = point_yx[dst] - point_yx[src]
    raw_dist = np.linalg.norm(delta, axis=1).astype(np.float32)

    include_dmag_node = edge_feature_mode.endswith("_dmag_node")
    include_dmag = include_dmag_node or edge_feature_mode.endswith("_dmag")
    if include_dmag_node:
        distance_mode = edge_feature_mode[: -len("_dmag_node")]
    elif include_dmag:
        distance_mode = edge_feature_mode[:-5]
    else:
        distance_mode = edge_feature_mode
    if distance_mode == "distance_raw":
        dist = raw_dist
    elif distance_mode == "distance_diagonal":
        diag = safe_feature_scale(float(np.hypot(float(height), float(width))))
        dist = (raw_dist / diag).astype(np.float32)
    elif distance_mode == "distance_max":
        scale = safe_feature_scale(float(np.max(raw_dist)) if raw_dist.size else 1.0)
        dist = (raw_dist / scale).astype(np.float32)
    else:
        raise ValueError(f"Unsupported edge_feature_mode: {edge_feature_mode}")

    features = [dist.astype(np.float32, copy=False)]
    if include_dmag:
        dmag = (point_mag[dst] - point_mag[src]).astype(np.float32)
        features.append(dmag)
    return np.stack(features, axis=1)


def build_graph_inputs(
    point_yx: np.ndarray,
    point_mag: np.ndarray,
    width: int,
    height: int,
    k_neighbors: int,
    graph_connectivity: str = "knn",
    node_feature_mode: str = "magnitude_subtracted_rank",
    edge_feature_mode: str = "distance_diagonal_dmag",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_x, point_mag_norm = build_node_features(point_mag, node_feature_mode=node_feature_mode)
    if edge_feature_mode.endswith("_dmag_node"):
        point_mag_for_edges = node_x[:, 0]
    else:
        point_mag_for_edges = point_mag_norm
    if graph_connectivity == "fully":
        edge_index = build_fully_connected_edges(point_yx=point_yx)
    elif graph_connectivity == "knn":
        edge_index = build_knn_edges(point_yx=point_yx, k_neighbors=k_neighbors)
    else:
        raise ValueError(f"Unsupported graph_connectivity: {graph_connectivity}")
    edge_attr = edge_attr_from_geometry(
        point_yx=point_yx,
        point_mag=point_mag_for_edges,
        edge_index=edge_index,
        width=width,
        height=height,
        edge_feature_mode=edge_feature_mode,
    )
    return node_x, edge_index, edge_attr


class SceneGraphDataset(Dataset):
    def __init__(
        self,
        *,
        chunk_paths: Sequence[Path],
        refs: Sequence[Tuple[int, int]],
        star_id_to_class: Dict[int, int],
        width: int,
        height: int,
        cache_chunks: int,
        top_n_choices: Sequence[int],
        top_n_mode: str,
        top_n_seed: int,
        graph_connectivity: str,
        node_feature_mode: str,
        edge_feature_mode: str,
        quad_combinations_top_n: int | None,
        quad_combination_mode: str,
    ) -> None:
        ensure_pyg_available()
        from torch_geometric.data import Data

        self.Data = Data
        self.chunk_paths = list(chunk_paths)
        self.refs = list(refs)
        self.star_id_to_class = star_id_to_class
        self.width = int(width)
        self.height = int(height)
        self.cache = ChunkCache(max_items=cache_chunks)
        self.top_n_choices = tuple(int(x) for x in top_n_choices)
        if top_n_mode not in {"expand", "sample", "max"}:
            raise ValueError("top_n_mode must be one of: expand, sample, max")
        self.top_n_mode = str(top_n_mode)
        self.top_n_seed = int(top_n_seed)
        self.expand_factor = max(1, len(self.top_n_choices)) if self.top_n_mode == "expand" else 1
        if graph_connectivity not in GRAPH_CONNECTIVITY_CHOICES:
            raise ValueError(f"graph_connectivity must be one of: {', '.join(GRAPH_CONNECTIVITY_CHOICES)}")
        self.graph_connectivity = str(graph_connectivity)
        if node_feature_mode not in NODE_FEATURE_MODE_CHOICES:
            raise ValueError(f"node_feature_mode must be one of: {', '.join(NODE_FEATURE_MODE_CHOICES)}")
        self.node_feature_mode = str(node_feature_mode)
        if edge_feature_mode not in EDGE_FEATURE_MODE_CHOICES:
            raise ValueError(f"edge_feature_mode must be one of: {', '.join(EDGE_FEATURE_MODE_CHOICES)}")
        self.edge_feature_mode = str(edge_feature_mode)
        self.quad_combinations_top_n = int(quad_combinations_top_n or 0)
        if self.quad_combinations_top_n and self.quad_combinations_top_n < 4:
            raise ValueError("quad_combinations_top_n must be >= 4")
        if quad_combination_mode not in QUAD_COMBINATION_MODE_CHOICES:
            raise ValueError(f"quad_combination_mode must be one of: {', '.join(QUAD_COMBINATION_MODE_CHOICES)}")
        self.quad_combination_mode = str(quad_combination_mode)
        self.quad_combo_counts: np.ndarray | None = None
        self.quad_combo_offsets: np.ndarray | None = None
        self.balanced_quad_combo_by_ref: Dict[Tuple[int, int], int] | None = None
        self.balance_star_ids: set[int] = set(int(star_id) for star_id in self.star_id_to_class.keys())
        self.star_as_input_count: Dict[int, int] = {}
        self.star_as_input_candidate_count: Dict[int, int] = {}
        self.star_as_input_candidate_ids: set[int] = set()
        if self.quad_combinations_top_n and self.quad_combination_mode == "all":
            self.quad_combo_counts = self._build_quad_combo_counts()
            self.quad_combo_offsets = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.cumsum(self.quad_combo_counts, dtype=np.int64),
                )
            )
        elif self.quad_combinations_top_n and self.quad_combination_mode == "balanced_sample":
            self.balanced_quad_combo_by_ref = self._build_balanced_quad_combo_samples()

    def _build_quad_combo_counts(self) -> np.ndarray:
        counts = np.zeros(len(self.refs), dtype=np.int64)
        refs_by_chunk: Dict[int, List[Tuple[int, int]]] = {}
        for ref_idx, (chunk_idx, scene_idx) in enumerate(self.refs):
            refs_by_chunk.setdefault(int(chunk_idx), []).append((int(ref_idx), int(scene_idx)))

        for chunk_idx, items in refs_by_chunk.items():
            shard = self.cache.get(chunk_idx, self.chunk_paths[chunk_idx])
            scene_point_count = shard["scene_point_count"]
            for ref_idx, scene_idx in items:
                n = min(int(scene_point_count[scene_idx]), self.quad_combinations_top_n)
                counts[ref_idx] = math.comb(n, 4) if n >= 4 else 0
        return counts

    def _build_balanced_quad_combo_samples(self) -> Dict[Tuple[int, int], int]:
        selected: Dict[Tuple[int, int], int] = {}
        rng = np.random.default_rng(int(self.top_n_seed))

        for chunk_idx, scene_idx in self.refs:
            chunk_idx = int(chunk_idx)
            scene_idx = int(scene_idx)
            shard = self.cache.get(chunk_idx, self.chunk_paths[chunk_idx])
            start = int(shard["scene_point_start"][scene_idx])
            count = int(shard["scene_point_count"][scene_idx])
            source_top_n = min(int(count), int(self.quad_combinations_top_n))
            combo_count = math.comb(source_top_n, 4) if source_top_n >= 4 else 0
            if combo_count <= 0:
                raise IndexError(
                    f"Scene {scene_idx} in chunk {chunk_idx} has only {source_top_n} points; "
                    "cannot build a balanced 4-star quad combination"
                )

            point_star_id = shard["point_star_id"][start : start + count].astype(np.int64, copy=False)
            for star_id in point_star_id[:source_top_n].tolist():
                star_id = int(star_id)
                if star_id >= 0 and star_id in self.balance_star_ids:
                    self.star_as_input_candidate_ids.add(star_id)
                    self.star_as_input_candidate_count[star_id] = int(
                        self.star_as_input_candidate_count.get(star_id, 0) + 1
                    )
            combo_idx, combo = choose_balanced_quad_combo(
                point_star_id=point_star_id,
                source_top_n=source_top_n,
                star_as_input_count=self.star_as_input_count,
                rng=rng,
                eligible_star_ids=self.balance_star_ids,
                star_as_input_candidate_count=self.star_as_input_candidate_count,
            )
            selected[(chunk_idx, scene_idx)] = int(combo_idx)
            for local_idx in combo:
                star_id = int(point_star_id[int(local_idx)])
                if star_id >= 0 and star_id in self.balance_star_ids:
                    self.star_as_input_count[star_id] = int(self.star_as_input_count.get(star_id, 0) + 1)

        return selected

    def star_as_input_count_summary(self) -> Dict[str, object] | None:
        if self.quad_combination_mode != "balanced_sample":
            return None

        candidate_ids = sorted(self.star_as_input_candidate_ids)
        selected_counts = np.asarray(
            [int(self.star_as_input_count.get(star_id, 0)) for star_id in candidate_ids],
            dtype=np.int64,
        )
        candidate_counts = np.asarray(
            [int(self.star_as_input_candidate_count.get(star_id, 0)) for star_id in candidate_ids],
            dtype=np.int64,
        )
        if selected_counts.size == 0:
            return {
                "all_stars": int(len(self.balance_star_ids)),
                "candidate_stars": 0,
                "selected_stars": 0,
                "top8_candidate_total": 0,
                "total_star_inputs": 0,
                "star_as_input_count_min": 0,
                "star_as_input_count_mean": 0.0,
                "star_as_input_count_max": 0,
                "top8_candidate_count_min": 0,
                "top8_candidate_count_mean": 0.0,
                "top8_candidate_count_max": 0,
                "selection_rate_min": 0.0,
                "selection_rate_mean": 0.0,
                "selection_rate_max": 0.0,
            }
        selection_rates = selected_counts.astype(np.float64) / np.maximum(candidate_counts, 1).astype(np.float64)
        return {
            "all_stars": int(len(self.balance_star_ids)),
            "candidate_stars": int(selected_counts.size),
            "selected_stars": int(np.sum(selected_counts > 0)),
            "top8_candidate_total": int(np.sum(candidate_counts)),
            "total_star_inputs": int(np.sum(selected_counts)),
            "star_as_input_count_min": int(np.min(selected_counts)),
            "star_as_input_count_mean": float(np.mean(selected_counts)),
            "star_as_input_count_max": int(np.max(selected_counts)),
            "top8_candidate_count_min": int(np.min(candidate_counts)),
            "top8_candidate_count_mean": float(np.mean(candidate_counts)),
            "top8_candidate_count_max": int(np.max(candidate_counts)),
            "selection_rate_min": float(np.min(selection_rates)),
            "selection_rate_mean": float(np.mean(selection_rates)),
            "selection_rate_max": float(np.max(selection_rates)),
        }

    def __len__(self) -> int:
        if self.quad_combo_offsets is not None:
            return int(self.quad_combo_offsets[-1])
        if self.quad_combinations_top_n and self.quad_combination_mode in {"sample", "balanced_sample"}:
            return len(self.refs)
        return len(self.refs) * self.expand_factor

    def __getitem__(self, idx: int):
        raw_idx = int(idx)
        combo_indices: Tuple[int, ...] | None = None
        sample_weight = 1.0
        if self.quad_combo_offsets is not None and self.quad_combo_counts is not None:
            ref_idx = int(np.searchsorted(self.quad_combo_offsets, raw_idx, side="right") - 1)
            combo_idx = raw_idx - int(self.quad_combo_offsets[ref_idx])
            combo_count = int(self.quad_combo_counts[ref_idx])
            if combo_count <= 0:
                raise IndexError(f"Scene {ref_idx} has no valid 4-star combinations")
            requested_top_n = int(self.quad_combinations_top_n)
            sample_weight = 1.0 / float(combo_count)
        elif self.quad_combinations_top_n and self.quad_combination_mode in {"sample", "balanced_sample"}:
            ref_idx = raw_idx
            combo_idx = None
            requested_top_n = int(self.quad_combinations_top_n)
        elif self.top_n_mode == "expand" and self.top_n_choices:
            ref_idx = raw_idx // self.expand_factor
            choice_idx = raw_idx % self.expand_factor
            requested_top_n = int(self.top_n_choices[choice_idx])
            sample_weight = 1.0 / float(self.expand_factor)
        else:
            ref_idx = raw_idx
            requested_top_n = None

        chunk_idx, scene_idx = self.refs[ref_idx]
        shard = self.cache.get(chunk_idx, self.chunk_paths[chunk_idx])

        start = int(shard["scene_point_start"][scene_idx])
        count = int(shard["scene_point_count"][scene_idx])
        end = start + count

        point_yx = shard["point_yx"][start:end].astype(np.float32, copy=False)
        point_star_id = shard["point_star_id"][start:end].astype(np.int64, copy=False)
        point_is_false_star = shard["point_is_false_star"][start:end].astype(bool, copy=False)
        point_mag = shard["point_magnitude"][start:end].astype(np.float32, copy=False)
        scene_seed = int(shard["scene_seed"][scene_idx])

        if point_yx.shape[0] == 0:
            point_yx = np.array([[self.height / 2.0, self.width / 2.0]], dtype=np.float32)
            point_star_id = np.array([-1], dtype=np.int64)
            point_is_false_star = np.array([True], dtype=bool)
            point_mag = np.array([0.0], dtype=np.float32)
            top_n = 1
        else:
            if self.quad_combinations_top_n:
                source_top_n = min(int(requested_top_n), int(point_yx.shape[0]))
                combo_count = math.comb(source_top_n, 4) if source_top_n >= 4 else 0
                if combo_count <= 0:
                    raise IndexError(
                        f"Scene has only {source_top_n} points; cannot build a 4-star quad combination"
                    )
                if combo_idx is None:
                    if self.quad_combination_mode == "balanced_sample":
                        if self.balanced_quad_combo_by_ref is None:
                            raise RuntimeError("Balanced quad sample index was not precomputed")
                        combo_idx = int(self.balanced_quad_combo_by_ref[(int(chunk_idx), int(scene_idx))])
                    else:
                        rng = np.random.default_rng(int(scene_seed) + int(self.top_n_seed) * 1000003)
                        combo_idx = int(rng.integers(0, combo_count))
                combo_indices = quad_combo_by_index(source_top_n, int(combo_idx))
                point_yx = point_yx[list(combo_indices)]
                point_star_id = point_star_id[list(combo_indices)]
                point_is_false_star = point_is_false_star[list(combo_indices)]
                point_mag = point_mag[list(combo_indices)]
                top_n = int(point_yx.shape[0])
            elif requested_top_n is None:
                top_n = choose_scene_top_n(
                    point_count=int(point_yx.shape[0]),
                    top_n_choices=self.top_n_choices,
                    scene_seed=scene_seed,
                    top_n_seed=self.top_n_seed,
                    sample_top_n=self.top_n_mode == "sample",
                )
            else:
                top_n = max(1, min(int(requested_top_n), int(point_yx.shape[0])))
            point_yx = point_yx[:top_n]
            point_star_id = point_star_id[:top_n]
            point_is_false_star = point_is_false_star[:top_n]
            point_mag = point_mag[:top_n]

        scene_k_neighbors = choose_graph_k_neighbors(int(point_yx.shape[0]))
        x_feat, edge_index, edge_attr = build_graph_inputs(
            point_yx=point_yx,
            point_mag=point_mag,
            width=self.width,
            height=self.height,
            k_neighbors=scene_k_neighbors,
            graph_connectivity=self.graph_connectivity,
            node_feature_mode=self.node_feature_mode,
            edge_feature_mode=self.edge_feature_mode,
        )

        id_y = np.zeros(point_star_id.shape[0], dtype=np.int64)
        all_real_mask = ~point_is_false_star
        real_mask = np.zeros(point_star_id.shape[0], dtype=bool)
        if np.any(all_real_mask):
            real_pos = np.flatnonzero(all_real_mask)
            real_ids = point_star_id[all_real_mask]
            mapped = np.asarray([self.star_id_to_class.get(int(star_id), -1) for star_id in real_ids], dtype=np.int64)
            keep = mapped >= 0
            if np.any(keep):
                kept_pos = real_pos[keep]
                id_y[kept_pos] = mapped[keep]
                real_mask[kept_pos] = True
        if np.any(real_mask):
            real_kept_pos = np.flatnonzero(real_mask)
            anchor_pos = int(real_kept_pos[int(np.argmin(point_mag[real_kept_pos]))])
            bright_anchor_y = int(id_y[anchor_pos])
        else:
            bright_anchor_y = -1

        data = self.Data(
            x=torch.from_numpy(x_feat),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(edge_attr),
        )
        data.id_y = torch.from_numpy(id_y).long()
        data.real_mask = torch.from_numpy(real_mask.astype(np.bool_))
        data.bright_anchor_y = torch.tensor([bright_anchor_y], dtype=torch.long)
        data.point_star_id = torch.from_numpy(point_star_id.astype(np.int64))
        guide_star = int(shard["guide_star_index"][scene_idx])
        data.guide_star_index = torch.tensor([guide_star], dtype=torch.long)
        data.roll_degree = torch.tensor([float(shard["roll_degree"][scene_idx])], dtype=torch.float32)
        data.scene_seed = torch.tensor([scene_seed], dtype=torch.long)
        data.top_n = torch.tensor([int(top_n)], dtype=torch.long)
        data.sample_weight = torch.tensor([float(sample_weight)], dtype=torch.float32)
        return data


def write_star_as_input_count_csv(run_dir: Path, split_name: str, dataset: SceneGraphDataset | None) -> Path | None:
    if dataset is None or dataset.quad_combination_mode != "balanced_sample":
        return None
    out = run_dir / f"star_as_input_count_{split_name}.csv"
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "star_id",
                "top8_candidate_count",
                "star_as_input_count",
                "selection_rate",
                "is_top8_candidate",
            ]
        )
        for star_id in sorted(dataset.balance_star_ids):
            candidate_count = int(dataset.star_as_input_candidate_count.get(int(star_id), 0))
            selected_count = int(dataset.star_as_input_count.get(int(star_id), 0))
            selection_rate = selected_count / candidate_count if candidate_count > 0 else ""
            writer.writerow(
                [
                    int(star_id),
                    candidate_count,
                    selected_count,
                    selection_rate,
                    int(candidate_count > 0),
                ]
            )
    return out


class PaperLikeStarGNN(nn.Module):
    """Paper-like message passing backbone for star ID."""

    def __init__(
        self,
        *,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        num_id_classes: int,
    ) -> None:
        super().__init__()
        ensure_pyg_available()

        self.dropout = float(dropout)

        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be > 0")
        if hidden_dim % max(heads, 1) != 0:
            raise ValueError("hidden_dim must be divisible by heads for gatv2")

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # nn.Linear aprende dois parâmetros: W (matriz) e b (vector)
        # fórmula: output = W · input + b

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        from torch_geometric.nn import GATv2Conv

        head_dim = hidden_dim // max(heads, 1)
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim, # ← diz ao GATv2Conv para usar edge features
                    dropout=self.dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.id_head = nn.Linear(hidden_dim, num_id_classes)

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) # x são as node features brutas, h é o embedding

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr) # edge_attr = tensor com todos os e_ij
            h = F.relu(norm(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self._encode(x, edge_index, edge_attr)
        return self.id_head(h)


class EdgeMLPStarGNN(nn.Module):
    """Node classifier that feeds edge features directly into a per-node MLP.

    For a 4-node fully connected graph each node receives:
      [x_i, e_i->j1, e_i->j2, e_i->j3, x_j1, x_j2, x_j3]
    with neighbours ordered by their local node index inside the graph.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        num_id_classes: int,
        max_neighbors: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be > 0")
        if in_dim <= 0 or edge_dim <= 0:
            raise ValueError("in_dim and edge_dim must be > 0")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be > 0")

        self.in_dim = int(in_dim)
        self.edge_dim = int(edge_dim)
        self.max_neighbors = int(max_neighbors)
        self.dropout = float(dropout)
        self.heads = int(heads)

        mlp_in_dim = self.in_dim + self.max_neighbors * self.edge_dim + self.max_neighbors * self.in_dim
        self.input_proj = nn.Linear(mlp_in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(max(0, int(num_layers) - 1))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(int(num_layers))])
        self.id_head = nn.Linear(hidden_dim, num_id_classes)
        dense_node_count = self.max_neighbors + 1
        dense_neighbour_index = [
            [j for j in range(dense_node_count) if j != i]
            for i in range(dense_node_count)
        ]
        self.register_buffer(
            "_dense_neighbour_index",
            torch.tensor(dense_neighbour_index, dtype=torch.long),
            persistent=False,
        )

    def _dense_node_inputs(self, x: torch.Tensor, edge_attr: torch.Tensor, graph_count: int) -> torch.Tensor:
        node_count = self.max_neighbors + 1
        x_graph = x.reshape(int(graph_count), node_count, self.in_dim)
        edge_graph = edge_attr.reshape(int(graph_count), node_count, self.max_neighbors, self.edge_dim)
        neighbour_x = x_graph[:, self._dense_neighbour_index.to(x.device), :]
        per_node = torch.cat(
            (
                x_graph,
                edge_graph.reshape(int(graph_count), node_count, self.max_neighbors * self.edge_dim),
                neighbour_x.reshape(int(graph_count), node_count, self.max_neighbors * self.in_dim),
            ),
            dim=2,
        )
        return per_node.reshape(x.shape[0], -1)

    def _node_inputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            batch = batch.to(device=x.device, dtype=torch.long)

        n_nodes = int(x.shape[0])
        out_dim = self.in_dim + self.max_neighbors * self.edge_dim + self.max_neighbors * self.in_dim
        if n_nodes == 0:
            return x.new_zeros((0, out_dim))

        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        dense_node_count = self.max_neighbors + 1
        graph_count = int(batch[-1].item()) + 1 if batch.numel() > 0 else 0
        if graph_count > 0:
            node_counts = torch.bincount(batch, minlength=graph_count)
            dense_edge_count = graph_count * dense_node_count * self.max_neighbors
            if (
                n_nodes == graph_count * dense_node_count
                and int(edge_attr.shape[0]) == int(dense_edge_count)
                and bool(torch.all(node_counts == dense_node_count).item())
            ):
                return self._dense_node_inputs(x, edge_attr, graph_count)

        out = x.new_zeros((n_nodes, out_dim))
        edge_start = self.in_dim
        neighbour_start = edge_start + self.max_neighbors * self.edge_dim

        graph_ids = torch.unique_consecutive(batch)
        for graph_id in graph_ids.tolist():
            node_idx = torch.nonzero(batch == int(graph_id), as_tuple=False).flatten()
            if node_idx.numel() == 0:
                continue
            start = int(node_idx[0].item())
            end = int(node_idx[-1].item()) + 1
            local_x = x[start:end]
            local_n = int(local_x.shape[0])
            out[start:end, : self.in_dim] = local_x

            edge_mask = (edge_index[0] >= start) & (edge_index[0] < end)
            graph_edge_pos = torch.nonzero(edge_mask, as_tuple=False).flatten()
            if graph_edge_pos.numel() == 0:
                continue
            src_local = edge_index[0, graph_edge_pos] - start
            dst_local = edge_index[1, graph_edge_pos] - start
            valid = (dst_local >= 0) & (dst_local < local_n)
            graph_edge_pos = graph_edge_pos[valid]
            src_local = src_local[valid]
            dst_local = dst_local[valid]

            for local_i in range(local_n):
                outgoing = torch.nonzero(src_local == local_i, as_tuple=False).flatten()
                if outgoing.numel() == 0:
                    continue
                outgoing_dst = dst_local[outgoing]
                order = torch.argsort(outgoing_dst)
                outgoing = outgoing[order[: self.max_neighbors]]
                for slot, rel_pos in enumerate(outgoing.tolist()):
                    edge_pos = int(graph_edge_pos[int(rel_pos)].item())
                    dst = int(dst_local[int(rel_pos)].item())
                    row = start + local_i
                    e0 = edge_start + slot * self.edge_dim
                    n0 = neighbour_start + slot * self.in_dim
                    out[row, e0 : e0 + self.edge_dim] = edge_attr[edge_pos]
                    out[row, n0 : n0 + self.in_dim] = local_x[dst]

        return out

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.input_proj(self._node_inputs(x, edge_index, edge_attr, batch))
        h = F.relu(self.norms[0](h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for layer, norm in zip(self.hidden_layers, self.norms[1:]):
            h = layer(h)
            h = F.relu(norm(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.id_head(self._encode(x, edge_index, edge_attr, batch))


def make_star_model(
    *,
    model_backbone: str,
    in_dim: int,
    edge_dim: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    dropout: float,
    num_id_classes: int,
    max_neighbors: int = 3,
) -> nn.Module:
    if model_backbone == "gatv2":
        return PaperLikeStarGNN(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            num_id_classes=num_id_classes,
        )
    if model_backbone == "edge_mlp":
        return EdgeMLPStarGNN(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            num_id_classes=num_id_classes,
            max_neighbors=int(max_neighbors),
        )
    raise ValueError(f"Unsupported model_backbone: {model_backbone}")


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        id_loss_weight: float,
        group_by_scene: bool = False,
        class_distance_loss_weight: float = 0.0,
        class_rank_loss_weight: float = 0.0,
        class_unit_vectors: torch.Tensor | None = None,
        class_mag_rank: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.id_loss_weight = float(id_loss_weight)
        self.group_by_scene = bool(group_by_scene)
        self.class_distance_loss_weight = float(class_distance_loss_weight)
        self.class_rank_loss_weight = float(class_rank_loss_weight)
        self.register_buffer(
            "class_unit_vectors",
            class_unit_vectors.detach().clone().float() if class_unit_vectors is not None else torch.empty(0, 3),
        )
        self.register_buffer(
            "class_mag_rank",
            class_mag_rank.detach().clone().float() if class_mag_rank is not None else torch.empty(0),
        )

    def _reduce_node_loss(self, per_node_loss: torch.Tensor, batch, real_mask: torch.Tensor) -> torch.Tensor:
        if self.group_by_scene and hasattr(batch, "batch") and hasattr(batch, "scene_seed"):
            graph_idx = batch.batch[real_mask]
            num_graphs = int(batch.scene_seed.view(-1).shape[0])
            graph_loss_sum = torch.zeros(num_graphs, dtype=per_node_loss.dtype, device=per_node_loss.device)
            graph_node_count = torch.zeros(num_graphs, dtype=per_node_loss.dtype, device=per_node_loss.device)
            graph_loss_sum.index_add_(0, graph_idx, per_node_loss)
            graph_node_count.index_add_(0, graph_idx, torch.ones_like(per_node_loss))

            graph_has_real = graph_node_count > 0
            graph_loss = graph_loss_sum[graph_has_real] / graph_node_count[graph_has_real].clamp_min(1.0)
            graph_scene_seed = batch.scene_seed.view(-1).to(per_node_loss.device)[graph_has_real]

            _, scene_inverse = torch.unique(graph_scene_seed, sorted=False, return_inverse=True)
            scene_count = int(scene_inverse.max().item()) + 1 if scene_inverse.numel() > 0 else 0
            if scene_count > 0:
                scene_loss_sum = torch.zeros(scene_count, dtype=graph_loss.dtype, device=graph_loss.device)
                scene_graph_count = torch.zeros(scene_count, dtype=graph_loss.dtype, device=graph_loss.device)
                scene_loss_sum.index_add_(0, scene_inverse, graph_loss)
                scene_graph_count.index_add_(0, scene_inverse, torch.ones_like(graph_loss))
                return (scene_loss_sum / scene_graph_count.clamp_min(1.0)).mean()
            return per_node_loss.mean()
        if hasattr(batch, "sample_weight") and hasattr(batch, "batch"):
            graph_idx = batch.batch[real_mask]
            graph_weights = batch.sample_weight.to(per_node_loss.device).view(-1)
            node_weights = graph_weights[graph_idx].to(per_node_loss.dtype)
            return (per_node_loss * node_weights).sum() / node_weights.sum().clamp_min(1e-12)
        return per_node_loss.mean()

    def forward(
        self,
        id_logits: torch.Tensor,
        batch,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        real_mask = batch.real_mask.bool()
        if bool(real_mask.any()):
            per_node_loss = F.cross_entropy(
                id_logits[real_mask],
                batch.id_y[real_mask].long(),
                reduction="none",
            )
            id_loss = self._reduce_node_loss(per_node_loss, batch, real_mask)

            true_class = batch.id_y[real_mask].long()
            prob = F.softmax(id_logits[real_mask], dim=1)
            if self.class_distance_loss_weight > 0 and self.class_unit_vectors.numel() > 0:
                class_vectors = self.class_unit_vectors.to(device=id_logits.device, dtype=id_logits.dtype)
                if hasattr(batch, "bright_anchor_y") and hasattr(batch, "batch"):
                    graph_idx = batch.batch[real_mask]
                    anchor_class = batch.bright_anchor_y.view(-1).to(id_logits.device)[graph_idx].long()
                    valid_anchor = anchor_class >= 0
                    distance_per_node = id_logits.new_zeros(true_class.shape[0])
                    if bool(valid_anchor.any()):
                        anchor_vectors = class_vectors[anchor_class[valid_anchor]]
                        true_vectors = class_vectors[true_class[valid_anchor]]
                        true_anchor_cosine = (true_vectors * anchor_vectors).sum(dim=1)
                        true_anchor_distance = ((1.0 - true_anchor_cosine.clamp(-1.0, 1.0)) * 0.5).clamp_min(0.0)

                        predicted_anchor_cosine = anchor_vectors @ class_vectors.t()
                        predicted_anchor_distance = (
                            (1.0 - predicted_anchor_cosine.clamp(-1.0, 1.0)) * 0.5
                        ).clamp_min(0.0)
                        distance_cost = torch.abs(predicted_anchor_distance - true_anchor_distance.unsqueeze(1))
                        distance_per_node[valid_anchor] = (prob[valid_anchor] * distance_cost).sum(dim=1)
                    distance_loss = self._reduce_node_loss(distance_per_node, batch, real_mask)
                else:
                    true_vectors = class_vectors[true_class]
                    cosine = true_vectors @ class_vectors.t()
                    distance_cost = ((1.0 - cosine.clamp(-1.0, 1.0)) * 0.5).clamp_min(0.0)
                    distance_loss = self._reduce_node_loss((prob * distance_cost).sum(dim=1), batch, real_mask)
            else:
                distance_loss = id_logits.sum() * 0.0

            if self.class_rank_loss_weight > 0 and self.class_mag_rank.numel() > 0:
                class_rank = self.class_mag_rank.to(device=id_logits.device, dtype=id_logits.dtype)
                rank_cost = torch.abs(class_rank.unsqueeze(0) - class_rank[true_class].unsqueeze(1))
                rank_loss = self._reduce_node_loss((prob * rank_cost).sum(dim=1), batch, real_mask)
            else:
                rank_loss = id_logits.sum() * 0.0
        else:
            id_loss = id_logits.sum() * 0.0
            distance_loss = id_logits.sum() * 0.0
            rank_loss = id_logits.sum() * 0.0
        total = (
            self.id_loss_weight * id_loss
            + self.class_distance_loss_weight * distance_loss
            + self.class_rank_loss_weight * rank_loss
        )
        return total, {
            "id_loss": float(id_loss.detach().cpu().item()),
            "class_distance_loss": float(distance_loss.detach().cpu().item()),
            "class_rank_loss": float(rank_loss.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
        }


@torch.no_grad()
def evaluate(
    loader,
    model,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    node_count = 0
    real_count = 0
    top1_ok = 0
    top5_ok = 0
    top10_ok = 0

    for batch in loader:
        batch = batch.to(device)
        id_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss, _ = criterion(id_logits, batch)

        n_nodes = int(batch.x.shape[0])
        node_count += n_nodes
        loss_sum += float(loss.detach().cpu().item()) * max(n_nodes, 1)

        real_mask = batch.real_mask.bool()
        if bool(real_mask.any()):
            logits_real = id_logits[real_mask]
            labels_real = batch.id_y[real_mask].long()
            real_n = int(labels_real.shape[0])
            real_count += real_n

            top1 = torch.topk(logits_real, k=1, dim=1).indices
            top5 = torch.topk(logits_real, k=min(5, logits_real.shape[1]), dim=1).indices
            top10 = torch.topk(logits_real, k=min(10, logits_real.shape[1]), dim=1).indices
            top1_ok += int((top1.squeeze(1) == labels_real).sum().item())
            top5_ok += int(top5.eq(labels_real.unsqueeze(1)).any(dim=1).sum().item())
            top10_ok += int(top10.eq(labels_real.unsqueeze(1)).any(dim=1).sum().item())

    return {
        "loss": float(loss_sum / max(node_count, 1)),
        "real_node_count": int(real_count),
        "top1_real": float(top1_ok / max(real_count, 1)),
        "top5_real": float(top5_ok / max(real_count, 1)),
        "top10_real": float(top10_ok / max(real_count, 1)),
    }


def load_checkpoint_with_fallback(path: Path, device: torch.device) -> Dict[str, object]:
    ckpt_path = path.expanduser().resolve()
    try:
        return torch.load(ckpt_path, map_location=device)
    except Exception as exc:
        log(f"Checkpoint requires full unpickling on this torch version. Retrying trusted load: {exc}")
        try:
            return torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location=device)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refactored paper-like GNN trainer (closed-set roll-group split within guide_star)")

    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--dataset-run", type=str, default=None)
    p.add_argument("--split-file", type=Path, default=None)

    p.add_argument("--runs-root", type=Path, default=Path("GNN") / "runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size-scenes", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--cache-chunks", type=int, default=2)
    p.add_argument("--log-every-batches", type=int, default=200)
    p.add_argument("--worker-timeout-sec", type=int, default=90)
    p.add_argument(
        "--class-scope",
        choices=("dataset", "train_candidates", "split_candidates"),
        default="dataset",
        help=(
            "Which star IDs become output classes. 'dataset' keeps the previous full manifest/dataset scope; "
            "'train_candidates' restricts outputs to stars that appear in the training candidate top-N; "
            "'split_candidates' uses the union of train/val/test candidate top-N for offline analysis."
        ),
    )
    p.add_argument(
        "--class-scope-top-n",
        type=int,
        default=None,
        help=(
            "Top-N points per scene used to define candidate classes for restricted class scopes. "
            "Defaults to --quad-combinations-top-n, else max(--top-n-choices), else all points."
        ),
    )

    p.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
        help=f"Deprecated compatibility flag. Ignored; graph degree is chosen per scene as {GRAPH_K_FORMULA}.",
    )
    p.add_argument(
        "--top-n-choices",
        type=str,
        default="4,5,6,7,8",
        help="Comma-separated top-N brightness truncation sizes.",
    )
    p.add_argument(
        "--top-n-mode",
        choices=("expand", "sample", "max"),
        default="expand",
        help=(
            "How to turn top-N choices into graph samples: "
            "expand creates one graph per scene per top-N choice; "
            "sample creates one deterministic random top-N graph per scene; "
            "max creates one graph per scene using the largest effective top-N."
        ),
    )
    p.add_argument(
        "--graph-connectivity",
        choices=GRAPH_CONNECTIVITY_CHOICES,
        default="fully",
        help=(
            "Graph edge construction. 'knn' uses the dynamic per-scene kNN degree; "
            "'fully' connects every node to every other node, useful for the Tetra-like N=4 baseline."
        ),
    )
    p.add_argument(
        "--node-feature-mode",
        choices=NODE_FEATURE_MODE_CHOICES,
        default="none",
        help=(
            "Node input features. Use 'none' for the Tetra-like no-node-feature baseline; "
            "the default keeps the previous magnitude-subtracted plus rank behaviour."
        ),
    )
    p.add_argument(
        "--edge-feature-mode",
        choices=EDGE_FEATURE_MODE_CHOICES,
        default="distance_diagonal",
        help=(
            "Edge input features. distance_max normalizes pair distance by the largest distance "
            "inside the graph; *_dmag also adds the current magnitude-difference feature."
        ),
    )
    p.add_argument(
        "--quad-combinations-top-n",
        type=int,
        default=None,
        help=(
            "If set, each scene is expanded into all 4-star combinations from the brightest N points. "
            "For example, 8 creates C(8, 4)=70 fully parametrized quad samples per scene."
        ),
    )
    p.add_argument(
        "--quad-combination-mode",
        choices=QUAD_COMBINATION_MODE_CHOICES,
        default="all",
        help=(
            "How to use --quad-combinations-top-n: 'all' creates every 4-star combination; "
            "'sample' creates one deterministic pseudo-random 4-star combination per scene; "
            "'balanced_sample' creates one combination per scene while balancing star_as_input_count."
        ),
    )
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument(
        "--max-neighbors",
        type=int,
        default=None,
        help=(
            "Fixed neighbour-slot width for edge_mlp. Defaults to 3 for quad-combination runs, "
            "or max(top_n_choices)-1 for fully connected variable top-N runs."
        ),
    )
    p.add_argument("--heads", type=int, default=HEADS)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument(
        "--model-backbone",
        choices=MODEL_BACKBONE_CHOICES,
        default="edge_mlp",
        help=(
            "Model backbone. 'gatv2' keeps the original PyG GATv2Conv model; "
            "'edge_mlp' builds explicit per-node inputs from node features, outgoing edge features, "
            "and neighbour node features."
        ),
    )

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)

    p.add_argument("--id-loss-weight", type=float, default=1.0)
    p.add_argument(
        "--class-distance-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a soft relative catalog-distance penalty: probability mass is penalized when "
            "the predicted star's distance to the graph's brightest real star differs from the true "
            "star's distance to that same anchor."
        ),
    )
    p.add_argument(
        "--class-rank-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a soft catalog-magnitude-rank penalty: probability mass assigned to stars with "
            "a very different catalog magnitude rank from the true class is penalized."
        ),
    )
    p.add_argument(
        "--loss-group-by-scene",
        action="store_true",
        help=(
            "Average graph losses inside each scene before averaging the batch. "
            "Use this when one scene is expanded into many top-N graphs or quad combinations."
        ),
    )

    p.add_argument("--early-stop-patience", type=int, default=40)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument(
        "--early-stop-monitor",
        choices=EARLY_STOP_MONITOR_CHOICES,
        default="val_loss",
        help=(
            "Validation metric used for best_checkpoint.pt and early stopping. "
            "Loss metrics are minimized; top-k metrics are maximized."
        ),
    )

    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default="auto")
    return p


def main(argv: List[str] | None = None) -> int:
    run_started_perf = time.perf_counter()
    args = parser().parse_args(argv)
    args_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    if (not args.eval_only) and args.epochs <= 0:
        raise ValueError("epochs must be > 0 unless --eval-only is set")
    if args.batch_size_scenes <= 0:
        raise ValueError("epochs and batch-size-scenes must be > 0")
    if args.eval_only and args.checkpoint is None:
        raise ValueError("--eval-only requires --checkpoint")

    ensure_pyg_available()
    from torch_geometric.loader import DataLoader

    set_seed(int(args.seed))
    top_n_choices = parse_top_n_choices(args.top_n_choices)
    if args.k_neighbors is not None:
        log(f"Ignoring legacy --k-neighbors={args.k_neighbors}; using dynamic graph degree {GRAPH_K_FORMULA}.")
    if sys.platform == "darwin":
        # Avoid OpenMP deadlocks observed on macOS with torch + PyG workloads.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        log("macOS detected: forcing torch num_threads=1 and num_interop_threads=1.")
    device = choose_device(args.device)

    dataset_dir = resolve_dataset_dir(args.dataset_dir, args.dataset_run)
    split_path = resolve_split_file(args.split_file)

    run_dir = next_run_dir(args.runs_root.expanduser().resolve(), run_name=args.run_name)
    history_path = run_dir / "train_history.jsonl"
    summary_path = run_dir / "train_summary.json"
    best_ckpt = run_dir / "best_checkpoint.pt"
    last_ckpt = run_dir / "last_checkpoint.pt"

    manifest = load_manifest(dataset_dir)
    paths = chunk_paths(dataset_dir, manifest)
    width, height = image_size_from_manifest(manifest)
    run_meta = manifest.get("run", {}) if isinstance(manifest.get("run"), dict) else {}
    run_params = run_meta.get("parameters", {}) if isinstance(run_meta.get("parameters"), dict) else {}
    guide_count = len(manifest.get("guide_star_indices", [])) if isinstance(manifest.get("guide_star_indices"), list) else -1
    log(
        "Dataset manifest: "
        f"run={run_meta.get('name', dataset_dir.name)} "
        f"guide_stars={run_params.get('guide_stars', 'unknown')} "
        f"guide_count={guide_count} "
        f"dataset_seed={run_params.get('seed', 'unknown')}"
    )
    train_refs = load_split_refs(split_path, "train")
    val_refs = load_split_refs(split_path, "val")
    test_refs = load_split_refs(split_path, "test")
    if args.eval_only:
        if not val_refs and not test_refs:
            raise RuntimeError("Eval-only mode needs val and/or test scenes in split")
    else:
        if not train_refs:
            raise RuntimeError("Train split has zero scenes")
        if not val_refs:
            raise RuntimeError("Val split has zero scenes")
        if not test_refs:
            raise RuntimeError("Test split has zero scenes")

    manifest_scope_star_ids = manifest_class_star_ids(manifest)
    checkpoint_payload = None
    class_scope_top_n = int(args.class_scope_top_n or 0)
    if class_scope_top_n <= 0:
        if args.quad_combinations_top_n:
            class_scope_top_n = int(args.quad_combinations_top_n)
        elif top_n_choices:
            class_scope_top_n = int(max(top_n_choices))
        else:
            class_scope_top_n = 0
    if args.eval_only:
        checkpoint_payload = load_checkpoint_with_fallback(args.checkpoint, device)
        class_to_star_id = np.asarray(checkpoint_payload["class_to_star_id"], dtype=np.int64)
        star_id_to_class = {int(star_id): i for i, star_id in enumerate(class_to_star_id.tolist())}
    elif args.class_scope == "dataset":
        class_to_star_id, star_id_to_class = build_star_mapping(paths, manifest_scope_star_ids)
    else:
        if args.class_scope == "train_candidates":
            scope_refs = train_refs
        elif args.class_scope == "split_candidates":
            scope_refs = list(train_refs) + list(val_refs) + list(test_refs)
        else:
            raise ValueError(f"Unsupported class scope: {args.class_scope}")
        class_to_star_id = collect_candidate_star_ids_from_refs(
            paths=paths,
            refs=scope_refs,
            top_n=class_scope_top_n if class_scope_top_n > 0 else None,
        )
        if class_to_star_id.size == 0:
            raise RuntimeError(f"No class IDs found for class_scope={args.class_scope}")
        star_id_to_class = {int(star_id): i for i, star_id in enumerate(class_to_star_id.tolist())}
        log(
            f"Class scope {args.class_scope}: "
            f"{int(class_to_star_id.shape[0])} classes from top{class_scope_top_n if class_scope_top_n > 0 else 'all'} candidates"
        )

    train_ds = SceneGraphDataset(
        chunk_paths=paths,
        refs=train_refs,
        star_id_to_class=star_id_to_class,
        width=width,
        height=height,
        cache_chunks=args.cache_chunks,
        top_n_choices=top_n_choices,
        top_n_mode=args.top_n_mode,
        top_n_seed=args.seed,
        graph_connectivity=args.graph_connectivity,
        node_feature_mode=args.node_feature_mode,
        edge_feature_mode=args.edge_feature_mode,
        quad_combinations_top_n=args.quad_combinations_top_n,
        quad_combination_mode=args.quad_combination_mode,
    ) if train_refs else None
    val_ds = SceneGraphDataset(
        chunk_paths=paths,
        refs=val_refs,
        star_id_to_class=star_id_to_class,
        width=width,
        height=height,
        cache_chunks=max(1, args.cache_chunks // 2),
        top_n_choices=top_n_choices,
        top_n_mode=args.top_n_mode,
        top_n_seed=args.seed,
        graph_connectivity=args.graph_connectivity,
        node_feature_mode=args.node_feature_mode,
        edge_feature_mode=args.edge_feature_mode,
        quad_combinations_top_n=args.quad_combinations_top_n,
        quad_combination_mode=args.quad_combination_mode,
    ) if val_refs else None
    test_ds = SceneGraphDataset(
        chunk_paths=paths,
        refs=test_refs,
        star_id_to_class=star_id_to_class,
        width=width,
        height=height,
        cache_chunks=max(1, args.cache_chunks // 2),
        top_n_choices=top_n_choices,
        top_n_mode=args.top_n_mode,
        top_n_seed=args.seed,
        graph_connectivity=args.graph_connectivity,
        node_feature_mode=args.node_feature_mode,
        edge_feature_mode=args.edge_feature_mode,
        quad_combinations_top_n=args.quad_combinations_top_n,
        quad_combination_mode=args.quad_combination_mode,
    ) if test_refs else None

    star_as_input_count_files = {
        "train": write_star_as_input_count_csv(run_dir, "train", train_ds),
        "val": write_star_as_input_count_csv(run_dir, "val", val_ds),
        "test": write_star_as_input_count_csv(run_dir, "test", test_ds),
    }
    star_as_input_count_summaries = {
        "train": train_ds.star_as_input_count_summary() if train_ds is not None else None,
        "val": val_ds.star_as_input_count_summary() if val_ds is not None else None,
        "test": test_ds.star_as_input_count_summary() if test_ds is not None else None,
    }
    star_as_input_count_file_json = {
        split_name: str(path)
        for split_name, path in star_as_input_count_files.items()
        if path is not None
    }

    def make_loader(dataset, shuffle: bool, workers: int):
        timeout = int(args.worker_timeout_sec) if workers > 0 else 0
        return DataLoader(
            dataset,
            batch_size=args.batch_size_scenes,
            shuffle=shuffle,
            num_workers=workers,
            persistent_workers=bool(workers > 0),
            timeout=timeout,
        )

    effective_workers = int(args.num_workers)
    if sys.platform == "darwin" and effective_workers > 0:
        log("macOS detected: forcing num_workers=0 for GNN DataLoader stability.")
        effective_workers = 0

    train_loader = make_loader(train_ds, shuffle=False, workers=effective_workers) if train_ds is not None else None
    val_loader = make_loader(val_ds, shuffle=False, workers=effective_workers) if val_ds is not None else None
    test_loader = make_loader(test_ds, shuffle=False, workers=effective_workers) if test_ds is not None else None

    if effective_workers > 0:
        try:
            probe_ds = train_ds or val_ds or test_ds
            if probe_ds is None:
                raise RuntimeError("No dataset available for worker probe")
            probe_loader = DataLoader(
                probe_ds,
                batch_size=1,
                shuffle=False,
                num_workers=effective_workers,
                persistent_workers=True,
                timeout=int(args.worker_timeout_sec),
            )
            _ = next(iter(probe_loader))
            log(f"Worker probe OK with num_workers={effective_workers}")
        except Exception as exc:
            log(f"Worker probe failed with num_workers={effective_workers}: {exc}")
            log("Falling back to num_workers=0 to avoid startup deadlock.")
            effective_workers = 0
            train_loader = make_loader(train_ds, shuffle=False, workers=effective_workers) if train_ds is not None else None
            val_loader = make_loader(val_ds, shuffle=False, workers=effective_workers) if val_ds is not None else None
            test_loader = make_loader(test_ds, shuffle=False, workers=effective_workers) if test_ds is not None else None

    sample_ds = train_ds or val_ds or test_ds
    if sample_ds is None:
        raise RuntimeError("No scenes available to build graph samples")
    sample = sample_ds[0]
    in_dim = int(sample.x.shape[1])
    edge_dim = int(sample.edge_attr.shape[1])
    num_id_classes = int(class_to_star_id.shape[0])

    if checkpoint_payload is not None:
        model_meta = checkpoint_payload.get("model", {})
        model_backbone = str(model_meta.get("name", model_meta.get("backbone", args.model_backbone)))
        if model_backbone not in MODEL_BACKBONE_CHOICES:
            model_backbone = "gatv2"
        hidden_dim = int(model_meta.get("hidden_dim", args.hidden_dim))
        num_layers = int(model_meta.get("num_layers", args.num_layers))
        heads = int(model_meta.get("heads", args.heads))
        dropout = float(model_meta.get("dropout", args.dropout))
        num_id_classes = int(model_meta.get("num_id_classes", num_id_classes))
        max_neighbors = int(model_meta.get("max_neighbors", args.max_neighbors or 3))
    else:
        model_backbone = str(args.model_backbone)
        hidden_dim = int(args.hidden_dim)
        num_layers = int(args.num_layers)
        heads = int(args.heads)
        dropout = float(args.dropout)
        max_neighbors = (
            int(args.max_neighbors)
            if args.max_neighbors is not None
            else infer_model_max_neighbors(
                graph_connectivity=str(args.graph_connectivity),
                top_n_choices=top_n_choices,
                quad_combinations_top_n=args.quad_combinations_top_n,
                sample_node_count=int(sample.x.shape[0]),
            )
        )

    model = make_star_model(
        model_backbone=model_backbone,
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        num_id_classes=num_id_classes,
        max_neighbors=max_neighbors,
    ).to(device)
    if (not args.eval_only) and args.init_checkpoint is not None:
        init_payload = load_checkpoint_with_fallback(args.init_checkpoint, device)
        model.load_state_dict(init_payload["model_state_dict"], strict=False)
        log(f"Initialized model weights from checkpoint: {args.init_checkpoint.expanduser().resolve()}")

    class_unit_vectors = None
    class_mag_rank = None
    if float(args.class_distance_loss_weight) > 0.0 or float(args.class_rank_loss_weight) > 0.0:
        database_path = database_path_from_manifest(manifest)
        class_unit_vectors, class_mag_rank = build_class_loss_features(
            database_path=database_path,
            class_to_star_id=class_to_star_id,
            device=device,
        )
        log(f"Class auxiliary loss features: database={database_path}")

    criterion = MultiTaskLoss(
        id_loss_weight=args.id_loss_weight,
        group_by_scene=args.loss_group_by_scene,
        class_distance_loss_weight=args.class_distance_loss_weight,
        class_rank_loss_weight=args.class_rank_loss_weight,
        class_unit_vectors=class_unit_vectors,
        class_mag_rank=class_mag_rank,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log(f"Run dir: {run_dir}")
    log(f"Dataset: {dataset_dir}")
    log(f"Split file: {split_path}")
    log(f"Device: {device}")
    log(f"Scenes train/val/test: {len(train_refs)}/{len(val_refs)}/{len(test_refs)}")
    log(
        "Graph samples train/val/test: "
        f"{len(train_ds) if train_ds is not None else 0}/"
        f"{len(val_ds) if val_ds is not None else 0}/"
        f"{len(test_ds) if test_ds is not None else 0}"
    )
    if args.eval_only:
        log(f"Classes (checkpoint): {num_id_classes}")
    elif args.class_scope != "dataset":
        log(f"Classes ({args.class_scope}): {num_id_classes}")
    elif manifest_scope_star_ids is not None:
        log(f"Classes (manifest scope): {num_id_classes}")
    else:
        log(f"Classes (real stars): {num_id_classes}")
    log(f"Top-N choices: {list(top_n_choices) if top_n_choices else ['all']}")
    log(f"Top-N mode: {args.top_n_mode}")
    log(f"Graph connectivity: {args.graph_connectivity}")
    log(f"Model backbone: {model_backbone}")
    log(f"Model max_neighbors: {int(max_neighbors)}")
    log(f"Node feature mode: {args.node_feature_mode}")
    log(f"Edge feature mode: {args.edge_feature_mode}")
    log(
        "Loss weights: "
        f"id={float(args.id_loss_weight):.3g} "
        f"class_distance={float(args.class_distance_loss_weight):.3g} "
        f"class_rank={float(args.class_rank_loss_weight):.3g}"
    )
    if args.quad_combinations_top_n:
        if args.quad_combination_mode == "all":
            log(f"Quad-combination mode: all C({args.quad_combinations_top_n}, 4) combinations per scene")
        elif args.quad_combination_mode == "balanced_sample":
            log(
                "Quad-combination mode: one balanced 4-star combination "
                f"from top{args.quad_combinations_top_n} per scene"
            )
        else:
            log(f"Quad-combination mode: one sampled 4-star combination from top{args.quad_combinations_top_n} per scene")
    for split_name, balance in star_as_input_count_summaries.items():
        if balance is None:
            continue
        log(
            f"{split_name} star_as_input_count: "
            f"selected/candidates/all={balance['selected_stars']}/{balance['candidate_stars']}/{balance['all_stars']} "
            f"selected_total={balance['total_star_inputs']} "
            f"min/mean/max={balance['star_as_input_count_min']}/"
            f"{balance['star_as_input_count_mean']:.3f}/"
            f"{balance['star_as_input_count_max']} "
            f"top8_total={balance['top8_candidate_total']} "
            f"rate_min/mean/max={balance['selection_rate_min']:.3f}/"
            f"{balance['selection_rate_mean']:.3f}/"
            f"{balance['selection_rate_max']:.3f}"
        )
    log(f"Loss grouped by scene: {bool(args.loss_group_by_scene)}")
    log(
        "Early stopping: "
        f"monitor={args.early_stop_monitor} "
        f"mode={'min' if lower_is_better_metric(str(args.early_stop_monitor)) else 'max'} "
        f"patience={int(args.early_stop_patience)} "
        f"min_delta={float(args.early_stop_min_delta):.3g}"
    )
    if args.graph_connectivity == "knn":
        log(f"Graph degree mode: dynamic {GRAPH_K_FORMULA}")
    else:
        log("Graph degree mode: not used for fully connected graphs")

    if args.eval_only:
        if checkpoint_payload is None:
            raise RuntimeError("Checkpoint payload missing in eval-only mode")
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        val_metrics = (
            evaluate(
                loader=val_loader,
                model=model,
                criterion=criterion,
                device=device,
            )
            if val_loader is not None
            else None
        )
        test_metrics = (
            evaluate(
                loader=test_loader,
                model=model,
                criterion=criterion,
                device=device,
            )
            if test_loader is not None
            else None
        )
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "eval_only",
            "run_dir": str(run_dir),
            "dataset_dir": str(dataset_dir),
            "split_file": str(split_path),
            "checkpoint": str(args.checkpoint.expanduser().resolve() if args.checkpoint else ""),
            "device": str(device),
            "width": int(width),
            "height": int(height),
            "top_n_choices": list(top_n_choices),
            "top_n_mode": str(args.top_n_mode),
            "graph_connectivity": str(args.graph_connectivity),
            "model_backbone": str(model_backbone),
            "node_feature_mode": str(args.node_feature_mode),
            "edge_feature_mode": str(args.edge_feature_mode),
            "quad_combinations_top_n": int(args.quad_combinations_top_n or 0),
            "quad_combination_mode": str(args.quad_combination_mode),
            "class_scope": str(args.class_scope),
            "class_scope_top_n": int(class_scope_top_n),
            "star_as_input_count": star_as_input_count_summaries,
            "star_as_input_count_files": star_as_input_count_file_json,
            "loss_group_by_scene": bool(args.loss_group_by_scene),
            "graph_k_mode": "dynamic" if args.graph_connectivity == "knn" else "not_used",
            "graph_k_formula": GRAPH_K_FORMULA,
            "scenes": {"train": len(train_refs), "val": len(val_refs), "test": len(test_refs)},
            "graph_samples": {
                "train": len(train_ds) if train_ds is not None else 0,
                "val": len(val_ds) if val_ds is not None else 0,
                "test": len(test_ds) if test_ds is not None else 0,
            },
            "num_id_classes": int(num_id_classes),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "duration_min": float((time.perf_counter() - run_started_perf) / 60.0),
            "args": args_json,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        log(f"Eval finished. Summary: {summary_path}")
        if test_metrics is not None:
            log(
                "Test metrics: "
                f"top1={test_metrics['top1_real']:.4f}, "
                f"top5={test_metrics['top5_real']:.4f}, "
                f"top10={test_metrics['top10_real']:.4f}"
            )
        return 0

    monitor_metric = str(args.early_stop_monitor)
    monitor_lower_is_better = lower_is_better_metric(monitor_metric)
    best_metric = float("inf") if monitor_lower_is_better else -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, object]] = []
    base_train_refs = list(train_refs)

    for epoch in range(1, int(args.epochs) + 1):
        # Deterministic per-epoch shuffle while keeping DataLoader(shuffle=False).
        # This avoids long contiguous blocks of the same guide class.
        if train_ds is not None and len(base_train_refs) > 1:
            order = np.random.default_rng(int(args.seed) + int(epoch)).permutation(len(base_train_refs))
            train_ds.refs = [base_train_refs[int(i)] for i in order]

        log(f"Epoch {epoch:03d}/{args.epochs} started")
        model.train()

        train_loss_sum = 0.0
        train_nodes = 0
        train_id_loss_sum = 0.0
        batch_count = len(train_loader)
        batch_idx = 0

        for batch in train_loader:
            batch_idx += 1
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            id_logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss, comps = criterion(id_logits, batch)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
            optimizer.step()

            n = int(batch.x.shape[0])
            train_nodes += n
            train_loss_sum += float(comps["total_loss"]) * max(n, 1)
            train_id_loss_sum += float(comps["id_loss"]) * max(n, 1)
            if args.log_every_batches > 0 and (batch_idx == 1 or batch_idx % int(args.log_every_batches) == 0 or batch_idx == batch_count):
                avg_loss = float(train_loss_sum / max(train_nodes, 1))
                log(f"Epoch {epoch:03d} batch {batch_idx}/{batch_count} avg_train_loss={avg_loss:.4f}")

        log(f"Epoch {epoch:03d} train finished. Running validation...")
        val_metrics = evaluate(
            loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
        )

        epoch_row: Dict[str, object] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss_sum / max(train_nodes, 1)),
            "train_id_loss": float(train_id_loss_sum / max(train_nodes, 1)),
            "val_loss": float(val_metrics["loss"]),
            "val_top1_real": float(val_metrics["top1_real"]),
            "val_top5_real": float(val_metrics["top5_real"]),
            "val_top10_real": float(val_metrics["top10_real"]),
        }
        history.append(epoch_row)
        with history_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(epoch_row) + "\n")

        metric = float(epoch_row[monitor_metric])
        min_delta = float(args.early_stop_min_delta)
        if monitor_lower_is_better:
            improved = metric < (best_metric - min_delta)
        else:
            improved = metric > (best_metric + min_delta)

        payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "model": {
                "name": str(model_backbone),
                "backbone": str(model_backbone),
                "in_dim": in_dim,
                "edge_dim": edge_dim,
                "hidden_dim": int(hidden_dim),
                "num_layers": int(num_layers),
                "heads": int(heads),
                "dropout": float(dropout),
                "num_id_classes": int(num_id_classes),
                "max_neighbors": int(max_neighbors),
            },
            "train_args": args_json,
            "dataset_dir": str(dataset_dir),
            "split_file": str(split_path),
            "class_to_star_id": class_to_star_id.astype(np.int64),
            "data_config": {
                "top_n_choices": list(top_n_choices),
                "top_n_mode": str(args.top_n_mode),
                "graph_connectivity": str(args.graph_connectivity),
                "model_backbone": str(model_backbone),
                "node_feature_mode": str(args.node_feature_mode),
                "edge_feature_mode": str(args.edge_feature_mode),
                "quad_combinations_top_n": int(args.quad_combinations_top_n or 0),
                "quad_combination_mode": str(args.quad_combination_mode),
                "class_scope": str(args.class_scope),
                "class_scope_top_n": int(class_scope_top_n),
                "star_as_input_count": star_as_input_count_summaries,
                "loss_group_by_scene": bool(args.loss_group_by_scene),
                "graph_k_mode": "dynamic" if args.graph_connectivity == "knn" else "not_used",
                "graph_k_formula": GRAPH_K_FORMULA,
            },
            "metrics": {
                "val": val_metrics,
                "train_loss": epoch_row["train_loss"],
            },
        }
        torch.save(payload, last_ckpt)

        if improved:
            best_metric = metric
            best_epoch = int(epoch)
            bad_epochs = 0
            torch.save(payload, best_ckpt)
        else:
            bad_epochs += 1

        log(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_top1_real={val_metrics['top1_real']:.4f} "
            f"val_top10_real={val_metrics['top10_real']:.4f} "
            f"monitor_{monitor_metric}={metric:.6f} "
            f"bad_epochs={bad_epochs}"
        )

        if args.early_stop_patience > 0 and bad_epochs >= int(args.early_stop_patience):
            log(f"Early stop triggered at epoch {epoch}")
            break

    if best_ckpt.exists():
        best = load_checkpoint_with_fallback(best_ckpt, device)
        model.load_state_dict(best["model_state_dict"])

    test_metrics = evaluate(
        loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
    )
    duration_min = float((time.perf_counter() - run_started_perf) / 60.0)
    epochs_run = int(history[-1]["epoch"]) if history else 0
    best_monitor_value = float(best_metric) if best_epoch > 0 and not math.isinf(best_metric) else 0.0
    best_val_loss_epoch, best_val_loss = best_history_value(history, "val_loss")
    best_val_top1_epoch, best_val_top1 = best_history_value(history, "val_top1_real")
    best_val_top5_epoch, best_val_top5 = best_history_value(history, "val_top5_real")
    best_val_top10_epoch, best_val_top10 = best_history_value(history, "val_top10_real")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "split_file": str(split_path),
        "device": str(device),
        "width": int(width),
        "height": int(height),
        "top_n_choices": list(top_n_choices),
        "top_n_mode": str(args.top_n_mode),
        "graph_connectivity": str(args.graph_connectivity),
        "model_backbone": str(model_backbone),
        "max_neighbors": int(max_neighbors),
        "node_feature_mode": str(args.node_feature_mode),
        "edge_feature_mode": str(args.edge_feature_mode),
        "quad_combinations_top_n": int(args.quad_combinations_top_n or 0),
        "quad_combination_mode": str(args.quad_combination_mode),
        "class_scope": str(args.class_scope),
        "class_scope_top_n": int(class_scope_top_n),
        "star_as_input_count": star_as_input_count_summaries,
        "star_as_input_count_files": star_as_input_count_file_json,
        "loss_group_by_scene": bool(args.loss_group_by_scene),
        "graph_k_mode": "dynamic" if args.graph_connectivity == "knn" else "not_used",
        "graph_k_formula": GRAPH_K_FORMULA,
        "scenes": {"train": len(train_refs), "val": len(val_refs), "test": len(test_refs)},
        "graph_samples": {
            "train": len(train_ds) if train_ds is not None else 0,
            "val": len(val_ds) if val_ds is not None else 0,
            "test": len(test_ds) if test_ds is not None else 0,
        },
        "num_id_classes": int(num_id_classes),
        "best_epoch": int(best_epoch),
        "best_monitor_metric": monitor_metric,
        "best_monitor_mode": "min" if monitor_lower_is_better else "max",
        "best_monitor_value": best_monitor_value,
        "best_val_loss_epoch": int(best_val_loss_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_top1_epoch": int(best_val_top1_epoch),
        "best_val_top1_real": float(best_val_top1),
        "best_val_top5_epoch": int(best_val_top5_epoch),
        "best_val_top5_real": float(best_val_top5),
        "best_val_top10_epoch": int(best_val_top10_epoch),
        "best_val_top10_real": float(best_val_top10),
        "epochs_run": int(epochs_run),
        "duration_min": duration_min,
        "min_per_epoch": float(duration_min / max(epochs_run, 1)),
        "test_metrics": test_metrics,
        "args": args_json,
        "files": {
            "history": str(history_path),
            "best_checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log(f"Training finished. Summary: {summary_path}")
    log(
        f"Best epoch: {best_epoch}, "
        f"{monitor_metric}={summary['best_monitor_value']:.6f} "
        f"({summary['best_monitor_mode']})"
    )
    log(
        "Test metrics: "
        f"top1={test_metrics['top1_real']:.4f}, "
        f"top5={test_metrics['top5_real']:.4f}, "
        f"top10={test_metrics['top10_real']:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
