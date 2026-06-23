#!/usr/bin/env python3
"""Run a trained GNN checkpoint on one image and print node-level predictions."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
import torch

import tetra3
from GNN.GNN import (
    EDGE_FEATURE_MODE_CHOICES,
    GRAPH_CONNECTIVITY_CHOICES,
    MODEL_BACKBONE_CHOICES,
    NODE_FEATURE_MODE_CHOICES,
    QUAD_COMBINATION_MODE_CHOICES,
    build_graph_inputs,
    choose_graph_k_neighbors,
    choose_device,
    effective_top_n_choices,
    load_checkpoint_with_fallback,
    make_star_model,
    parse_top_n_choices,
    quad_combo_by_index,
)


DEFAULT_TOP_N_CHOICES = (4, 5, 6, 7, 8)
DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "tetra3" / "data" / "default_database.npz"


def normalize_catalog_id(value) -> int | tuple[int, ...]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return int(arr.item())
    return tuple(int(v) for v in arr.reshape(-1).tolist())


def format_catalog_id(value: int | tuple[int, ...]) -> str:
    if isinstance(value, tuple):
        return "(" + ",".join(str(v) for v in value) + ")"
    return str(value)


def build_catalog_lookup(database_path: Path) -> np.ndarray:
    with np.load(database_path, allow_pickle=False) as data:
        if "star_catalog_IDs" not in data:
            raise RuntimeError(f"Database {database_path} does not contain star_catalog_IDs")
        return np.asarray(data["star_catalog_IDs"])


def solve_tetra3_matches(image_path: Path) -> tuple[np.ndarray, list[int | tuple[int, ...]]]:
    solver = tetra3.Tetra3()
    with Image.open(image_path) as img:
        solution = solver.solve_from_image(
            img,
            distortion=[-0.2, 0.1],
            return_matches=True,
        )

    if not solution:
        return np.empty((0, 2), dtype=np.float32), []

    matched_centroids = np.asarray(solution.get("matched_centroids", []), dtype=np.float32)
    matched_cat_ids = [normalize_catalog_id(v) for v in (solution.get("matched_catID") or [])]
    return matched_centroids, matched_cat_ids


def find_nearest_match(
    centroid_yx: np.ndarray,
    matched_centroids: np.ndarray,
    matched_cat_ids: list[int | tuple[int, ...]],
    max_dist_px: float = 6.0,
) -> int | tuple[int, ...] | None:
    if matched_centroids.size == 0 or not matched_cat_ids:
        return None

    distances = np.linalg.norm(matched_centroids - centroid_yx[None, :], axis=1)
    best_idx = int(np.argmin(distances))
    if float(distances[best_idx]) > max_dist_px:
        return None
    return matched_cat_ids[best_idx]


def find_match_rank(
    target_catalog_id: int | tuple[int, ...] | None,
    predicted_catalog_ids: list[int | tuple[int, ...]],
) -> int | None:
    if target_catalog_id is None:
        return None
    for rank, catalog_id in enumerate(predicted_catalog_ids, start=1):
        if catalog_id == target_catalog_id:
            return rank
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a GNN checkpoint on one TIFF image and print top-k classes for the brightest centroids."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--k-neighbors",
        type=int,
        default=None,
        help="Deprecated compatibility flag. Ignored; graph degree is chosen dynamically per graph.",
    )
    p.add_argument("--top-n-choices", type=str, default=None)
    p.add_argument(
        "--graph-connectivity",
        choices=GRAPH_CONNECTIVITY_CHOICES,
        default=None,
        help="Override checkpoint graph connectivity. Defaults to the checkpoint setting.",
    )
    p.add_argument(
        "--node-feature-mode",
        choices=NODE_FEATURE_MODE_CHOICES,
        default=None,
        help="Override checkpoint node feature mode. Defaults to the checkpoint setting.",
    )
    p.add_argument(
        "--edge-feature-mode",
        choices=EDGE_FEATURE_MODE_CHOICES,
        default=None,
        help="Override checkpoint edge feature mode. Defaults to the checkpoint setting.",
    )
    p.add_argument(
        "--quad-combinations-top-n",
        type=int,
        default=None,
        help="Override checkpoint quad-combination mode. Example: 8 evaluates all C(8, 4) quads.",
    )
    p.add_argument(
        "--quad-combination-mode",
        choices=QUAD_COMBINATION_MODE_CHOICES,
        default=None,
        help=(
            "Override checkpoint quad-combination mode. For real-image reporting, 'all' is usually best; "
            "'balanced_sample' is evaluated over all combinations because there is no dataset-level counter."
        ),
    )
    p.add_argument("--brightest-k", type=int, default=5)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    return p.parse_args()


def resolve_top_n_choices(payload: dict, raw: str | None) -> Tuple[int, ...]:
    if raw is not None:
        return parse_top_n_choices(raw)

    data_config = payload.get("data_config", {}) if isinstance(payload.get("data_config"), dict) else {}
    values = data_config.get("top_n_choices")
    if isinstance(values, list) and values:
        return tuple(int(v) for v in values)

    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    raw_train = train_args.get("top_n_choices")
    if isinstance(raw_train, str) and raw_train.strip():
        return parse_top_n_choices(raw_train)

    return DEFAULT_TOP_N_CHOICES


def resolve_graph_connectivity(payload: dict, raw: str | None) -> str:
    if raw is not None:
        return raw

    data_config = payload.get("data_config", {}) if isinstance(payload.get("data_config"), dict) else {}
    value = data_config.get("graph_connectivity")
    if isinstance(value, str) and value in GRAPH_CONNECTIVITY_CHOICES:
        return value

    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    value = train_args.get("graph_connectivity")
    if isinstance(value, str) and value in GRAPH_CONNECTIVITY_CHOICES:
        return value

    return "knn"


def resolve_node_feature_mode(payload: dict, raw: str | None) -> str:
    if raw is not None:
        return raw

    data_config = payload.get("data_config", {}) if isinstance(payload.get("data_config"), dict) else {}
    value = data_config.get("node_feature_mode")
    if isinstance(value, str) and value in NODE_FEATURE_MODE_CHOICES:
        return value

    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    value = train_args.get("node_feature_mode")
    if isinstance(value, str) and value in NODE_FEATURE_MODE_CHOICES:
        return value

    return "magnitude_subtracted_rank"


def resolve_edge_feature_mode(payload: dict, raw: str | None) -> str:
    if raw is not None:
        return raw

    data_config = payload.get("data_config", {}) if isinstance(payload.get("data_config"), dict) else {}
    value = data_config.get("edge_feature_mode")
    if isinstance(value, str) and value in EDGE_FEATURE_MODE_CHOICES:
        return value

    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    value = train_args.get("edge_feature_mode")
    if isinstance(value, str) and value in EDGE_FEATURE_MODE_CHOICES:
        return value

    return "distance_diagonal_dmag"


def resolve_quad_combinations_top_n(payload: dict, raw: int | None) -> int:
    if raw is not None:
        return int(raw)

    data_config = payload.get("data_config", {}) if isinstance(payload.get("data_config"), dict) else {}
    value = data_config.get("quad_combinations_top_n")
    if value is not None:
        return int(value)

    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    value = train_args.get("quad_combinations_top_n")
    if value is not None:
        return int(value)

    return 0


def resolve_quad_combination_mode(payload: dict, raw: str | None) -> str:
    if raw is not None:
        return raw

    data_config = payload.get("data_config", {}) if isinstance(payload.get("data_config"), dict) else {}
    value = data_config.get("quad_combination_mode")
    if isinstance(value, str) and value in QUAD_COMBINATION_MODE_CHOICES:
        return value

    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    value = train_args.get("quad_combination_mode")
    if isinstance(value, str) and value in QUAD_COMBINATION_MODE_CHOICES:
        return value

    return "all"


def resolve_seed(payload: dict) -> int:
    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    value = train_args.get("seed")
    if value is None:
        return 12345
    return int(value)


def load_sorted_centroids(image_path: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
        centroids, moments = tetra3.get_centroids_from_image(img, return_moments=True)

    point_yx = np.asarray(centroids, dtype=np.float32)
    if point_yx.ndim != 2 or point_yx.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32), width, height

    flux_sum = np.asarray(moments[0], dtype=np.float32)
    flux_sum = np.clip(flux_sum, 1e-6, None)
    order = np.argsort(-flux_sum, kind="stable")
    return point_yx[order], flux_sum[order], width, height


def topk_catalog_predictions(
    class_indices: Iterable[int],
    class_to_star_id: np.ndarray,
    catalog_lookup: np.ndarray,
) -> list[int | tuple[int, ...]]:
    output: list[int | tuple[int, ...]] = []
    for class_idx in class_indices:
        star_id = int(class_to_star_id[int(class_idx)])
        output.append(normalize_catalog_id(catalog_lookup[star_id]))
    return output


def main() -> int:
    args = parse_args()

    # tetra3 currently uses np.math in a few paths; newer numpy versions removed this alias.
    np.math = math  # type: ignore[attr-defined]

    device = choose_device(args.device)
    payload = load_checkpoint_with_fallback(args.checkpoint, device)

    model_meta = payload.get("model", {}) if isinstance(payload, dict) else {}
    model_backbone = str(model_meta.get("name", model_meta.get("backbone", "edge_mlp")))
    if model_backbone not in MODEL_BACKBONE_CHOICES:
        model_backbone = "edge_mlp"
    class_to_star_id = np.asarray(payload["class_to_star_id"], dtype=np.int64)
    top_n_choices = resolve_top_n_choices(payload, args.top_n_choices)
    graph_connectivity = resolve_graph_connectivity(payload, args.graph_connectivity)
    node_feature_mode = resolve_node_feature_mode(payload, args.node_feature_mode)
    edge_feature_mode = resolve_edge_feature_mode(payload, args.edge_feature_mode)
    quad_combinations_top_n = resolve_quad_combinations_top_n(payload, args.quad_combinations_top_n)
    quad_combination_mode = resolve_quad_combination_mode(payload, args.quad_combination_mode)
    seed = resolve_seed(payload)
    catalog_lookup = build_catalog_lookup(args.database.expanduser().resolve())

    model = make_star_model(
        model_backbone=model_backbone,
        in_dim=int(model_meta.get("in_dim", 2)),
        edge_dim=int(model_meta.get("edge_dim", 2)),
        hidden_dim=int(model_meta.get("hidden_dim", 128)),
        num_layers=int(model_meta.get("num_layers", 3)),
        heads=int(model_meta.get("heads", 4)),
        dropout=float(model_meta.get("dropout", 0.2)),
        num_id_classes=int(model_meta.get("num_id_classes", class_to_star_id.shape[0])),
        max_neighbors=int(model_meta.get("max_neighbors", 3)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    image_path = args.image.expanduser().resolve()
    point_yx, flux_sum, width, height = load_sorted_centroids(image_path)
    if point_yx.shape[0] == 0:
        raise RuntimeError(f"No centroids found in {image_path}")

    matched_centroids, matched_cat_ids = solve_tetra3_matches(image_path)
    matched_cat_id_set = set(matched_cat_ids)

    brightest_k = max(1, min(int(args.brightest_k), int(point_yx.shape[0])))
    topk = max(1, int(args.topk))
    point_mag = (-2.5 * np.log10(flux_sum)).astype(np.float32)

    logits_sum = torch.zeros((brightest_k, class_to_star_id.shape[0]), dtype=torch.float32, device=device)
    logits_count = torch.zeros((brightest_k,), dtype=torch.float32, device=device)
    graph_k_by_top_n: dict[int, int] = {}
    quad_combination_count = 0

    if quad_combinations_top_n > 0:
        source_top_n = min(int(quad_combinations_top_n), int(point_yx.shape[0]))
        if source_top_n < 4:
            raise ValueError(f"Need at least 4 centroids for quad-combination eval, got {source_top_n}")
        if brightest_k > source_top_n:
            raise ValueError(
                f"brightest-k={brightest_k} is larger than quad source top_n={source_top_n}. "
                "Use a smaller --brightest-k or a larger --quad-combinations-top-n."
            )
        top_n_effective = (source_top_n,)
        graph_k_by_top_n[source_top_n] = int(choose_graph_k_neighbors(4))

        with torch.no_grad():
            if quad_combination_mode == "sample":
                combo_count = math.comb(source_top_n, 4)
                rng = np.random.default_rng(int(seed) * 1000003 + source_top_n)
                combos_iter = (quad_combo_by_index(source_top_n, int(rng.integers(0, combo_count))),)
            else:
                combos_iter = itertools.combinations(range(source_top_n), 4)
            for combo in combos_iter:
                combo_idx = np.asarray(combo, dtype=np.int64)
                node_x, edge_index, edge_attr = build_graph_inputs(
                    point_yx=point_yx[combo_idx],
                    point_mag=point_mag[combo_idx],
                    width=width,
                    height=height,
                    k_neighbors=graph_k_by_top_n[source_top_n],
                    graph_connectivity=graph_connectivity,
                    node_feature_mode=node_feature_mode,
                    edge_feature_mode=edge_feature_mode,
                )
                x_t = torch.from_numpy(node_x).to(device)
                ei_t = torch.from_numpy(edge_index).long().to(device)
                ea_t = torch.from_numpy(edge_attr).to(device)
                logits = model(x_t, ei_t, ea_t)
                quad_combination_count += 1
                for local_idx, global_idx in enumerate(combo):
                    if global_idx < brightest_k:
                        logits_sum[global_idx] += logits[local_idx]
                        logits_count[global_idx] += 1.0
    else:
        top_n_effective = tuple(n for n in effective_top_n_choices(len(point_yx), top_n_choices) if n >= brightest_k)
        if not top_n_effective:
            raise ValueError(
                "No effective top_n is large enough for the requested brightest-k. "
                f"top_n_choices={list(top_n_choices)}, brightest_k={brightest_k}. "
                "Use a smaller --brightest-k or a larger --top-n-choices value."
            )
        graph_k_by_top_n = {int(top_n): int(choose_graph_k_neighbors(int(top_n))) for top_n in top_n_effective}

        with torch.no_grad():
            for top_n in top_n_effective:
                graph_k = graph_k_by_top_n[int(top_n)]
                node_x, edge_index, edge_attr = build_graph_inputs(
                    point_yx=point_yx[:top_n],
                    point_mag=point_mag[:top_n],
                    width=width,
                    height=height,
                    k_neighbors=graph_k,
                    graph_connectivity=graph_connectivity,
                    node_feature_mode=node_feature_mode,
                    edge_feature_mode=edge_feature_mode,
                )
                x_t = torch.from_numpy(node_x).to(device)
                ei_t = torch.from_numpy(edge_index).long().to(device)
                ea_t = torch.from_numpy(edge_attr).to(device)
                logits = model(x_t, ei_t, ea_t)
                logits_sum += logits[:brightest_k]
                logits_count += 1.0

    logits_mean = logits_sum / logits_count.clamp_min(1.0).unsqueeze(1)
    probs = torch.softmax(logits_mean, dim=1)

    print(f"image={image_path}")
    print(f"centroid_count={int(point_yx.shape[0])}")
    print(f"brightest_k={brightest_k}")
    print(f"top_n_choices_used={list(top_n_effective)}")
    print(f"graph_connectivity={graph_connectivity}")
    print(f"model_backbone={model_backbone}")
    print(f"node_feature_mode={node_feature_mode}")
    print(f"edge_feature_mode={edge_feature_mode}")
    if quad_combinations_top_n > 0:
        print(f"quad_combinations_top_n={quad_combinations_top_n}")
        print(f"quad_combination_mode={quad_combination_mode}")
        print(f"quad_combination_count={quad_combination_count}")
    if graph_connectivity == "knn":
        print("k_neighbors_mode=dynamic[min(max(3, round(0.25 * n)), 8)]")
        print(f"k_neighbors_by_top_n={graph_k_by_top_n}")
    else:
        print("k_neighbors_mode=not_used_fully_connected")
    print(f"tetra3_match_count={len(matched_cat_ids)}")

    max_topk = min(topk, int(class_to_star_id.shape[0]))
    hit_count_any = 0
    hit_count_exact = 0
    for node_idx in range(brightest_k):
        top_idx = torch.topk(probs[node_idx], k=max_topk, dim=0).indices.tolist()
        top_catalog_ids = topk_catalog_predictions(top_idx, class_to_star_id, catalog_lookup)
        top_catalog_set = set(top_catalog_ids)
        hit_in_top10 = bool(top_catalog_set & matched_cat_id_set)
        if hit_in_top10:
            hit_count_any += 1
        centroid_y = float(point_yx[node_idx, 0])
        centroid_x = float(point_yx[node_idx, 1])
        flux = float(flux_sum[node_idx])
        assigned_match = find_nearest_match(point_yx[node_idx], matched_centroids, matched_cat_ids)
        assigned_match_rank = find_match_rank(assigned_match, top_catalog_ids)
        exact_hit = assigned_match_rank is not None
        if exact_hit:
            hit_count_exact += 1
        print(
            f"centroid_rank={node_idx + 1} y={centroid_y:.3f} x={centroid_x:.3f} "
            f"flux={flux:.3f}"
        )
        print(
            f"  top{max_topk}_hit_any_tetra3_match={'yes' if hit_in_top10 else 'no'}"
            f" matched_catID_nearest={format_catalog_id(assigned_match) if assigned_match is not None else 'none'}"
            f" top{max_topk}_hit_nearest_tetra_match={'yes' if exact_hit else 'no'}"
            f" nearest_match_rank={assigned_match_rank if assigned_match_rank is not None else 'none'}"
        )
        for rank, class_idx in enumerate(top_idx, start=1):
            star_id = int(class_to_star_id[int(class_idx)])
            catalog_id = top_catalog_ids[rank - 1]
            prob = float(probs[node_idx, int(class_idx)].item())
            is_match = catalog_id in matched_cat_id_set
            print(
                f"  {rank:02d}. star_id={star_id} catalog_id={format_catalog_id(catalog_id)} "
                f"prob={prob:.6f} tetra3_match={'yes' if is_match else 'no'}"
            )

    print(f"summary_top{max_topk}_hits_any_tetra3_match_among_brightest_{brightest_k}={hit_count_any}/{brightest_k}")
    print(f"summary_top{max_topk}_hits_nearest_tetra_match_among_brightest_{brightest_k}={hit_count_exact}/{brightest_k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
