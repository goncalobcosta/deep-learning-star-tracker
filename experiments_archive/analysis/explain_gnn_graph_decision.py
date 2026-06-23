from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tetra3
from GNN.GNN import (
    MODEL_BACKBONE_CHOICES,
    build_graph_inputs,
    choose_device,
    choose_graph_k_neighbors,
    make_star_model,
    magnitude_rank_feature,
    normalize_point_magnitude,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain one GNN graph decision on a real image, with node and edge inputs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--combo",
        default="1,2,3,4",
        help="1-based centroid ranks to place in the graph, e.g. 1,2,3,4 or 1,4,7,8.",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--database", type=Path, default=REPO_ROOT / "tetra3" / "data" / "default_database.npz")
    parser.add_argument(
        "--expected-catalog-ids",
        default="",
        help="Optional comma-separated expected catalogue IDs aligned with --combo.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: str):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def config_value(payload: dict, key: str, default):
    data_config = payload.get("data_config", {}) if isinstance(payload.get("data_config"), dict) else {}
    if key in data_config and data_config[key] is not None:
        return data_config[key]
    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    if key in train_args and train_args[key] is not None:
        return train_args[key]
    return default


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.replace(";", ",").split(",") if item.strip()]


def load_sorted_centroids(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size
        centroids, moments = tetra3.get_centroids_from_image(img, return_moments=True)
    point_yx = np.asarray(centroids, dtype=np.float32)
    flux = np.asarray(moments[0], dtype=np.float32)
    order = np.argsort(-flux, kind="stable")
    return point_yx[order], np.clip(flux[order], 1e-6, None), width, height


def main() -> int:
    args = parse_args()
    import torch

    device = choose_device(args.device)
    payload = load_checkpoint(args.checkpoint.expanduser().resolve(), device)
    class_to_star_id = np.asarray(payload["class_to_star_id"], dtype=np.int64)

    model_meta = payload.get("model", {}) if isinstance(payload, dict) else {}
    model_backbone = str(model_meta.get("name", model_meta.get("backbone", "gatv2")))
    if model_backbone not in MODEL_BACKBONE_CHOICES:
        model_backbone = "gatv2"
    model = make_star_model(
        model_backbone=model_backbone,
        in_dim=int(model_meta.get("in_dim", 2)),
        edge_dim=int(model_meta.get("edge_dim", 2)),
        hidden_dim=int(model_meta.get("hidden_dim", 128)),
        num_layers=int(model_meta.get("num_layers", 3)),
        heads=int(model_meta.get("heads", 4)),
        dropout=float(model_meta.get("dropout", 0.2)),
        num_id_classes=int(model_meta.get("num_id_classes", class_to_star_id.shape[0])),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    graph_connectivity = str(config_value(payload, "graph_connectivity", "fully"))
    node_feature_mode = str(config_value(payload, "node_feature_mode", "magnitude_subtracted_rank"))
    edge_feature_mode = str(config_value(payload, "edge_feature_mode", "distance_max"))

    point_yx, flux, width, height = load_sorted_centroids(args.image.expanduser().resolve())
    point_mag = (-2.5 * np.log10(flux)).astype(np.float32)
    combo_ranks = parse_int_list(args.combo)
    combo_idx = np.asarray([rank - 1 for rank in combo_ranks], dtype=np.int64)
    if np.any(combo_idx < 0) or np.any(combo_idx >= len(point_yx)):
        raise ValueError(f"Invalid combo {combo_ranks}; image has {len(point_yx)} centroids")

    db = np.load(args.database.expanduser().resolve(), allow_pickle=True)
    catalog_ids = np.asarray(db["star_catalog_IDs"])
    star_table = np.asarray(db["star_table"])
    catalog_mag = star_table[:, 5]
    expected = parse_int_list(args.expected_catalog_ids) if args.expected_catalog_ids.strip() else []

    graph_k = choose_graph_k_neighbors(len(combo_idx))
    node_x, edge_index, edge_attr = build_graph_inputs(
        point_yx=point_yx[combo_idx],
        point_mag=point_mag[combo_idx],
        width=width,
        height=height,
        k_neighbors=graph_k,
        graph_connectivity=graph_connectivity,
        node_feature_mode=node_feature_mode,
        edge_feature_mode=edge_feature_mode,
    )

    with torch.no_grad():
        logits = model(
            torch.from_numpy(node_x).to(device),
            torch.from_numpy(edge_index).long().to(device),
            torch.from_numpy(edge_attr).to(device),
        )
        probs = torch.softmax(logits, dim=1)
        top_prob, top_class = torch.topk(probs, k=min(int(args.topk), probs.shape[1]), dim=1)

    print(f"imagem={args.image}")
    print(f"checkpoint={args.checkpoint}")
    print(f"grafo_ranks_1_based={combo_ranks}")
    print(f"node_feature_mode={node_feature_mode}")
    print(f"edge_feature_mode={edge_feature_mode}")
    print(f"graph_connectivity={graph_connectivity}")
    print(f"model_backbone={model_backbone}")
    print()

    mag_sub = normalize_point_magnitude(point_mag[combo_idx])
    rank_feature = magnitude_rank_feature(point_mag[combo_idx])
    print("nos:")
    for local_i, global_i in enumerate(combo_idx):
        exp = expected[local_i] if local_i < len(expected) else None
        print(
            f"  no={local_i} centroid_rank={int(global_i)+1} "
            f"y={point_yx[global_i,0]:.3f} x={point_yx[global_i,1]:.3f} "
            f"flux={flux[global_i]:.3f} mag_inst={point_mag[global_i]:.3f} "
            f"mag_sub={mag_sub[local_i]:.3f} rank={rank_feature[local_i]:.3f} "
            f"node_x={node_x[local_i].tolist()} expected_catID={exp}"
        )

    print()
    print("edges:")
    for edge_i, (src, dst) in enumerate(edge_index.T):
        dyx = point_yx[combo_idx[int(dst)]] - point_yx[combo_idx[int(src)]]
        dist_px = float(np.linalg.norm(dyx))
        print(
            f"  {int(src)}->{int(dst)} dist_px={dist_px:.3f} "
            f"edge_attr={edge_attr[edge_i].tolist()}"
        )

    print()
    print("previsoes_por_no:")
    for local_i in range(len(combo_idx)):
        exp = expected[local_i] if local_i < len(expected) else None
        print(f"  no={local_i} centroid_rank={combo_ranks[local_i]} expected_catID={exp}")
        for rank, (cls, prob) in enumerate(zip(top_class[local_i].cpu().numpy(), top_prob[local_i].cpu().numpy()), start=1):
            star_id = int(class_to_star_id[int(cls)])
            cat_id = int(catalog_ids[star_id])
            hit = " <-- esperado" if exp is not None and cat_id == int(exp) else ""
            print(
                f"    {rank:02d}. star_id={star_id} catalog_id={cat_id} "
                f"cat_mag={float(catalog_mag[star_id]):.3f} prob={float(prob):.6f}{hit}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
