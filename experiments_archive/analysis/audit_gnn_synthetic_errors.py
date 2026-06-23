#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditar erros da GNN em exemplos sintéticos.

O objetivo é produzir exemplos legíveis do tipo:
  - que grafo entrou no modelo;
  - que nó foi avaliado;
  - que estrela o modelo escolheu;
  - que estrela devia ter escolhido;
  - que magnitudes/distâncias normalizadas foram vistas pela GNN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GNN.GNN import (  # noqa: E402
    ChunkCache,
    MODEL_BACKBONE_CHOICES,
    build_graph_inputs,
    choose_balanced_quad_combo,
    choose_device,
    choose_graph_k_neighbors,
    chunk_paths,
    edge_attr_from_geometry,
    image_size_from_manifest,
    load_checkpoint_with_fallback,
    load_manifest,
    load_split_refs,
    make_star_model,
    magnitude_rank_feature,
    manifest_class_star_ids,
    normalize_point_magnitude,
    quad_combo_by_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explica erros da GNN em amostras sintéticas, ao nível do grafo/nó/features."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("GNN/error_audits"))
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=5000,
        help="Número máximo de cenas a varrer. Usa 0 para varrer o split todo.",
    )
    parser.add_argument(
        "--mode",
        choices=("errors", "all"),
        default="errors",
        help="Guardar só nós onde o ID correto falhou o top-k, ou todos os nós reais.",
    )
    parser.add_argument(
        "--continue-after-max-examples",
        action="store_true",
        help="Continua a varrer cenas para estatísticas globais mesmo depois de escrever max-examples.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-chunks", type=int, default=2)
    parser.add_argument(
        "--recompute-balanced-on-scan",
        action="store_true",
        help=(
            "Recalcula o balanced_sample só nas cenas varridas. É rápido, mas não reproduz "
            "exatamente o treino se --max-scenes cortar o split antes do fim."
        ),
    )
    return parser.parse_args()


def config_value(payload: dict[str, Any], key: str, default: Any) -> Any:
    data_config = payload.get("data_config", {}) if isinstance(payload.get("data_config"), dict) else {}
    if key in data_config and data_config[key] is not None:
        return data_config[key]
    train_args = payload.get("train_args", {}) if isinstance(payload.get("train_args"), dict) else {}
    if key in train_args and train_args[key] is not None:
        return train_args[key]
    return default


def load_model(payload: dict[str, Any], device: torch.device) -> torch.nn.Module:
    class_to_star_id = np.asarray(payload["class_to_star_id"], dtype=np.int64)
    model_meta = payload.get("model", {}) if isinstance(payload.get("model"), dict) else {}
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
    return model


def build_class_catalog(payload: dict[str, Any], database_path: Path) -> dict[str, np.ndarray]:
    class_to_star_id = np.asarray(payload["class_to_star_id"], dtype=np.int64)
    with np.load(database_path, allow_pickle=False) as data:
        star_table = np.asarray(data["star_table"], dtype=np.float32)
        catalog_ids = np.asarray(data["star_catalog_IDs"])
    vectors = star_table[class_to_star_id, 2:5].astype(np.float32, copy=False)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return {
        "class_to_star_id": class_to_star_id,
        "catalog_ids": catalog_ids,
        "catalog_mag": star_table[:, 5].astype(np.float32, copy=False),
        "class_vectors": vectors,
    }


def catalog_id_for_star(catalog_ids: np.ndarray, star_id: int) -> str:
    raw = np.asarray(catalog_ids[int(star_id)])
    if raw.ndim == 0:
        return str(int(raw.item()))
    return "(" + ",".join(str(int(v)) for v in raw.reshape(-1).tolist()) + ")"


def angular_sep_deg(class_vectors: np.ndarray, class_a: int, class_b: int) -> float:
    va = class_vectors[int(class_a)]
    vb = class_vectors[int(class_b)]
    cosine = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def format_float(value: float | np.floating[Any], digits: int = 6) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def resolve_database_path(dataset_dir: Path, manifest: dict[str, Any]) -> Path:
    run = manifest.get("run", {})
    params = run.get("parameters", {}) if isinstance(run, dict) else {}
    raw = params.get("database_path") or run.get("database_path") if isinstance(run, dict) else None
    if raw:
        candidate = Path(str(raw).replace("\\", "/"))
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            rel = (dataset_dir / candidate).resolve()
            if rel.exists():
                return rel
    return (REPO_ROOT / "tetra3" / "data" / "default_database.npz").resolve()


def get_scene(cache: ChunkCache, paths: list[Path], chunk_idx: int, scene_idx: int) -> dict[str, Any]:
    shard = cache.get(int(chunk_idx), paths[int(chunk_idx)])
    start = int(shard["scene_point_start"][int(scene_idx)])
    count = int(shard["scene_point_count"][int(scene_idx)])
    end = start + count
    return {
        "point_yx": shard["point_yx"][start:end].astype(np.float32, copy=False),
        "point_star_id": shard["point_star_id"][start:end].astype(np.int64, copy=False),
        "point_is_false_star": shard["point_is_false_star"][start:end].astype(bool, copy=False),
        "point_mag": shard["point_magnitude"][start:end].astype(np.float32, copy=False),
        "scene_seed": int(shard["scene_seed"][int(scene_idx)]),
        "roll_degree": float(shard["roll_degree"][int(scene_idx)]),
    }


def choose_combo_for_scene(
    *,
    point_star_id: np.ndarray,
    source_top_n: int,
    mode: str,
    scene_seed: int,
    seed: int,
    balanced_state: dict[str, Any],
) -> tuple[int, tuple[int, int, int, int]]:
    combo_count = math.comb(source_top_n, 4)
    if mode == "sample":
        rng = np.random.default_rng(int(scene_seed) + int(seed) * 1000003)
        combo_idx = int(rng.integers(0, combo_count))
        return combo_idx, quad_combo_by_index(source_top_n, combo_idx)
    if mode == "balanced_sample":
        eligible = balanced_state["eligible_star_ids"]
        candidate_counts = balanced_state["candidate_counts"]
        for star_id in point_star_id[:source_top_n].tolist():
            star_id = int(star_id)
            if star_id >= 0 and star_id in eligible:
                candidate_counts[star_id] = int(candidate_counts.get(star_id, 0) + 1)
        return choose_balanced_quad_combo(
            point_star_id=point_star_id,
            source_top_n=source_top_n,
            star_as_input_count=balanced_state["selected_counts"],
            rng=balanced_state["rng"],
            eligible_star_ids=balanced_state["eligible_star_ids"],
            star_as_input_candidate_count=balanced_state["candidate_counts"],
        )
    return 0, tuple(range(4))


def update_balanced_state_after_choice(
    *,
    point_star_id: np.ndarray,
    source_top_n: int,
    combo: tuple[int, int, int, int],
    balanced_state: dict[str, Any],
) -> None:
    eligible = balanced_state["eligible_star_ids"]
    for local_idx in combo:
        star_id = int(point_star_id[int(local_idx)])
        if star_id >= 0 and star_id in eligible:
            balanced_state["selected_counts"][star_id] = int(balanced_state["selected_counts"].get(star_id, 0) + 1)


def describe_node_features(node_feature_mode: str, point_mag: np.ndarray, node_x: np.ndarray) -> list[dict[str, str]]:
    mag_sub = normalize_point_magnitude(point_mag)
    rank_feature = magnitude_rank_feature(point_mag)
    rows: list[dict[str, str]] = []
    for i in range(point_mag.shape[0]):
        rows.append(
            {
                "mag_instrumental": format_float(point_mag[i], 6),
                "mag_subtraida": format_float(mag_sub[i], 6),
                "rank_magnitude": format_float(rank_feature[i], 6),
                "node_feature_mode": node_feature_mode,
                "node_x": json.dumps([float(v) for v in node_x[i].tolist()], ensure_ascii=False),
            }
        )
    return rows


def describe_edges(
    *,
    point_yx: np.ndarray,
    point_mag_for_edges: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    width: int,
    height: int,
    edge_feature_mode: str,
) -> list[dict[str, Any]]:
    src, dst = edge_index
    raw_dist = np.linalg.norm(point_yx[dst] - point_yx[src], axis=1).astype(np.float32)
    if edge_feature_mode.endswith("_dmag_node"):
        include_dmag = True
    elif edge_feature_mode.endswith("_dmag"):
        include_dmag = True
    else:
        include_dmag = False

    rows: list[dict[str, Any]] = []
    for edge_i, (s, d) in enumerate(edge_index.T.tolist()):
        item = {
            "src": int(s),
            "dst": int(d),
            "distancia_px": float(raw_dist[edge_i]),
            "edge_attr": [float(v) for v in edge_attr[edge_i].tolist()],
        }
        if include_dmag:
            item["delta_magnitude_dst_menos_src"] = float(point_mag_for_edges[int(d)] - point_mag_for_edges[int(s)])
        rows.append(item)
    return rows


def markdown_case(record: dict[str, Any]) -> str:
    top_rows = "\n".join(
        (
            f"| {p['rank']} | {p['star_id']} | {p['catalog_id']} | "
            f"{format_float(p['catalog_mag'], 3)} | {format_float(p['prob'], 6)} | "
            f"{format_float(p['sep_deg_to_true'], 3)} |"
        )
        for p in record["top_predictions"]
    )
    edge_rows = "\n".join(
        (
            f"| {e['src']}->{e['dst']} | {format_float(e['distancia_px'], 3)} | "
            f"`{json.dumps(e['edge_attr'], ensure_ascii=False)}` | "
            f"{format_float(e.get('delta_magnitude_dst_menos_src', float('nan')), 6)} |"
        )
        for e in record["edges_from_node"]
    )
    return f"""## Caso {record['case_index']:03d}: erro no nó local {record['node_local_index']}

Cena sintética: `chunk={record['chunk_idx']}`, `scene={record['scene_idx']}`, `seed={record['scene_seed']}`, `roll={format_float(record['roll_degree'], 3)}°`.

Grafo usado pela GNN: centróides de ranking {record['combo_centroid_ranks_1_based']} dentro do top{record['source_top_n']} da cena. As estrelas reais nesse grafo eram {record['combo_star_ids']}.

Nó analisado: local `{record['node_local_index']}`, centróide rank `{record['centroid_rank_1_based']}`, posição `(y={format_float(record['y'], 3)}, x={format_float(record['x'], 3)})`.

O correto era prever `star_id={record['true_star_id']}`, `catalog_id={record['true_catalog_id']}`, magnitude de catálogo `{format_float(record['true_catalog_mag'], 3)}`. O modelo colocou como top1 `star_id={record['pred_star_id']}`, `catalog_id={record['pred_catalog_id']}`, magnitude de catálogo `{format_float(record['pred_catalog_mag'], 3)}`. O ID correto ficou em `rank={record['true_rank_in_topk']}` dentro do top{record['topk']} (`none` significa que falhou o top{record['topk']}).

Features do nó que entraram no modelo:

| variável | valor |
| --- | ---: |
| magnitude instrumental | {record['mag_instrumental']} |
| magnitude subtraída à mais brilhante do grafo | {record['mag_subtraida']} |
| rank de magnitude no grafo | {record['rank_magnitude']} |
| node_feature_mode | `{record['node_feature_mode']}` |
| node_x final | `{record['node_x']}` |

Arestas que saem deste nó:

| aresta | distância px | edge_attr final | delta magnitude |
| --- | ---: | --- | ---: |
{edge_rows}

Top{record['topk']} previsto para este nó:

| rank | star_id | catalog_id | mag catálogo | prob | sep. angular ao correto |
| ---: | ---: | --- | ---: | ---: | ---: |
{top_rows}

Leitura técnica: este erro compara o padrão local visto pela GNN com `8818` classes possíveis. Se várias estrelas de catálogo tiverem magnitude e relações geométricas parecidas neste grafo, a loss por nó pode preferir uma classe globalmente plausível mas errada para esta zona do céu.
"""


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    payload = load_checkpoint_with_fallback(args.checkpoint, device)
    model = load_model(payload, device)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    manifest = load_manifest(dataset_dir)
    paths = chunk_paths(dataset_dir, manifest)
    width, height = image_size_from_manifest(manifest)
    database_path = resolve_database_path(dataset_dir, manifest)

    class_to_star_id = np.asarray(payload["class_to_star_id"], dtype=np.int64)
    star_id_to_class = {int(star_id): idx for idx, star_id in enumerate(class_to_star_id.tolist())}
    class_catalog = build_class_catalog(payload, database_path)
    catalog_ids = class_catalog["catalog_ids"]
    catalog_mag = class_catalog["catalog_mag"]
    class_vectors = class_catalog["class_vectors"]

    graph_connectivity = str(config_value(payload, "graph_connectivity", "fully"))
    node_feature_mode = str(config_value(payload, "node_feature_mode", "magnitude_norm_median"))
    edge_feature_mode = str(config_value(payload, "edge_feature_mode", "distance_max"))
    quad_combinations_top_n = int(config_value(payload, "quad_combinations_top_n", 8) or 0)
    quad_combination_mode = str(config_value(payload, "quad_combination_mode", "balanced_sample"))
    seed = int(config_value(payload, "seed", 12345))

    if quad_combinations_top_n < 4:
        raise ValueError("Este auditor assume treino por quads; quad_combinations_top_n tem de ser >= 4.")

    refs = load_split_refs(split_file, args.split)
    max_scenes = len(refs) if int(args.max_scenes) <= 0 else min(int(args.max_scenes), len(refs))
    refs_to_scan = refs[:max_scenes]

    explicit_class_star_ids = manifest_class_star_ids(manifest)
    eligible_star_ids = set(int(x) for x in (explicit_class_star_ids if explicit_class_star_ids is not None else class_to_star_id))
    balanced_state: dict[str, Any] = {
        "rng": np.random.default_rng(seed),
        "eligible_star_ids": eligible_star_ids,
        "selected_counts": {},
        "candidate_counts": {},
    }

    if quad_combination_mode == "balanced_sample" and not args.recompute_balanced_on_scan and max_scenes < len(refs):
        print(
            "Aviso: balanced_sample depende da ordem do split inteiro. "
            "Com --max-scenes menor que o split, usa --recompute-balanced-on-scan para assumir "
            "diagnóstico local, ou --max-scenes 0 para reproduzir o split completo.",
            file=sys.stderr,
        )

    cache = ChunkCache(max_items=args.cache_chunks)
    rows: list[dict[str, Any]] = []
    markdown_blocks: list[str] = []
    scanned_real_nodes = 0
    scanned_error_nodes = 0
    top1_hits = 0
    top5_hits = 0
    topk_hits = 0
    by_centroid_rank: dict[int, dict[str, int]] = {}

    for ref_index, (chunk_idx, scene_idx) in enumerate(refs_to_scan):
        scene = get_scene(cache, paths, chunk_idx, scene_idx)
        count = int(scene["point_yx"].shape[0])
        source_top_n = min(count, quad_combinations_top_n)
        if source_top_n < 4:
            continue

        combo_idx, combo = choose_combo_for_scene(
            point_star_id=scene["point_star_id"],
            source_top_n=source_top_n,
            mode=quad_combination_mode,
            scene_seed=scene["scene_seed"],
            seed=seed,
            balanced_state=balanced_state,
        )
        if quad_combination_mode == "balanced_sample":
            update_balanced_state_after_choice(
                point_star_id=scene["point_star_id"],
                source_top_n=source_top_n,
                combo=combo,
                balanced_state=balanced_state,
            )

        combo_arr = np.asarray(combo, dtype=np.int64)
        point_yx = scene["point_yx"][combo_arr]
        point_star_id = scene["point_star_id"][combo_arr]
        point_false = scene["point_is_false_star"][combo_arr]
        point_mag = scene["point_mag"][combo_arr]
        graph_k = choose_graph_k_neighbors(point_yx.shape[0])

        node_x, edge_index, edge_attr = build_graph_inputs(
            point_yx=point_yx,
            point_mag=point_mag,
            width=width,
            height=height,
            k_neighbors=graph_k,
            graph_connectivity=graph_connectivity,
            node_feature_mode=node_feature_mode,
            edge_feature_mode=edge_feature_mode,
        )
        node_feature_rows = describe_node_features(node_feature_mode, point_mag, node_x)
        point_mag_norm = normalize_point_magnitude(point_mag)
        point_mag_for_edges = node_x[:, 0] if edge_feature_mode.endswith("_dmag_node") else point_mag_norm
        edge_rows = describe_edges(
            point_yx=point_yx,
            point_mag_for_edges=point_mag_for_edges,
            edge_index=edge_index,
            edge_attr=edge_attr,
            width=width,
            height=height,
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

        for local_i, star_id in enumerate(point_star_id.tolist()):
            if bool(point_false[local_i]) or int(star_id) < 0 or int(star_id) not in star_id_to_class:
                continue
            scanned_real_nodes += 1
            true_class = int(star_id_to_class[int(star_id)])
            top_classes = [int(x) for x in top_class[local_i].detach().cpu().numpy().tolist()]
            top_probs = [float(x) for x in top_prob[local_i].detach().cpu().numpy().tolist()]
            true_rank = None
            for rank, cls in enumerate(top_classes, start=1):
                if int(cls) == true_class:
                    true_rank = rank
                    break
            is_error = true_rank is None
            centroid_rank = int(combo[local_i]) + 1
            bucket = by_centroid_rank.setdefault(centroid_rank, {"real": 0, "top1": 0, "top5": 0, "topk": 0})
            bucket["real"] += 1
            if true_rank == 1:
                top1_hits += 1
                bucket["top1"] += 1
            if true_rank is not None and true_rank <= 5:
                top5_hits += 1
                bucket["top5"] += 1
            if true_rank is not None:
                topk_hits += 1
                bucket["topk"] += 1
            if is_error:
                scanned_error_nodes += 1
            if args.mode == "errors" and not is_error:
                continue
            if len(rows) >= int(args.max_examples):
                if args.continue_after_max_examples:
                    continue
                break

            pred_class = top_classes[0]
            pred_star_id = int(class_to_star_id[pred_class])
            true_star_id = int(star_id)
            edges_from_node = [item for item in edge_rows if int(item["src"]) == int(local_i)]
            top_predictions = []
            for rank, (cls, prob) in enumerate(zip(top_classes, top_probs), start=1):
                pred_sid = int(class_to_star_id[int(cls)])
                top_predictions.append(
                    {
                        "rank": rank,
                        "class": int(cls),
                        "star_id": pred_sid,
                        "catalog_id": catalog_id_for_star(catalog_ids, pred_sid),
                        "catalog_mag": float(catalog_mag[pred_sid]),
                        "prob": float(prob),
                        "sep_deg_to_true": angular_sep_deg(class_vectors, true_class, int(cls)),
                    }
                )

            record: dict[str, Any] = {
                "case_index": len(rows) + 1,
                "ref_index": int(ref_index),
                "chunk_idx": int(chunk_idx),
                "scene_idx": int(scene_idx),
                "scene_seed": int(scene["scene_seed"]),
                "roll_degree": float(scene["roll_degree"]),
                "source_top_n": int(source_top_n),
                "combo_idx": int(combo_idx),
                "combo_centroid_ranks_1_based": [int(x) + 1 for x in combo],
                "combo_star_ids": [int(x) for x in point_star_id.tolist()],
                "node_local_index": int(local_i),
                "centroid_rank_1_based": centroid_rank,
                "y": float(point_yx[local_i, 0]),
                "x": float(point_yx[local_i, 1]),
                "true_star_id": true_star_id,
                "true_class": true_class,
                "true_catalog_id": catalog_id_for_star(catalog_ids, true_star_id),
                "true_catalog_mag": float(catalog_mag[true_star_id]),
                "pred_star_id": pred_star_id,
                "pred_class": pred_class,
                "pred_catalog_id": catalog_id_for_star(catalog_ids, pred_star_id),
                "pred_catalog_mag": float(catalog_mag[pred_star_id]),
                "pred_prob": float(top_probs[0]),
                "true_rank_in_topk": true_rank if true_rank is not None else "none",
                "topk": int(args.topk),
                "graph_connectivity": graph_connectivity,
                "node_feature_mode": node_feature_mode,
                "edge_feature_mode": edge_feature_mode,
                "edges_from_node": edges_from_node,
                "top_predictions": top_predictions,
                **node_feature_rows[local_i],
            }
            rows.append(record)
            markdown_blocks.append(markdown_case(record))
        if len(rows) >= int(args.max_examples) and not args.continue_after_max_examples:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.expanduser().resolve().parent.name
    csv_path = args.output_dir / f"{stem}_{args.split}_error_audit.csv"
    md_path = args.output_dir / f"{stem}_{args.split}_error_audit.md"

    fieldnames = [
        "case_index",
        "ref_index",
        "chunk_idx",
        "scene_idx",
        "scene_seed",
        "roll_degree",
        "source_top_n",
        "combo_idx",
        "combo_centroid_ranks_1_based",
        "combo_star_ids",
        "node_local_index",
        "centroid_rank_1_based",
        "y",
        "x",
        "true_star_id",
        "true_class",
        "true_catalog_id",
        "true_catalog_mag",
        "pred_star_id",
        "pred_class",
        "pred_catalog_id",
        "pred_catalog_mag",
        "pred_prob",
        "true_rank_in_topk",
        "topk",
        "mag_instrumental",
        "mag_subtraida",
        "rank_magnitude",
        "node_feature_mode",
        "node_x",
        "graph_connectivity",
        "edge_feature_mode",
        "edges_from_node",
        "top_predictions",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["combo_centroid_ranks_1_based"] = json.dumps(out["combo_centroid_ranks_1_based"], ensure_ascii=False)
            out["combo_star_ids"] = json.dumps(out["combo_star_ids"], ensure_ascii=False)
            out["edges_from_node"] = json.dumps(out["edges_from_node"], ensure_ascii=False)
            out["top_predictions"] = json.dumps(out["top_predictions"], ensure_ascii=False)
            writer.writerow({key: out.get(key, "") for key in fieldnames})

    top1_rate = top1_hits / max(scanned_real_nodes, 1)
    top5_rate = top5_hits / max(scanned_real_nodes, 1)
    topk_rate = topk_hits / max(scanned_real_nodes, 1)
    by_rank_lines = []
    for rank in sorted(by_centroid_rank):
        item = by_centroid_rank[rank]
        real = max(int(item["real"]), 1)
        by_rank_lines.append(
            f"| {rank} | {item['real']} | {item['top1'] / real:.6f} | "
            f"{item['top5'] / real:.6f} | {item['topk'] / real:.6f} |"
        )
    by_rank_table = "\n".join(by_rank_lines)

    header = f"""# Auditoria de erros GNN

Checkpoint: `{args.checkpoint}`

Dataset: `{dataset_dir}`

Split: `{args.split}`

Configuração: `graph_connectivity={graph_connectivity}`, `node_feature_mode={node_feature_mode}`, `edge_feature_mode={edge_feature_mode}`, `quad_combinations_top_n={quad_combinations_top_n}`, `quad_combination_mode={quad_combination_mode}`.

Cenas varridas: `{max_scenes}`. Nós reais avaliados: `{scanned_real_nodes}`. Nós com erro top{args.topk}: `{scanned_error_nodes}`. Casos escritos: `{len(rows)}`.

Métricas nesta auditoria: `top1={top1_rate:.6f}`, `top5={top5_rate:.6f}`, `top{args.topk}={topk_rate:.6f}`.

## Taxa por ranking de centróide

| centroid_rank | nós reais | top1 | top5 | top{args.topk} |
| ---: | ---: | ---: | ---: | ---: |
{by_rank_table}

"""
    md_path.write_text(header + "\n\n".join(markdown_blocks) + "\n", encoding="utf-8")

    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"cenas_varridas={max_scenes}")
    print(f"nos_reais_avaliados={scanned_real_nodes}")
    print(f"erros_top{args.topk}={scanned_error_nodes}")
    print(f"top1={top1_rate:.6f}")
    print(f"top5={top5_rate:.6f}")
    print(f"top{args.topk}={topk_rate:.6f}")
    print(f"casos_escritos={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
