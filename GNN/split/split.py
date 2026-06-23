#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _resolve_dataset_dir(dataset_dir: str | None, dataset_run: str | None) -> Path:
    if dataset_dir:
        path = Path(dataset_dir).expanduser().resolve()
    elif dataset_run:
        path = (Path("synth_dataset") / "runs" / dataset_run).resolve()
    else:
        runs_root = (Path("synth_dataset") / "runs").resolve()
        runs = sorted(
            [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run")],
            key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else -1,
        )
        if not runs:
            raise FileNotFoundError("No dataset runs found under synth_dataset/runs")
        path = runs[-1]

    manifest = path / "dataset_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing dataset_manifest.json at {manifest}")
    return path


def _split_positions(
    n_items: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = float(train_frac + val_frac + test_frac)
    if total <= 0:
        raise ValueError("train/val/test fractions must be positive")
    train_r = train_frac / total
    val_r = val_frac / total

    n = int(n_items)
    if n <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    n_train = int(round(train_r * n))
    n_val = int(round(val_r * n))
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, min(n_train, n))
    n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train)
    n_test = n - n_train - n_val
    if n_test <= 0 and n > 1:
        if n_val > 0:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        n_test = n - n_train - n_val

    idx = np.arange(n, dtype=np.int64)
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def _chunk_paths(dataset_dir: Path, manifest: Dict[str, object]) -> List[Path]:
    paths: List[Path] = []
    for item in manifest.get("chunks", []):
        rel = Path(str(item["file"]).replace("\\", "/"))
        paths.append((dataset_dir / rel).resolve())
    return paths


def _candidate_star_ids_for_refs(
    chunk_paths: List[Path],
    refs: List[Tuple[int, int]],
    top_n: int,
) -> set[int]:
    by_chunk: Dict[int, List[int]] = {}
    for chunk_idx, scene_idx in refs:
        by_chunk.setdefault(int(chunk_idx), []).append(int(scene_idx))

    out: set[int] = set()
    for chunk_idx, scene_indices in by_chunk.items():
        with np.load(chunk_paths[int(chunk_idx)], allow_pickle=False) as data:
            scene_start = np.asarray(data["scene_point_start"], dtype=np.int64)
            scene_count = np.asarray(data["scene_point_count"], dtype=np.int64)
            point_star_id = np.asarray(data["point_star_id"], dtype=np.int64)

            for scene_idx in scene_indices:
                start = int(scene_start[int(scene_idx)])
                count = int(scene_count[int(scene_idx)])
                limit = min(count, int(top_n)) if int(top_n) > 0 else count
                if limit <= 0:
                    continue
                for star_id in point_star_id[start : start + limit].tolist():
                    star_id = int(star_id)
                    if star_id >= 0:
                        out.add(star_id)
    return out


def build_split_by_guide(
    dataset_dir: Path,
    output_npz: Path,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    max_train_groups_per_guide: int | None = None,
    max_val_groups_per_guide: int | None = None,
    max_test_groups_per_guide: int | None = None,
    ensure_val_test_candidates_in_train: bool = False,
    coverage_top_n: int = 8,
) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "dataset_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    guide_indices_manifest_raw = manifest.get("guide_star_indices", [])
    guide_indices_manifest = (
        np.asarray(guide_indices_manifest_raw, dtype=np.int64)
        if isinstance(guide_indices_manifest_raw, list)
        else np.empty((0,), dtype=np.int64)
    )

    # Closed-set split: keep all repeats of the same roll_degree together.
    refs_by_guide_roll: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
    candidate_ids_by_guide_roll: Dict[int, Dict[int, set[int]]] = {}
    chunks = manifest.get("chunks", [])
    for chunk_i, chunk in enumerate(chunks):
        rel = Path(str(chunk["file"]).replace("\\", "/"))
        npz_path = (dataset_dir / rel).resolve()
        with np.load(npz_path, allow_pickle=False) as data:
            scene_guides = np.asarray(data["guide_star_index"], dtype=np.int64)
            scene_rolls = np.asarray(data["roll_degree"], dtype=np.int64)
            if ensure_val_test_candidates_in_train:
                scene_start = np.asarray(data["scene_point_start"], dtype=np.int64)
                scene_count = np.asarray(data["scene_point_count"], dtype=np.int64)
                point_star_id = np.asarray(data["point_star_id"], dtype=np.int64)

        for scene_i, (guide_star, roll_degree) in enumerate(zip(scene_guides.tolist(), scene_rolls.tolist())):
            g = int(guide_star)
            r = int(roll_degree)
            refs_by_guide_roll.setdefault(g, {}).setdefault(r, []).append((int(chunk_i), int(scene_i)))
            if ensure_val_test_candidates_in_train:
                start = int(scene_start[int(scene_i)])
                count = int(scene_count[int(scene_i)])
                limit = min(count, int(coverage_top_n)) if int(coverage_top_n) > 0 else count
                if limit > 0:
                    candidates = candidate_ids_by_guide_roll.setdefault(g, {}).setdefault(r, set())
                    for star_id in point_star_id[start : start + limit].tolist():
                        star_id = int(star_id)
                        if star_id >= 0:
                            candidates.add(star_id)

    split_groups: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    split_guides: Dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}

    for guide_star in sorted(refs_by_guide_roll.keys()):
        refs_for_roll = refs_by_guide_roll[guide_star]
        roll_values = sorted(refs_for_roll.keys())
        if not roll_values:
            continue

        guide_rng = np.random.default_rng(int(seed) + int(guide_star) * 1000003)
        roll_order = guide_rng.permutation(len(roll_values))
        roll_values_perm = [roll_values[int(i)] for i in roll_order]

        train_pos, val_pos, test_pos = _split_positions(
            n_items=len(roll_values_perm),
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )

        if max_train_groups_per_guide is not None:
            train_pos = train_pos[: max(0, int(max_train_groups_per_guide))]
        if max_val_groups_per_guide is not None:
            val_pos = val_pos[: max(0, int(max_val_groups_per_guide))]
        if max_test_groups_per_guide is not None:
            test_pos = test_pos[: max(0, int(max_test_groups_per_guide))]

        def _expand_groups(positions: np.ndarray) -> List[Dict[str, object]]:
            out: List[Dict[str, object]] = []
            for pos in positions.tolist():
                roll = int(roll_values_perm[int(pos)])
                out.append(
                    {
                        "guide": int(guide_star),
                        "roll": int(roll),
                        "refs": list(refs_for_roll[roll]),
                    }
                )
            return out

        train_groups = _expand_groups(train_pos)
        val_groups = _expand_groups(val_pos)
        test_groups = _expand_groups(test_pos)

        split_groups["train"].extend(train_groups)
        split_groups["val"].extend(val_groups)
        split_groups["test"].extend(test_groups)

        train_refs = [ref for group in train_groups for ref in group["refs"]]  # type: ignore[index]
        val_refs = [ref for group in val_groups for ref in group["refs"]]  # type: ignore[index]
        test_refs = [ref for group in test_groups for ref in group["refs"]]  # type: ignore[index]

        if train_refs:
            split_guides["train"].add(int(guide_star))
        if val_refs:
            split_guides["val"].add(int(guide_star))
        if test_refs:
            split_guides["test"].add(int(guide_star))

    if not refs_by_guide_roll:
        raise RuntimeError("No guide_star_index groups found in dataset chunks")

    moved_groups_for_coverage: Dict[str, int] = {"val": 0, "test": 0}
    coverage_stats: Dict[str, object] | None = None
    if ensure_val_test_candidates_in_train:
        def _group_candidates(group: Dict[str, object]) -> set[int]:
            return candidate_ids_by_guide_roll.get(int(group["guide"]), {}).get(int(group["roll"]), set())

        train_candidates: set[int] = set()
        for group in split_groups["train"]:
            train_candidates.update(_group_candidates(group))

        changed = True
        while changed:
            changed = False
            for split_name in ("val", "test"):
                keep: List[Dict[str, object]] = []
                move: List[Dict[str, object]] = []
                for group in split_groups[split_name]:
                    unseen = _group_candidates(group) - train_candidates
                    if unseen:
                        move.append(group)
                    else:
                        keep.append(group)
                if move:
                    split_groups[split_name] = keep
                    split_groups["train"].extend(move)
                    moved_groups_for_coverage[split_name] += len(move)
                    for group in move:
                        train_candidates.update(_group_candidates(group))
                    changed = True

        val_candidates = set()
        for group in split_groups["val"]:
            val_candidates.update(_group_candidates(group))
        test_candidates = set()
        for group in split_groups["test"]:
            test_candidates.update(_group_candidates(group))
        coverage_stats = {
            "coverage_top_n": int(coverage_top_n),
            "moved_roll_groups_to_train": moved_groups_for_coverage,
            "train_candidate_stars": int(len(train_candidates)),
            "val_candidate_stars": int(len(val_candidates)),
            "test_candidate_stars": int(len(test_candidates)),
            "val_missing_from_train": int(len(val_candidates - train_candidates)),
            "test_missing_from_train": int(len(test_candidates - train_candidates)),
        }

    split_refs: Dict[str, List[Tuple[int, int]]] = {"train": [], "val": [], "test": []}
    split_roll_groups: Dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        split_refs[split_name] = [
            ref
            for group in split_groups[split_name]
            for ref in list(group["refs"])  # type: ignore[arg-type]
        ]
        split_roll_groups[split_name] = int(len(split_groups[split_name]))
        split_guides[split_name] = {int(group["guide"]) for group in split_groups[split_name]}

    for split_name in ("train", "val", "test"):
        split_refs[split_name].sort(key=lambda x: (x[0], x[1]))

    def _to_arrays(refs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if not refs:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
        chunk_idx = np.asarray([c for c, _ in refs], dtype=np.int32)
        scene_idx = np.asarray([s for _, s in refs], dtype=np.int32)
        return chunk_idx, scene_idx

    tr_chunk_idx, tr_scene_idx = _to_arrays(split_refs["train"])
    va_chunk_idx, va_scene_idx = _to_arrays(split_refs["val"])
    te_chunk_idx, te_scene_idx = _to_arrays(split_refs["test"])

    train_guides = np.asarray(sorted(split_guides["train"]), dtype=np.int32)
    val_guides = np.asarray(sorted(split_guides["val"]), dtype=np.int32)
    test_guides = np.asarray(sorted(split_guides["test"]), dtype=np.int32)

    out: Dict[str, np.ndarray] = {
        "train_chunk_idx": tr_chunk_idx,
        "train_scene_idx": tr_scene_idx,
        "val_chunk_idx": va_chunk_idx,
        "val_scene_idx": va_scene_idx,
        "test_chunk_idx": te_chunk_idx,
        "test_scene_idx": te_scene_idx,
        "train_guides": train_guides,
        "val_guides": val_guides,
        "test_guides": test_guides,
        "seed": np.asarray([seed], dtype=np.int64),
        "split_mode_code": np.asarray([1], dtype=np.int8),  # 1 = closed-set split within each guide.
    }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out)

    summary = {
        "dataset_dir": str(dataset_dir),
        "split_file": str(output_npz),
        "seed": int(seed),
        "split_mode": "roll_degree_group_within_guide_closed_set",
        "ensure_val_test_candidates_in_train": bool(ensure_val_test_candidates_in_train),
        "candidate_coverage": coverage_stats,
        "max_groups_per_guide": {
            "train": None if max_train_groups_per_guide is None else int(max_train_groups_per_guide),
            "val": None if max_val_groups_per_guide is None else int(max_val_groups_per_guide),
            "test": None if max_test_groups_per_guide is None else int(max_test_groups_per_guide),
        },
        "total_guides_in_manifest": int(guide_indices_manifest.shape[0]),
        "total_guides_in_dataset": int(len(refs_by_guide_roll)),
        "guide_counts": {
            "train": int(train_guides.shape[0]),
            "val": int(val_guides.shape[0]),
            "test": int(test_guides.shape[0]),
        },
        "roll_group_counts": {
            "train": int(split_roll_groups["train"]),
            "val": int(split_roll_groups["val"]),
            "test": int(split_roll_groups["test"]),
        },
        "scene_counts": {
            "train": int(tr_scene_idx.shape[0]),
            "val": int(va_scene_idx.shape[0]),
            "test": int(te_scene_idx.shape[0]),
        },
    }
    summary_path = output_npz.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _default_split_output(dataset_dir: Path, seed: int) -> Path:
    split_root = (Path("GNN") / "split" / "runs" / dataset_dir.name).resolve()
    split_root.mkdir(parents=True, exist_ok=True)
    return split_root / f"guide_split_seed{seed}.npz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create closed-set train/val/test split by roll_degree groups within each guide star."
    )
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--dataset-run", type=str, default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-train-groups-per-guide", type=int, default=None)
    parser.add_argument("--max-val-groups-per-guide", type=int, default=None)
    parser.add_argument("--max-test-groups-per-guide", type=int, default=None)
    parser.add_argument(
        "--ensure-val-test-candidates-in-train",
        action="store_true",
        help="Move roll groups from val/test into train until their top-N candidate stars are covered by train.",
    )
    parser.add_argument(
        "--coverage-top-n",
        type=int,
        default=8,
        help="Top-N brightest points per scene used by --ensure-val-test-candidates-in-train.",
    )
    args = parser.parse_args(argv)

    dataset_dir = _resolve_dataset_dir(args.dataset_dir, args.dataset_run)
    output_npz = Path(args.output).expanduser().resolve() if args.output else _default_split_output(dataset_dir, args.seed)
    summary = build_split_by_guide(
        dataset_dir=dataset_dir,
        output_npz=output_npz,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        max_train_groups_per_guide=args.max_train_groups_per_guide,
        max_val_groups_per_guide=args.max_val_groups_per_guide,
        max_test_groups_per_guide=args.max_test_groups_per_guide,
        ensure_val_test_candidates_in_train=args.ensure_val_test_candidates_in_train,
        coverage_top_n=args.coverage_top_n,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
