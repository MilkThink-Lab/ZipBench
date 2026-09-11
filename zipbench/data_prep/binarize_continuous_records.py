#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binarize continuous ZipBench records with per-task dynamic thresholds.

Input: ``--root/<model>/<dataset>.json`` records in the unified continuous
schema written by ``convert_continuous_native_records.py`` (``details[qid]``
carries a float ``is_correct`` in [0, 1], higher is better, and a ``task``
grouping key). All models under ``--root`` are the *real training records*;
thresholds are fitted on them only, once, before any synthesis.

Threshold rule (per task group; Y has shape (models, items))::

    cs = np.linspace(0.01, .99, 100)
    c = cs[np.argmin([np.mean(np.abs((Y[:, ind] > c).mean(axis=1)
                                     - Y[:, ind].mean(axis=1))) for c in cs])]
    Y_bin[:, ind] = (Y[:, ind] > c).astype(int)

i.e. the grid threshold whose binary per-model accuracy is, on average, closest
to the continuous per-model mean; ``argmin`` takes the first minimum; the
comparison is strict ``>``.

Output (``--output-dir/<model>/<dataset>.json``) keeps the old Zoom
``binarize_*_records.py`` field convention so the downstream voting /
aggregation scripts read it unchanged:

* per item: ``binary_is_correct`` (0/1), ``is_correct = any_correct =
  binary_is_correct`` (overwritten -- ``batch_vote_uniform_bins_results_json_multiqa``
  applies ``bool()`` to ``is_correct`` for several datasets, so a leftover
  float would be mis-read), ``continuous_is_correct`` (the original value),
  ``binarize_task``, ``binarize_threshold``;
* top level: ``is_correct`` / ``accuracy`` = binary mean,
  ``continuous_accuracy`` = continuous mean, ``binarization = {method,
  threshold_source, dataset, ...}``.

Also written: ``<output-dir>/<dataset>_binarization_thresholds.json`` (old
key layout, loadable through ``--apply-thresholds``) and
``<output-dir>/binarize_summary.jsonl`` (per-model accuracy before / after).

Every model must carry the same qid set (a missing item is an error, matching
the pipeline's ``missing_vs_union == 0`` check). Items flagged ``ignored``
(OCRBench_v2) in *any* model are dropped from every model (union) and listed
in the threshold file.

``--apply-thresholds`` binarizes with fixed thresholds from a threshold file
instead of fitting; it is an extension hook only -- the standard pipeline
does NOT binarize the test split (see docs/PIPELINE.md 1.4).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

GLOBAL_TASK = "all"


# ---------------------------------------------------------------------------
# threshold rule
# ---------------------------------------------------------------------------


def threshold_grid(start: float = 0.01, end: float = 0.99, num: int = 100) -> np.ndarray:
    return np.linspace(float(start), float(end), int(num))


def fit_threshold(scores: np.ndarray, cs: np.ndarray) -> Tuple[float, float, List[float]]:
    """``scores``: (models, items) of one task group. Returns (c, loss, losses)."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError(f"expected a non-empty (models, items) matrix, got shape {scores.shape}")
    cont_means = scores.mean(axis=1)
    losses = [float(np.mean(np.abs((scores > c).mean(axis=1) - cont_means))) for c in cs]
    best = int(np.argmin(losses))
    return float(cs[best]), losses[best], losses


def compute_thresholds(
    scores: np.ndarray,
    tasks: Sequence[str],
    cs: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Per-task thresholds in the old Zoom layout. ``scores``: (models, items)."""
    tasks = [str(t) for t in tasks]
    if scores.shape[1] != len(tasks):
        raise ValueError(f"scores has {scores.shape[1]} items but {len(tasks)} task labels")
    out: Dict[str, Dict[str, Any]] = {}
    for task in sorted(set(tasks)):
        ind = [i for i, t in enumerate(tasks) if t == task]
        sub = scores[:, ind]
        c, loss, _ = fit_threshold(sub, cs)
        out[task] = {
            "threshold": c,
            "loss": loss,
            "num_items": len(ind),
            "continuous_model_means": sub.mean(axis=1).tolist(),
            "binary_model_means": (sub > c).mean(axis=1).tolist(),
        }
    return out


def binarize_matrix(scores: np.ndarray, tasks: Sequence[str], thresholds: Dict[str, Dict[str, Any]]) -> np.ndarray:
    thr = np.array([float(thresholds[str(t)]["threshold"]) for t in tasks])
    return (np.asarray(scores, dtype=float) > thr[None, :]).astype(int)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def _resolve_dataset_json_path(model_dir: Path, dataset: str) -> Path:
    direct = model_dir / f"{dataset}.json"
    if direct.exists():
        return direct
    for path in sorted(model_dir.glob("*.json"), key=lambda p: p.name):
        if path.name.lower() == f"{dataset}.json".lower():
            return path
    return direct


def _split_csv(raw: str) -> List[str]:
    return [p.strip() for p in str(raw or "").split(",") if p.strip()]


def load_records(root: Path, dataset: str, include_models: Sequence[str] = ()) -> List[Tuple[str, Path, Dict[str, Any]]]:
    include = set(include_models)
    model_dirs = sorted((p for p in Path(root).iterdir() if p.is_dir()), key=lambda p: p.name)
    if include:
        model_dirs = [p for p in model_dirs if p.name in include]
    records: List[Tuple[str, Path, Dict[str, Any]]] = []
    for model_dir in model_dirs:
        path = _resolve_dataset_json_path(model_dir, dataset)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or not isinstance(payload.get("details"), dict):
            raise ValueError(f"{path}: expected a dict payload with a dict 'details'")
        records.append((model_dir.name, path, payload))
    if include:
        missing = sorted(include - {name for name, _, _ in records})
        if missing:
            raise ValueError(f"--include-models not found under {root}: {', '.join(missing)}")
    if not records:
        raise ValueError(f"no <model>/{dataset}.json records under {root}")
    return records


def _score(rec: Any, key: str, model: str, qid: str) -> float:
    if not isinstance(rec, dict) or key not in rec:
        raise ValueError(f"item {qid!r} of {model} has no {key!r} field")
    try:
        v = float(rec[key])
    except (TypeError, ValueError) as e:
        raise ValueError(f"item {qid!r} of {model}: non-numeric {key}={rec[key]!r}") from e
    if not np.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"item {qid!r} of {model}: {key}={v!r} outside [0, 1]")
    return v


def _is_ignored(rec: Any) -> bool:
    return isinstance(rec, dict) and bool(rec.get("ignored", False))


def align_records(
    records: List[Tuple[str, Path, Dict[str, Any]]],
    score_key: str,
    task_key: str,
) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """Check id sets, drop the ignored union, build (ids, tasks, scores, ignored)."""
    id_sets = [set(map(str, payload["details"].keys())) for _, _, payload in records]
    union = set.union(*id_sets)
    problems = []
    for (name, path, _), ids in zip(records, id_sets):
        missing = sorted(union - ids)
        if missing:
            problems.append(f"{name} ({path}) misses {len(missing)} item(s), e.g. {missing[:5]}")
    if problems:
        raise ValueError("qid sets differ across models (missing_vs_union must be 0):\n  " + "\n  ".join(problems))

    ignored = sorted(
        qid for qid in union
        if any(_is_ignored(payload["details"].get(qid)) for _, _, payload in records)
    )
    ignored_set = set(ignored)
    first = records[0][2]["details"]
    ordered = [str(q) for q in first.keys() if str(q) not in ignored_set]
    if not ordered:
        raise ValueError("no items left after dropping the ignored union")

    tasks: List[str] = []
    for qid in ordered:
        if not task_key:
            tasks.append(GLOBAL_TASK)
            continue
        task = None
        for _, _, payload in records:
            rec = payload["details"].get(qid)
            if isinstance(rec, dict) and rec.get(task_key) not in (None, ""):
                task = str(rec[task_key])
                break
        tasks.append(task if task is not None else "unknown")

    scores = np.zeros((len(records), len(ordered)), dtype=float)
    for m, (name, _, payload) in enumerate(records):
        for i, qid in enumerate(ordered):
            scores[m, i] = _score(payload["details"][qid], score_key, name, qid)
    return ordered, tasks, scores, ignored


def load_thresholds_file(path: Path) -> Dict[str, Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("thresholds") if isinstance(payload, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: not a thresholds file (missing 'thresholds')")
    out: Dict[str, Dict[str, Any]] = {}
    for task, info in raw.items():
        value = info.get("threshold") if isinstance(info, dict) else info
        try:
            c = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: bad threshold for task {task!r}: {value!r}") from e
        if not np.isfinite(c):
            raise ValueError(f"{path}: non-finite threshold for task {task!r}")
        out[str(task)] = {"threshold": c}
    return out


def build_binarized_payload(
    payload: Dict[str, Any],
    ordered_ids: Sequence[str],
    tasks: Sequence[str],
    scores_row: np.ndarray,
    binary_row: np.ndarray,
    thresholds: Dict[str, Dict[str, Any]],
    score_key: str,
    dataset: str,
    method: str,
    threshold_source: str,
    task_key: str,
    ignored: Sequence[str],
) -> Dict[str, Any]:
    out = copy.deepcopy(payload)
    details: Dict[str, Any] = {}
    for i, qid in enumerate(ordered_ids):
        rec = copy.deepcopy(payload["details"][qid])
        b = int(binary_row[i])
        rec["continuous_is_correct"] = float(scores_row[i])
        if score_key != "is_correct":
            rec[f"continuous_{score_key}"] = float(scores_row[i])
        rec["binarize_task"] = str(tasks[i])
        rec["binarize_threshold"] = float(thresholds[str(tasks[i])]["threshold"])
        rec["binary_is_correct"] = b
        rec["is_correct"] = b
        rec["any_correct"] = b
        details[qid] = rec
    binary_mean = float(binary_row.mean()) if len(binary_row) else 0.0
    out["details"] = details
    out["continuous_accuracy"] = float(scores_row.mean()) if len(scores_row) else 0.0
    out["is_correct"] = binary_mean
    out["accuracy"] = binary_mean
    out["binarization"] = {
        "method": method,
        "threshold_source": threshold_source,
        "dataset": dataset,
        "task_key": task_key or None,
        "score_key": score_key,
        "num_items": len(ordered_ids),
        "num_ignored_dropped": len(ignored),
    }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(
    root: Path,
    dataset: str,
    output_dir: Path,
    *,
    task_key: str = "task",
    score_key: str = "is_correct",
    apply_thresholds: Optional[Path] = None,
    require_count: Optional[int] = None,
    include_models: Sequence[str] = (),
    grid_start: float = 0.01,
    grid_end: float = 0.99,
    grid_num: int = 100,
    overwrite: bool = False,
) -> Dict[str, Any]:
    records = load_records(root, dataset, include_models)
    if require_count is not None and len(records) != require_count:
        raise ValueError(f"--require-count {require_count} but found {len(records)} record(s): "
                         f"{[n for n, _, _ in records]}")
    ordered_ids, tasks, scores, ignored = align_records(records, score_key, task_key)
    cs = threshold_grid(grid_start, grid_end, grid_num)

    if apply_thresholds is not None:
        thresholds = load_thresholds_file(apply_thresholds)
        missing = sorted(set(tasks) - set(thresholds))
        if missing:
            raise ValueError(f"{apply_thresholds} has no threshold for task(s): {missing}")
        method = "fixed_per_task_threshold"
        threshold_source = f"external_thresholds_json:{apply_thresholds}"
    else:
        thresholds = compute_thresholds(scores, tasks, cs)
        method = "per_task_threshold_grid" if task_key else "global_threshold_grid"
        threshold_source = "real_train_records"
    binary = binarize_matrix(scores, tasks, thresholds)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name / f"{dataset}.json" for name, _, _ in records]
    existing = [str(t) for t in targets if t.exists()]
    if existing and not overwrite:
        raise FileExistsError("output record(s) already exist; pass --overwrite to replace them: "
                              + ", ".join(existing[:5]))

    summary_rows = []
    for m, ((name, path, payload), target) in enumerate(zip(records, targets)):
        out = build_binarized_payload(
            payload, ordered_ids, tasks, scores[m], binary[m], thresholds, score_key,
            dataset, method, threshold_source, task_key, ignored,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.write("\n")
        summary_rows.append({
            "model": name,
            "input": str(path),
            "output": str(target),
            "num_items": len(ordered_ids),
            "continuous_accuracy": float(scores[m].mean()),
            "binary_accuracy": float(binary[m].mean()),
        })

    threshold_payload = {
        "dataset": dataset,
        "input_dir": str(root),
        "output_dir": str(output_dir),
        "models": [name for name, _, _ in records],
        "num_models": len(records),
        "num_common_items": len(ordered_ids),
        "task_key": task_key or None,
        "score_key": score_key,
        "method": method,
        "threshold_source": threshold_source,
        "grid": cs.tolist(),
        "thresholds": thresholds,
        "ignored_ids": ignored,
    }
    with (output_dir / f"{dataset}_binarization_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with (output_dir / "binarize_summary.jsonl").open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    threshold_payload["summary"] = summary_rows
    return threshold_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Record root with <model>/<dataset>.json continuous records.")
    ap.add_argument("--dataset", required=True, help="Dataset stem (<dataset>.json).")
    ap.add_argument("--output-dir", required=True, help="Where to write binarized <model>/<dataset>.json.")
    ap.add_argument("--task-key", default="task",
                    help="Per-item field that groups items for threshold fitting; '' = one threshold "
                         "for the whole dataset (use this for arenahard). Default: task.")
    ap.add_argument("--score-key", default="is_correct", help="Per-item continuous score field. Default: is_correct.")
    ap.add_argument("--apply-thresholds", default=None,
                    help="Thresholds JSON (this script's or an old Zoom binarize_*_records.py one): binarize with "
                         "these fixed per-task thresholds instead of fitting. Extension only; the standard "
                         "pipeline does not binarize the test split.")
    ap.add_argument("--require-count", type=int, default=None, help="Require exactly N real records.")
    ap.add_argument("--include-models", default="", help="Comma-separated model dir names to use.")
    ap.add_argument("--grid-start", type=float, default=0.01)
    ap.add_argument("--grid-end", type=float, default=0.99)
    ap.add_argument("--grid-num", type=int, default=100)
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing output records (only files this script writes are touched).")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run(
        Path(args.root), args.dataset, Path(args.output_dir),
        task_key=args.task_key, score_key=args.score_key,
        apply_thresholds=Path(args.apply_thresholds) if args.apply_thresholds else None,
        require_count=args.require_count, include_models=_split_csv(args.include_models),
        grid_start=args.grid_start, grid_end=args.grid_end, grid_num=args.grid_num,
        overwrite=args.overwrite,
    )
    print(f"models={result['num_models']} items={result['num_common_items']} "
          f"ignored_dropped={len(result['ignored_ids'])} method={result['method']}")
    for task, info in result["thresholds"].items():
        extra = f"\tloss={info['loss']:.6f}\titems={info['num_items']}" if "loss" in info else ""
        print(f"{task}\tthreshold={info['threshold']:.6f}{extra}")
    for row in result["summary"]:
        print(f"{row['model']}\tcontinuous={row['continuous_accuracy']:.6f}\tbinary={row['binary_accuracy']:.6f}")
    print(f"wrote={args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
