#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch voting driver that uniformly covers an accuracy range with bins (JSON results version, based on exhaustive combinations).

Core mechanism:
1) Split [acc_min, acc_max] into m equal-width bins;
2) The candidate pool comes from the filtered output of vote_offline_results_json.py (include/exclude/min-acc/max-acc, etc.),
   obtained as a list of model_name via --print-candidates;
3) Enumerate all combinations for each k (order controlled by --k-order, default 2..N, i.e. start from the smallest ensembles), calling vote_offline_results_json.py once per combination to produce an ensemble JSON;
   k=1 would merely copy a real model record and is not allowed as a synthetic record;
4) The top-level evaluation metric of the ensemble JSON decides the bin:
   - simpleqa / humaneval_plus: is_correct(0-1)
   - commonsense_qa: accuracy(0-100)
   - lands in an unfilled bin => accept and keep the JSON
   - otherwise delete the JSON
5) After all k are traversed, only the accepted records are emitted, whether or not every bin is filled.

Notes:
- Bins only hold newly generated ensemble JSONs; existing model JSONs are used solely to infer acc_min/acc_max (when not given by the user).
- This script takes over the vote script's --root/--dataset/--k/--out/--seed/--print-candidates/--selected-identifiers.
- Supports humaneval_plus results JSON (details is a list; each record has is_correct).
- Supports commonsense_qa results JSON (details is a dict; records carry predictions/references).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Constants & Configuration ─────────────────────────────────────────

DEFAULT_VOTE_SCRIPT = Path(__file__).with_name("vote_offline_results_json_correctness.py")

META_DETAIL_KEYS = frozenset({"type", "meta", "metadata"})
PER_INSTANCE_SUFFIX = ".per_instance.json"

CHOICE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "simpleqa":        re.compile(r"\b([ABC])\b", re.IGNORECASE),
    "c3":              re.compile(r"\b([ABCD])\b", re.IGNORECASE),
    "commonsense_qa":  re.compile(r"\b([ABCDE])\b", re.IGNORECASE),
    "openbookqa":      re.compile(r"\b([ABCD])\b", re.IGNORECASE),
    "openbookqa_fact": re.compile(r"\b([ABCD])\b", re.IGNORECASE),
    "longbenchv2_0shot": re.compile(r"\b([ABCD])\b", re.IGNORECASE),
}

FORBIDDEN_FORWARD_ARGS = frozenset({
    "--out", "--seed", "--root", "--dataset",
    "--k", "--print-candidates", "--selected-identifiers",
})


@dataclass(frozen=True)
class MetricConfig:
    """Per-dataset metric configuration."""
    kind: str   # "accuracy" (0-100) or "is_correct" (0-1)
    fmt: str    # format spec for output filenames


_ACC_CFG = MetricConfig("accuracy", ".2f")
_IC_CFG = MetricConfig("is_correct", ".6f")

# Datasets whose primary metric is accuracy (0-100); all others use is_correct (0-1).
METRIC_BY_DATASET: Dict[str, MetricConfig] = {
    "c3":                _ACC_CFG,
    "commonsense_qa":    _ACC_CFG,
    "openbookqa":        _ACC_CFG,
    "openbookqa_fact":   _ACC_CFG,
    "mbpp_plus":         _ACC_CFG,
    "scibench":          _ACC_CFG,
    "scicode":           _IC_CFG,
    "scicode_with_background": _IC_CFG,
    "livecodebench":     _IC_CFG,
    "longbenchv2_0shot": _ACC_CFG,
    "swe_bench_verified": _IC_CFG,
    "swe_bench_multilingual": _IC_CFG,
    "swe_bench_pro": _IC_CFG,
    "osworld_verified":  _IC_CFG,
    "toolathlon":        _IC_CFG,
    "terminal_bench":    _IC_CFG,
    "ocrbench_v2":       _IC_CFG,
    "omnidocbench_text_edit": _IC_CFG,
    "omnidocbench_tab_edit": _IC_CFG,
    "omnidocbench_tab_teds": _IC_CFG,
    "omnidocbench_formula_edit": _IC_CFG,
    "omnidocbench_formula_cdm": _IC_CFG,
    "omnidocbench_read_order_edit": _IC_CFG,
    "arenahard":         _IC_CFG,
    "tau2_bench":        _IC_CFG,
    # TheoremQA: real records have top-level `score` (0-100) and list-form
    # `details` with per-record `is_correct`. Synthetic JSONs from the vote
    # script use top-level `is_correct` (0-1). We unify on the 0-1 scale.
    "theoremqa":         _IC_CFG,
}


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class CandidateRecord:
    json_path: Path
    accuracy: float   # metric value (dataset-specific)
    source: str       # e.g. generated_k3


# ── Low-level Helpers ─────────────────────────────────────────────────

def _normalize_dataset(dataset: str) -> str:
    return dataset.strip().lower()


def _get_metric_config(dataset_key: str) -> MetricConfig:
    if dataset_key == "tau2_bench" or dataset_key.startswith("tau2_bench_"):
        return _IC_CFG
    return METRIC_BY_DATASET.get(dataset_key, _IC_CFG)


def _extract_choice_letter(
    prediction: Any, dataset_key: str, default: Optional[str] = None,
) -> Optional[str]:
    if prediction is None:
        return default
    pattern = CHOICE_PATTERNS.get(dataset_key, CHOICE_PATTERNS["simpleqa"])
    m = pattern.search(str(prediction))
    return m.group(1).upper() if m else default


def _normalize_accuracy_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        acc = float(value)
    except (TypeError, ValueError):
        return None
    return acc * 100.0 if acc <= 1.0 else acc


def _normalize_is_correct(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _normalize_is_correct(value[0]) if value else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "success", "correct"}:
            return True
        if s in {"false", "0", "no", "n", "fail", "failed", "incorrect"}:
            return False
    return None


# ── Metric Computation from Details ───────────────────────────────────

def _iter_detail_records(details: Dict[str, Any]):
    """Yield (key, record) pairs, filtering out meta keys."""
    for key, rec in details.items():
        if str(key).lower() not in META_DETAIL_KEYS:
            yield key, rec


def _compute_is_correct_from_dict(details: Dict[str, Any], dataset_key: str) -> float:
    """simpleqa-like: fraction of predictions matching 'A'."""
    if not details:
        return 0.0
    total = correct = 0
    for _, rec in _iter_detail_records(details):
        total += 1
        val = rec.get("prediction") if isinstance(rec, dict) else rec
        if _extract_choice_letter(val, dataset_key, default="C") == "A":
            correct += 1
    return correct / total if total > 0 else 0.0


def _compute_accuracy_from_dict(details: Dict[str, Any], dataset_key: str) -> float:
    """commonsense_qa-like: % of predictions matching references."""
    if not details:
        return 0.0
    total = correct = 0
    for _, rec in _iter_detail_records(details):
        total += 1
        if not isinstance(rec, dict):
            continue
        pred = _extract_choice_letter(
            rec.get("predictions", rec.get("prediction")), dataset_key,
        )
        ref = _extract_choice_letter(
            rec.get("references", rec.get("reference")), dataset_key,
        )
        if pred is None or ref is None:
            continue
        if pred == ref:
            correct += 1
    return (correct / total * 100.0) if total > 0 else 0.0


def _compute_accuracy_mbpp_plus(details: Dict[str, Any]) -> float:
    """mbpp_plus: correctness from any_correct / is_correct booleans."""
    if not details:
        return 0.0
    total = correct = 0
    for _, rec in _iter_detail_records(details):
        if not isinstance(rec, dict):
            continue
        flag = _normalize_is_correct(rec.get("any_correct"))
        if flag is None:
            flag = _normalize_is_correct(rec.get("is_correct"))
        if flag is None:
            continue
        total += 1
        if flag:
            correct += 1
    return (correct / total * 100.0) if total > 0 else 0.0


def _compute_is_correct_from_bool_dict(details: Dict[str, Any]) -> float:
    """swe_bench_verified-like: fraction of records with is_correct/any_correct true."""
    if not details:
        return 0.0
    total = correct = 0
    for _, rec in _iter_detail_records(details):
        if not isinstance(rec, dict):
            continue
        flag = _normalize_is_correct(rec.get("is_correct"))
        if flag is None:
            flag = _normalize_is_correct(rec.get("any_correct"))
        if flag is None:
            continue
        total += 1
        if flag:
            correct += 1
    return (correct / total) if total > 0 else 0.0


def _compute_is_correct_from_scicode(details: Dict[str, Any]) -> float:
    """SciCode: raw id -> 0/1 map, or generated records with voted_correctness."""
    if not details:
        return 0.0
    total = correct = 0
    for _, rec in _iter_detail_records(details):
        total += 1
        if isinstance(rec, dict):
            flag = _normalize_is_correct(rec.get("voted_correctness"))
            if flag is None:
                flag = _normalize_is_correct(rec.get("is_correct"))
            if flag is None:
                flag = _normalize_is_correct(rec.get("any_correct"))
        else:
            flag = _normalize_is_correct(rec)
        if flag:
            correct += 1
    return (correct / total) if total > 0 else 0.0


def _compute_is_correct_from_list(details: List[Any]) -> float:
    """List-style details with explicit correctness fields."""
    if not details:
        return 0.0
    total = correct = 0
    for rec in details:
        if not isinstance(rec, dict):
            continue
        flag = None
        for key in ("voted_correctness", "correct", "is_correct", "any_correct"):
            if key in rec:
                flag = _normalize_is_correct(rec.get(key))
                if flag is not None:
                    break
        if flag is None:
            continue
        total += 1
        if flag:
            correct += 1
    return correct / total if total > 0 else 0.0


def _compute_is_correct_from_terminal_bench(details: Any) -> float:
    """Terminal-Bench after binarization: only trust binary/voted fields."""
    if not details:
        return 0.0
    if isinstance(details, list):
        items = enumerate(details)
    elif isinstance(details, dict):
        items = details.items()
    else:
        return 0.0

    total = correct = 0
    for _, rec in items:
        if not isinstance(rec, dict):
            continue
        flag = None
        for key in ("voted_correctness", "binary_is_correct"):
            if key in rec:
                flag = _normalize_is_correct(rec.get(key))
                if flag is not None:
                    break
        if flag is None:
            continue
        total += 1
        if flag:
            correct += 1
    return correct / total if total > 0 else 0.0


def _compute_metric_from_details(
    details: Any, dataset_key: str, metric_kind: str,
) -> Optional[float]:
    """Unified dispatcher: compute metric from details (list or dict)."""
    if dataset_key in {"terminal_bench", "omnidocbench_text_edit", "omnidocbench_tab_edit", "omnidocbench_tab_teds", "omnidocbench_formula_edit", "omnidocbench_formula_cdm", "omnidocbench_read_order_edit", "arenahard"} and metric_kind == "is_correct":
        return _compute_is_correct_from_terminal_bench(details)
    if isinstance(details, list):
        return _compute_is_correct_from_list(details) if metric_kind == "is_correct" else None
    if not isinstance(details, dict) or not details:
        return None
    if metric_kind == "accuracy":
        if dataset_key == "mbpp_plus":
            return _compute_accuracy_mbpp_plus(details)
        return _compute_accuracy_from_dict(details, dataset_key)
    # metric_kind == "is_correct"
    if dataset_key in {"scicode", "scicode_with_background"}:
        return _compute_is_correct_from_scicode(details)
    if dataset_key in {"swe_bench_verified", "swe_bench_multilingual", "swe_bench_pro", "osworld_verified", "toolathlon", "ocrbench_v2", "livecodebench", "arc_challenge", "hellaswag"}:
        return _compute_is_correct_from_bool_dict(details)
    return _compute_is_correct_from_dict(details, dataset_key)


# ── JSON Reading ──────────────────────────────────────────────────────

def read_metric_from_results_json(path: Path, dataset_key: str) -> Optional[float]:
    """Read the primary metric from a results JSON file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    cfg = _get_metric_config(dataset_key)

    # ── Try top-level keys first ──
    if cfg.kind == "accuracy":
        # mbpp_plus has a special top-level key
        if dataset_key == "mbpp_plus":
            v = _normalize_accuracy_value(data.get("mbpp_plus_pass@1"))
            if v is not None:
                return v
        v = _normalize_accuracy_value(data.get("accuracy"))
        if v is not None:
            return v
    else:
        if "is_correct" in data:
            try:
                return float(data["is_correct"])
            except (TypeError, ValueError):
                pass
        if (dataset_key == "tau2_bench" or dataset_key.startswith("tau2_bench_")) and "acc" in data:
            try:
                return float(data["acc"])
            except (TypeError, ValueError):
                pass
        # TheoremQA real records carry a top-level `score` in [0, 100].
        if dataset_key == "theoremqa" and "score" in data:
            try:
                s = float(data["score"])
                return s / 100.0 if s > 1.0 else s
            except (TypeError, ValueError):
                pass

    # ── Fallback: compute from details (or treat data itself as details) ──
    details = data.get("details", data.get("results", data))
    return _compute_metric_from_details(details, dataset_key, cfg.kind)


# ── Bin Management ────────────────────────────────────────────────────

def build_bins(m: int, acc_min: float, acc_max: float) -> List[Tuple[float, float]]:
    if m <= 0:
        raise ValueError("--m must be a positive integer")
    if acc_min == acc_max:
        return [(acc_min, acc_max)] * m
    delta = (acc_max - acc_min) / m
    return [
        (acc_min + i * delta, acc_max if i == m - 1 else acc_min + (i + 1) * delta)
        for i in range(m)
    ]


def find_bin_index(acc: float, bins: Sequence[Tuple[float, float]]) -> Optional[int]:
    for i, (low, high) in enumerate(bins):
        # Last bin is inclusive on both ends [low, high]; others are half-open [low, high)
        if (low <= acc < high) if i < len(bins) - 1 else (low <= acc <= high):
            return i
    return None


def select_best_from_bucket(
    bucket: Sequence[CandidateRecord], low: float, high: float,
) -> Optional[CandidateRecord]:
    """Select the record closest to the bin center."""
    if not bucket:
        return None
    center = 0.5 * (low + high)
    return min(bucket, key=lambda c: abs(c.accuracy - center))


# ── Directory & File Utilities ────────────────────────────────────────

def list_model_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda x: x.name,
    )


def _is_tau2_dataset(dataset_key: str) -> bool:
    return dataset_key == "tau2_bench" or dataset_key.startswith("tau2_bench_")


def _derive_per_instance_model_name(path: Path) -> str:
    name = path.name
    if name.endswith(PER_INSTANCE_SUFFIX):
        return name[: -len(PER_INSTANCE_SUFFIX)]
    return path.stem


def list_flat_per_instance_files(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob(f"*{PER_INSTANCE_SUFFIX}"), key=lambda x: x.name)


def _resolve_dataset_json_path(model_dir: Path, dataset: str, dataset_key: str) -> Path:
    """
    Resolve dataset JSON with tolerant matching.

    This supports case differences such as C3.json vs --dataset c3.
    """
    candidates: List[Path] = []
    for stem in (str(dataset).strip(), dataset_key):
        if stem:
            candidates.append(model_dir / f"{stem}.json")
    if dataset_key and any(ch.isdigit() for ch in dataset_key):
        candidates.append(model_dir / f"{dataset_key.upper()}.json")

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path

    target_lower = f"{dataset_key}.json"
    for path in sorted(model_dir.glob("*.json"), key=lambda p: p.name):
        if path.name.lower() == target_lower:
            return path

    return candidates[0] if candidates else model_dir / f"{dataset}.json"


def _safe_unlink(path: Path) -> None:
    """Remove a file if it exists (no error if missing)."""
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        # Python < 3.8 fallback
        if path.exists():
            path.unlink()


# ── Accuracy Range Inference ──────────────────────────────────────────

def _infer_per_model_metrics(
    root_dir: Path, dataset: str, dataset_key: str,
) -> List[Tuple[str, float]]:
    """Read per-model metric values from root_dir/*/<dataset>.json."""
    per_model: List[Tuple[str, float]] = []
    for mdir in list_model_dirs(root_dir):
        jpath = _resolve_dataset_json_path(mdir, dataset, dataset_key)
        if not jpath.exists():
            continue
        acc = read_metric_from_results_json(jpath, dataset_key)
        if acc is not None:
            per_model.append((mdir.name, float(acc)))
    if _is_tau2_dataset(dataset_key):
        for jpath in list_flat_per_instance_files(root_dir):
            acc = read_metric_from_results_json(jpath, dataset_key)
            if acc is not None:
                per_model.append((_derive_per_instance_model_name(jpath), float(acc)))
    return per_model


def _patch_scibench_metrics(
    per_model: List[Tuple[str, float]], root_dir: Path, dataset: str, dataset_key: str,
) -> List[Tuple[str, float]]:
    """For scibench, fall back to total_accuracy when accuracy reads as 0."""
    patched: List[Tuple[str, float]] = []
    for model_name, acc in per_model:
        if acc != 0.0:
            patched.append((model_name, acc))
            continue
        jpath = _resolve_dataset_json_path(root_dir / model_name, dataset, dataset_key)
        try:
            with jpath.open("r", encoding="utf-8") as f:
                v = _normalize_accuracy_value(json.load(f).get("total_accuracy"))
        except Exception:
            v = None
        patched.append((model_name, float(v) if v is not None else acc))
    return patched


def infer_acc_range(
    root_dir: Path, dataset: str, dataset_key: str,
) -> Tuple[List[Tuple[str, float]], float, float]:
    """Infer (per_model_metrics, acc_min, acc_max) from model directories."""
    per_model = _infer_per_model_metrics(root_dir, dataset, dataset_key)
    if not per_model:
        raise ValueError(
            "Cannot infer the accuracy range from root/dataset (no readable <dataset>.json). "
            "Please pass --min-final-acc/--max-final-acc explicitly."
        )
    if dataset_key == "scibench":
        per_model = _patch_scibench_metrics(per_model, root_dir, dataset, dataset_key)
    values = [a for _, a in per_model]
    return per_model, min(values), max(values)


# ── Subprocess Wrappers ───────────────────────────────────────────────

def _build_vote_cmd(
    vote_script: Path, root_dir: Path, dataset: str, extra_args: Sequence[str],
) -> List[str]:
    """Build base command for the vote script."""
    return [
        sys.executable, str(vote_script),
        "--root", str(root_dir),
        "--dataset", str(dataset),
        *extra_args,
    ]


def run_vote_once(
    vote_script: Path,
    root_dir: Path,
    dataset: str,
    vote_args: Sequence[str],
    out_path: Path,
    seed: int,
) -> subprocess.CompletedProcess:
    cmd = _build_vote_cmd(vote_script, root_dir, dataset, [
        "--out", str(out_path), "--seed", str(seed), *vote_args,
    ])
    return subprocess.run(cmd)


def run_vote_print_candidates(
    vote_script: Path,
    root_dir: Path,
    dataset: str,
    vote_args: Sequence[str],
) -> List[str]:
    cmd = _build_vote_cmd(vote_script, root_dir, dataset, [
        "--print-candidates", *vote_args,
    ])
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "vote_offline_results_json.py --print-candidates failed "
            f"(returncode={completed.returncode}). stderr:\n{completed.stderr}"
        )
    return [ln for ln in (completed.stdout or "").splitlines() if ln.strip()]


# ── Argument Parsing ──────────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="JSON results version: enumerate k-combinations in --k-order (default 2..N) to build ensembles, and uniformly cover the accuracy range with bins.",
        allow_abbrev=False,
    )
    parser.add_argument("--m", type=int, required=True, help="Number of bins (at most m records are selected in the end).")
    parser.add_argument("--root", required=True, help="Root directory containing one sub-folder per model.")
    parser.add_argument("--dataset", required=True, help="Dataset name (expects <dataset>.json under each model folder).")
    parser.add_argument("--output-dir", required=True, help="Output directory (generated/, selected_list, etc.).")
    parser.add_argument("--min-final-acc", type=float, default=None, help="Lower bound (inclusive) of the accuracy range; inferred from the candidate pool if omitted.")
    parser.add_argument("--max-final-acc", type=float, default=None, help="Upper bound (inclusive) of the accuracy range; inferred from the candidate pool if omitted.")
    parser.add_argument("--vote-script", default=str(DEFAULT_VOTE_SCRIPT), help="Path to vote_offline_results_json.py.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Global maximum number of attempts (0 = unlimited).")
    parser.add_argument("--seed-start", type=int, default=0, help="Initial seed.")
    parser.add_argument("--seed-step", type=int, default=1, help="Seed increment after each call.")
    parser.add_argument(
        "--k-order",
        choices=("asc", "desc"),
        default="asc",
        help="Traversal order of k: asc=2..N (default; start from small ensembles and grow k only when bins stay empty); desc=N..2 (prefer the majority consensus of large ensembles).",
    )
    parser.add_argument("--strict-bins", action="store_true", help="Strict mode: exit with non-zero status if any bin is left empty.")

    known_args, leftover = parser.parse_known_args(argv)
    return known_args, leftover


def ensure_vote_args(leftover: Sequence[str]) -> List[str]:
    """Validate and strip optional '--' separator from forwarded args."""
    forwarded = list(leftover)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    conflicts = FORBIDDEN_FORWARD_ARGS & set(forwarded)
    if conflicts:
        raise ValueError(
            f"Do not set {', '.join(sorted(conflicts))} in the pass-through arguments; this script manages them automatically."
        )
    return forwarded


# ── Combination Attempt ───────────────────────────────────────────────

def _try_combination(
    *,
    vote_script: Path,
    root_dir: Path,
    dataset: str,
    dataset_key: str,
    vote_forwarded: List[str],
    combo: Tuple[str, ...],
    k: int,
    seed: int,
    generated_root: Path,
    attempt_index: int,
    bins: List[Tuple[float, float]],
    filled: List[bool],
    acc_min: float,
    acc_max: float,
    metric_cfg: MetricConfig,
) -> Optional[Tuple[int, CandidateRecord]]:
    """
    Run one vote combination. Returns (bin_idx, record) if the result fills
    an empty bin, otherwise cleans up the temp file and returns None.
    """
    temp_out = generated_root / f"temp_{attempt_index:06d}_k_{k}_seed_{seed}.json"
    try:
        vote_args = vote_forwarded + [
            "--k", str(k),
            "--selected-identifiers", ",".join(combo),
        ]
        completed = run_vote_once(vote_script, root_dir, dataset, vote_args, temp_out, seed)
        if completed.returncode != 0:
            print(
                f"[WARN] Vote script call #{attempt_index} failed: "
                f"k={k}, seed={seed}, returncode={completed.returncode}",
                file=sys.stderr,
            )
            return None

        acc = read_metric_from_results_json(temp_out, dataset_key)
        if acc is None:
            print(
                f"[WARN] Could not read {metric_cfg.kind} from JSON generated at attempt #{attempt_index}: "
                f"k={k}, seed={seed}; discarding.",
                file=sys.stderr,
            )
            return None

        acc = float(acc)
        if not (acc_min <= acc <= acc_max):
            return None

        bin_idx = find_bin_index(acc, bins)
        if bin_idx is None or filled[bin_idx]:
            return None

        # ── Accept: rename to final path ──
        final_path = generated_root / f"acc_{acc:{metric_cfg.fmt}}_k_{k}_seed_{seed}.json"
        temp_out.rename(final_path)
        return bin_idx, CandidateRecord(
            json_path=final_path, accuracy=acc, source=f"generated_k{k}",
        )
    finally:
        # Clean up temp file if it still exists (i.e. not renamed on success)
        _safe_unlink(temp_out)


# ── Output ────────────────────────────────────────────────────────────

def _write_results(
    output_dir: Path,
    bins: List[Tuple[float, float]],
    buckets: List[List[CandidateRecord]],
    dataset: str,
    metric_name: str,
    strict_bins: bool,
) -> int:
    """Select representatives, write output files, and return exit code."""
    selected: List[Tuple[int, float, float, CandidateRecord]] = []
    missing: List[int] = []

    for i, (low, high) in enumerate(bins):
        c = select_best_from_bucket(buckets[i], low, high)
        if c is None:
            missing.append(i)
        else:
            selected.append((i, low, high, c))

    if missing:
        print(
            "[WARN] The following bins are still empty after the traversal: "
            + ", ".join(f"#{i}" for i in missing),
            file=sys.stderr,
        )
        if strict_bins:
            print("[ERROR] --strict-bins is enabled; exiting because some bins are empty.", file=sys.stderr)
            return 1

    if not selected:
        print("[ERROR] Could not select a record for any bin.", file=sys.stderr)
        return 1

    selected.sort(key=lambda x: x[0])

    list_path = output_dir / "selected_list.txt"
    summary_path = output_dir / "selected_summary.jsonl"

    with list_path.open("w", encoding="utf-8") as fh_txt, \
         summary_path.open("w", encoding="utf-8") as fh_json:
        for i, low, high, rec in selected:
            rel_path = os.path.relpath(rec.json_path, output_dir)
            fh_txt.write(f"{i}\t{low:.6f}\t{high:.6f}\t{rec.accuracy:.6f}\t{rel_path}\n")
            fh_json.write(json.dumps({
                "bin_index": i,
                "bin_low": low,
                "bin_high": high,
                "accuracy": rec.accuracy,
                "json_path": str(rec.json_path),
                "source": rec.source,
                "metric": metric_name,
                "dataset": dataset,
            }, ensure_ascii=False) + "\n")

    for _, _, _, rec in selected:
        print(str(rec.json_path))
    return 0


# ── Main ──────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    args, leftover = parse_args(argv)
    vote_forwarded = ensure_vote_args(leftover)

    output_dir = Path(args.output_dir).resolve()
    generated_root = output_dir / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    vote_script = Path(args.vote_script).resolve()
    if not vote_script.exists():
        print(f"[ERROR] vote script not found: {vote_script}", file=sys.stderr)
        return 2

    root_dir = Path(args.root).resolve()
    if not root_dir.is_dir():
        print(f"[ERROR] --root not found or not a directory: {root_dir}", file=sys.stderr)
        return 2

    dataset = str(args.dataset)
    dataset_key = _normalize_dataset(dataset)
    metric_cfg = _get_metric_config(dataset_key)

    # ── Infer accuracy range ──
    per_model_acc: List[Tuple[str, float]] = []
    if args.min_final_acc is None or args.max_final_acc is None:
        try:
            per_model_acc, acc_min_auto, acc_max_auto = infer_acc_range(
                root_dir, dataset, dataset_key,
            )
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 2
        acc_min = acc_min_auto if args.min_final_acc is None else float(args.min_final_acc)
        acc_max = acc_max_auto if args.max_final_acc is None else float(args.max_final_acc)
    else:
        acc_min = float(args.min_final_acc)
        acc_max = float(args.max_final_acc)

    if acc_min > acc_max:
        print("[ERROR] --min-final-acc must not be greater than --max-final-acc", file=sys.stderr)
        return 2

    print(
        f"[INFO] Global accuracy range ({metric_cfg.kind}): [{acc_min:.6f}, {acc_max:.6f}]",
        file=sys.stderr,
    )

    bins = build_bins(args.m, acc_min, acc_max)

    # ── Print per-model bin statistics (informational only, no bin reservation) ──
    if per_model_acc:
        counts = [0] * len(bins)
        for _, acc in per_model_acc:
            idx = find_bin_index(acc, bins)
            if idx is not None:
                counts[idx] += 1
        for i, (low, high) in enumerate(bins):
            print(
                f"[INFO] Bin #{i:03d} [{low:.6f}, {high:.6f}] "
                f"already has {counts[i]} candidate(s) (used for voting only; does not occupy the bin).",
                file=sys.stderr,
            )

    # ── Enumerate combinations and fill bins ──
    buckets: List[List[CandidateRecord]] = [[] for _ in bins]
    filled = [False] * len(bins)
    filled_count = 0
    total_bins = len(bins)
    attempt_index = 0
    seed = args.seed_start
    seed_step = args.seed_step
    max_attempts = args.max_attempts

    try:
        vote_candidates = run_vote_print_candidates(
            vote_script, root_dir, dataset, vote_forwarded,
        )
    except Exception as e:
        print(f"[ERROR] Failed to obtain the voting candidate list: {e}", file=sys.stderr)
        return 2

    if not vote_candidates:
        print("[ERROR] The voting candidate list is empty; cannot enumerate combinations.", file=sys.stderr)
        return 2

    n_candidates = len(vote_candidates)
    if args.k_order == "desc":
        k_values = range(n_candidates, 1, -1)
    else:
        k_values = range(2, n_candidates + 1)
    print(
        f"[INFO] Candidate models available for voting: N={n_candidates}; enumerating combinations with k={'N..2' if args.k_order == 'desc' else '2..N'}"
        f" ({args.k_order}).",
        file=sys.stderr,
    )

    for k in k_values:
        if filled_count >= total_bins:
            break
        for combo in itertools.combinations(vote_candidates, k):
            if filled_count >= total_bins:
                break
            if 0 < max_attempts <= attempt_index:
                break

            attempt_index += 1
            result = _try_combination(
                vote_script=vote_script,
                root_dir=root_dir,
                dataset=dataset,
                dataset_key=dataset_key,
                vote_forwarded=vote_forwarded,
                combo=combo,
                k=k,
                seed=seed,
                generated_root=generated_root,
                attempt_index=attempt_index,
                bins=bins,
                filled=filled,
                acc_min=acc_min,
                acc_max=acc_max,
                metric_cfg=metric_cfg,
            )
            seed += seed_step

            if result is not None:
                bin_idx, record = result
                buckets[bin_idx].append(record)
                filled[bin_idx] = True
                filled_count += 1
                print(
                    f"[INFO] Accepted sample fills bin #{bin_idx:03d} "
                    f"[{bins[bin_idx][0]:.6f}, {bins[bin_idx][1]:.6f}] "
                    f"{metric_cfg.kind}={record.accuracy:.6f} k={k}; "
                    f"filled {filled_count}/{total_bins} so far.",
                    file=sys.stderr,
                )
        else:
            # Inner loop completed normally — continue to next k
            continue
        # Inner loop was broken (bins full or max attempts) — break outer too
        break

    return _write_results(
        output_dir, bins, buckets, dataset, metric_cfg.kind, args.strict_bins,
    )


if __name__ == "__main__":
    sys.exit(main())
