#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline voting on *correctness* (True/False) over OpenCompass results JSON files.

Differences from `vote_offline_results_json_multiqa.py`:
- multiqa: votes over option letters (A/B/C/D/E)
- this script: first judges each model's prediction vs reference as True/False according to the dataset config,
          then runs majority voting over the True/False results.

Currently targets QA-style datasets:
- simpleqa: options A/B/C, A=correct, B/C=wrong; correctness follows from the prediction alone, no gold comparison needed
- c3: options A/B/C/D, compare predictions against references
- commonsense_qa: options A/B/C/D/E, compare predictions against references
- openbookqa / openbookqa_fact: options A/B/C/D
- scibench: numeric / free-text answer comparison
- SciCode: read the raw id -> 0/1 correctness map directly
- mbpp_plus: read the any_correct / is_correct fields directly

Expected file layout:
root_dir/
  model_A/
    commonsense_qa.json
  model_B/
    commonsense_qa.json
  ...

Output of this script:
{
  "is_correct": 0.755,     # fraction correct after voting (0-1)
  "accuracy": 75.5,        # accuracy after voting (percentage)
  "details": {
    "0": { ... "voted_correctness": true },
    ...
  }
}
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Constants ─────────────────────────────────────────────────────────

# SciBench: numeric answer comparison tolerances.
# Aligned with OpenCompass NumericAccEvaluator config in eval_scibench.py:
#   rtol=1e-2 (relative tolerance 1%), atol=1e-3 (absolute tolerance 0.001)
SCIBENCH_ATOL_DEFAULT = 1e-3
SCIBENCH_RTOL_DEFAULT = 1e-2

# Match signed floats / scientific notation; take the *last* match as answer.
SCIBENCH_NUMBER_RE = re.compile(
    r"[-+]?((?:\d+\.?\d*)|(?:\.\d+))(?:[eE][-+]?\d+)?"
)

# Match LaTeX scientific notation: e.g. "4.737 \times 10^{14}" -> keep only
# the coefficient "4.737" (the reference stores only the coefficient).
# NOTE: Use \d{1,30} instead of \d+ to prevent catastrophic backtracking on
#       very long digit strings (some model predictions can be thousands of
#       digits long).
_LATEX_SCI_RE = re.compile(
    r"([-+]?\d{1,30}(?:\.\d{0,30})?)\s*\\times\s*10\s*\^\s*\{?\s*[-+]?\d{1,10}\s*\}?"
)

META_DETAIL_KEYS = frozenset({"type", "meta", "metadata"})
PER_INSTANCE_SUFFIX = ".per_instance.json"


# ── Dataset Specification ─────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    choices: Tuple[str, ...]
    prediction_keys: Tuple[str, ...]
    reference_keys: Tuple[str, ...]
    output_prediction_key: str
    output_reference_key: str
    default_choice: Optional[str]
    # If the dataset already provides a correctness boolean in details (e.g. any_correct / is_correct),
    # read it directly instead of parsing prediction/reference again.
    correctness_keys: Tuple[str, ...] = ()


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "simpleqa": DatasetSpec(
        name="simpleqa",
        choices=("A", "B", "C"),
        prediction_keys=("prediction",),
        reference_keys=("gold",),
        output_prediction_key="prediction",
        output_reference_key="gold",
        default_choice="C",
    ),
    "c3": DatasetSpec(
        name="c3",
        choices=("A", "B", "C", "D"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        output_prediction_key="predictions",
        output_reference_key="references",
        default_choice=None,
    ),
    "commonsense_qa": DatasetSpec(
        name="commonsense_qa",
        choices=("A", "B", "C", "D", "E"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        output_prediction_key="predictions",
        output_reference_key="references",
        default_choice=None,
    ),
    "openbookqa": DatasetSpec(
        name="openbookqa",
        choices=("A", "B", "C", "D"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        output_prediction_key="predictions",
        output_reference_key="references",
        default_choice=None,
    ),
    "openbookqa_fact": DatasetSpec(
        name="openbookqa_fact",
        choices=("A", "B", "C", "D"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        output_prediction_key="predictions",
        output_reference_key="references",
        default_choice=None,
    ),
    "scibench": DatasetSpec(
        name="scibench",
        choices=(),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference", "answer", "gold"),
        output_prediction_key="predictions",
        output_reference_key="references",
        default_choice=None,
    ),
    # SciCode real records are a raw mapping: {subproblem_id: 0/1}.
    # `_normalize_details` wraps scalar values with output_prediction_key,
    # so using "is_correct" here makes them explicit correctness records.
    "scicode": DatasetSpec(
        name="scicode",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="is_correct",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "is_correct", "any_correct"),
    ),
    "scicode_with_background": DatasetSpec(
        name="scicode",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="is_correct",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "is_correct", "any_correct"),
    ),
    "livecodebench": DatasetSpec(
        name="livecodebench",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="is_correct",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "is_correct", "any_correct"),
    ),
    "longbenchv2_0shot": DatasetSpec(
        name="longbenchv2_0shot",
        choices=("A", "B", "C", "D"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        output_prediction_key="predictions",
        output_reference_key="references",
        default_choice=None,
    ),
    # Code evaluation (MBPP-PLUS): OpenCompass's MBPPEvaluator injects per-item any_correct
    # into details. Read any_correct directly as per-item correctness for majority voting.
    "mbpp_plus": DatasetSpec(
        name="mbpp_plus",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("any_correct", "is_correct"),
    ),
    # ARC-Challenge (Open LLM Leaderboard v1 records converted to explicit
    # per-question correctness). Same shape as swe_bench_verified.
    "arc_challenge": DatasetSpec(
        name="arc_challenge",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "is_correct", "any_correct"),
    ),
    # HellaSwag (Open LLM Leaderboard v1 records converted to explicit
    # per-question correctness). Same shape as arc_challenge.
    "hellaswag": DatasetSpec(
        name="hellaswag",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "is_correct", "any_correct"),
    ),
    # SWE-bench Verified: per-instance only carries a 0/1 correctness flag
    # (see convert_swe_bench_pre_record.py). Same shape as mbpp_plus.
    "swe_bench_verified": DatasetSpec(
        name="swe_bench_verified",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("is_correct", "any_correct"),
    ),
    # SWE-bench Multilingual converted records use the same per-instance
    # correctness shape as SWE-bench Verified.
    "swe_bench_multilingual": DatasetSpec(
        name="swe_bench_multilingual",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("is_correct", "any_correct"),
    ),
    # SWE-bench Pro records use the same per-instance correctness shape:
    # details is id -> {instance_id, is_correct, any_correct}.
    "swe_bench_pro": DatasetSpec(
        name="swe_bench_pro",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("is_correct", "any_correct"),
    ),
    # OSWorld-Verified raw records contain continuous per-task scores in
    # `is_correct`. They must be thresholded first with
    # binarize_continuous_records.py, which writes binary_is_correct.
    "osworld_verified": DatasetSpec(
        name="osworld_verified",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    # Toolathlon converted records use the same explicit correctness shape:
    # details is id -> {instance_id, is_correct, any_correct}.
    "toolathlon": DatasetSpec(
        name="toolathlon",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "is_correct", "any_correct"),
    ),
    # Terminal-Bench raw records contain continuous per-task scores in
    # `is_correct`. They must be thresholded first with
    # binarize_continuous_records.py, which writes binary_is_correct.
    "terminal_bench": DatasetSpec(
        name="terminal_bench",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    # OCRBench_v2 per-item scores are continuous before binarization.
    # binarize_continuous_records.py writes binary_is_correct for voting.
    "ocrbench_v2": DatasetSpec(
        name="ocrbench_v2",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    # ArenaHard converted records carry a continuous per-question soft win
    # rate in `is_correct` (convert_continuous_native_records.py); one global
    # threshold is fitted by binarize_continuous_records.py --task-key '',
    # which writes binary_is_correct.
    "arenahard": DatasetSpec(
        name="arenahard",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    # OmniDocBench edit raw records are continuous score maps and must
    # be binarized before voting.
    "omnidocbench_text_edit": DatasetSpec(
        name="omnidocbench_text_edit",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    "omnidocbench_tab_edit": DatasetSpec(
        name="omnidocbench_tab_edit",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    "omnidocbench_tab_teds": DatasetSpec(
        name="omnidocbench_tab_teds",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    "omnidocbench_formula_edit": DatasetSpec(
        name="omnidocbench_formula_edit",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    "omnidocbench_formula_cdm": DatasetSpec(
        name="omnidocbench_formula_cdm",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    "omnidocbench_read_order_edit": DatasetSpec(
        name="omnidocbench_read_order_edit",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "binary_is_correct"),
    ),
    # TheoremQA: details is a *list* of records, each carries `is_correct`
    # (often wrapped in a 1-element list, e.g. {"is_correct": [true]}).
    # `_normalize_details` handles list -> dict conversion below.
    "theoremqa": DatasetSpec(
        name="theoremqa",
        choices=(),
        prediction_keys=("pred", "prediction", "predictions"),
        reference_keys=(),
        output_prediction_key="pred",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("is_correct", "any_correct"),
    ),
    # tau2_bench domains are stored as flat *.per_instance.json files:
    # {acc,total,correct,results:[{id,instance_id,correct}, ...]}.
    "tau2_bench": DatasetSpec(
        name="tau2_bench",
        choices=(),
        prediction_keys=(),
        reference_keys=(),
        output_prediction_key="prediction",
        output_reference_key="references",
        default_choice=None,
        correctness_keys=("voted_correctness", "correct", "is_correct", "any_correct"),
    ),
}

# Pre-compile choice extraction regexes for datasets that have choices.
CHOICE_RE: Dict[str, re.Pattern[str]] = {
    name: re.compile(r"\b(" + "|".join(spec.choices) + r")\b", re.IGNORECASE)
    for name, spec in DATASET_SPECS.items()
    if spec.choices
}


# ── Logging ───────────────────────────────────────────────────────────

def eprint(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _log_model_list(header: str, models: List[ModelResults]) -> None:
    """Print a labeled model list with accuracy to stderr."""
    eprint(f"[INFO] {header}")
    for m in models:
        src = "top_level" if m.metrics_from_top_level else "computed"
        eprint(f"  - {m.model_name}: accuracy={m.accuracy_pct:.4f} ({src})")


# ── Low-level Helpers ─────────────────────────────────────────────────

def extract_choice_letter(
    value: Any, choice_re: re.Pattern[str], default: Optional[str],
) -> Optional[str]:
    """Extract a choice letter from a string; return default on failure."""
    if value is None:
        return default
    m = choice_re.search(str(value))
    return m.group(1).upper() if m else default


def get_field_by_keys(record: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    """Return the value of the first matching key in record, or None."""
    for key in keys:
        if key in record:
            return record[key]
    return None


def _normalize_accuracy_value(value: Any) -> Optional[float]:
    """Normalize accuracy to [0,1] fraction. Supports both [0,1] and [0,100] input."""
    if value is None:
        return None
    try:
        acc = float(value)
    except (TypeError, ValueError):
        return None
    return acc / 100.0 if acc > 1.0 else acc


def _normalize_bool(value: Any) -> Optional[bool]:
    """Normalize a correctness field to bool / None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _normalize_bool(value[0]) if value else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "1", "yes", "y", "success", "correct", "pass"}:
            return True
        if s in {"false", "0", "no", "n", "fail", "failed", "incorrect"}:
            return False
    return None


# ── SciBench Helpers ──────────────────────────────────────────────────

def _extract_last_number(value: Any) -> Optional[float]:
    """Extract the last numeric value from a string.

    Handles LaTeX scientific notation (e.g. ``4.737 \\times 10^{14}``)
    by converting it to standard form before extraction.
    """
    if value is None:
        return None
    s = str(value)
    # Pre-process: strip LaTeX scientific notation, keeping only the coefficient
    # e.g. "4.737 \times 10^{14}" -> "4.737 "
    s = _LATEX_SCI_RE.sub(r"\1", s)
    matches = list(SCIBENCH_NUMBER_RE.finditer(s))
    if not matches:
        return None
    try:
        return float(matches[-1].group(0))
    except (TypeError, ValueError):
        return None


def _numeric_close(a: float, b: float, *, atol: float, rtol: float) -> bool:
    """Check if two numbers are close within absolute and relative tolerance.

    Formula aligned with OpenCompass NumericAccEvaluator:
        abs(pred - ref) <= atol + rtol * abs(ref)
    where *a* is the prediction and *b* is the reference.
    """
    return abs(a - b) <= atol + rtol * abs(b)


def _normalize_free_text(value: Any) -> Optional[str]:
    """Normalize free-text answer for comparison."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'`.,;:!?")
    return s or None


# ── Correctness Computation ──────────────────────────────────────────

def compute_correctness_for_record(
    record: Dict[str, Any],
    spec: DatasetSpec,
    choice_re: Optional[re.Pattern[str]],
    scibench_atol: float,
    scibench_rtol: float,
) -> Optional[bool]:
    """
    Judge a single record's correctness based on the dataset spec.

    Returns True/False, or None if judgement is impossible.
    """
    if not isinstance(record, dict):
        return None

    # 1) Direct correctness field (e.g. mbpp_plus: any_correct)
    if spec.correctness_keys:
        for key in spec.correctness_keys:
            if key in record:
                return _normalize_bool(record[key])
        return None

    # 2) SimpleQA: A=correct, B/C=incorrect, only prediction matters
    if spec.name == "simpleqa":
        pred_val = get_field_by_keys(record, spec.prediction_keys)
        if pred_val is None:
            return None
        pred_letter = extract_choice_letter(
            pred_val, choice_re, spec.default_choice,
        ) if choice_re is not None else None
        if pred_letter is None:
            return None
        return pred_letter == "A"

    # 3) SciBench: numeric / free-text comparison
    if spec.name == "scibench":
        pred_val = get_field_by_keys(record, spec.prediction_keys)
        ref_val = get_field_by_keys(record, spec.reference_keys)
        if pred_val is None or ref_val is None:
            return None

        pred_num = _extract_last_number(pred_val)
        ref_num = _extract_last_number(ref_val)
        if pred_num is not None and ref_num is not None:
            return _numeric_close(
                pred_num, ref_num, atol=scibench_atol, rtol=scibench_rtol,
            )

        pred_s = _normalize_free_text(pred_val)
        ref_s = _normalize_free_text(ref_val)
        if pred_s is None or ref_s is None:
            return None
        return pred_s == ref_s

    # 3) Standard QA: compare choice letters
    if choice_re is None:
        return None
    pred_val = get_field_by_keys(record, spec.prediction_keys)
    ref_val = get_field_by_keys(record, spec.reference_keys)
    if pred_val is None or ref_val is None:
        return False
    pred_letter = extract_choice_letter(pred_val, choice_re, spec.default_choice)
    ref_letter = extract_choice_letter(ref_val, choice_re, None)
    if pred_letter is None or ref_letter is None:
        return False
    return pred_letter == ref_letter


# ── Model Data ────────────────────────────────────────────────────────

@dataclass
class ModelResults:
    model_name: str
    json_path: Path
    details: Dict[str, Dict[str, Any]]
    correctness: Dict[str, bool]         # id -> True/False
    score: float                          # accuracy as fraction [0, 1]
    metrics_from_top_level: bool
    accuracy_pct: float                   # accuracy as percentage [0, 100]


def list_model_dirs(root: Path) -> List[Path]:
    """List immediate subdirectories of root, sorted by name."""
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda x: x.name,
    )


def _is_tau2_dataset(dataset_key: str) -> bool:
    return dataset_key == "tau2_bench" or dataset_key.startswith("tau2_bench_")


def _get_dataset_spec(dataset_key: str) -> DatasetSpec:
    if _is_tau2_dataset(dataset_key):
        return DATASET_SPECS["tau2_bench"]
    return DATASET_SPECS[dataset_key]


def _derive_per_instance_model_name(path: Path) -> str:
    name = path.name
    if name.endswith(PER_INSTANCE_SUFFIX):
        return name[: -len(PER_INSTANCE_SUFFIX)]
    return path.stem


def list_flat_per_instance_files(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob(f"*{PER_INSTANCE_SUFFIX}"), key=lambda x: x.name)


def _resolve_dataset_json_path(model_dir: Path, dataset_arg: str, dataset_key: str) -> Path:
    """
    Resolve dataset JSON path with tolerant matching.

    This supports case differences such as C3.json vs --dataset c3.
    """
    candidates: List[Path] = []
    for stem in (str(dataset_arg).strip(), dataset_key):
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

    return candidates[0] if candidates else model_dir / f"{dataset_key}.json"


def _filter_model_dirs(
    all_dirs: List[Path], include_models: str, exclude_models: str,
) -> List[Path]:
    """Apply include/exclude filters to model directory list."""
    selected = list(all_dirs)
    if include_models:
        inc = {m.strip() for m in include_models.split(",") if m.strip()}
        selected = [d for d in selected if d.name in inc]
    if exclude_models:
        exc = {m.strip() for m in exclude_models.split(",") if m.strip()}
        selected = [d for d in selected if d.name not in exc]
    return selected


def _record_id(rec: Dict[str, Any], fallback_index: int) -> str:
    """Pick a stable id from a list-style record, or fall back to its index."""
    if isinstance(rec, dict):
        for key in ("example_abbr", "instance_id", "id", "task_id", "example_id", "qid", "index"):
            if key in rec:
                val = rec[key]
                if isinstance(val, (list, tuple)) and val:
                    return str(val[0])
                if val is not None:
                    return str(val)
    return str(fallback_index)


def _normalize_details(data: Any, spec: DatasetSpec) -> Dict[str, Dict[str, Any]]:
    """
    Normalize input JSON to { id: record } mapping.
    Supports:
      - { "details": { id: rec, ... } }   (most datasets)
      - { "details": [ rec, ... ] }       (TheoremQA-style)
      - { id: rec, ... }                   (raw mapping)
    Filters out meta keys.
    """
    if isinstance(data, dict) and "details" in data:
        raw = data["details"]
    elif isinstance(data, dict) and "results" in data:
        raw = data["results"]
    elif isinstance(data, dict):
        raw = data
    else:
        raise ValueError("Input JSON is not an object/dict.")

    details: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw, list):
        for idx, v in enumerate(raw):
            if not isinstance(v, dict):
                v = {spec.output_prediction_key: v}
            qid = _record_id(v, idx)
            details[qid] = v
        return details

    if not isinstance(raw, dict):
        raise ValueError("`details` must be a dict or list.")

    for k, v in raw.items():
        key = str(k)
        if key.lower() in META_DETAIL_KEYS:
            continue
        details[key] = v if isinstance(v, dict) else {spec.output_prediction_key: v}
    return details


def _load_single_model(
    json_path: Path,
    spec: DatasetSpec,
    scibench_atol: float,
    scibench_rtol: float,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, bool]]:
    """Load a single model's JSON. Returns (top_level, details, correctness)."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    details = _normalize_details(data, spec)
    choice_re = CHOICE_RE.get(spec.name)
    correctness: Dict[str, bool] = {}

    for qid, rec in details.items():
        result = compute_correctness_for_record(
            rec, spec, choice_re,
            scibench_atol=scibench_atol,
            scibench_rtol=scibench_rtol,
        )
        if result is not None:
            correctness[qid] = result

    top_level = data if isinstance(data, dict) else {}
    return top_level, details, correctness


def build_models_info(
    root: Path,
    dataset_key: str,
    dataset_arg: str,
    include_models: str,
    exclude_models: str,
    scibench_atol: float,
    scibench_rtol: float,
) -> List[ModelResults]:
    """Load all model results from root directory."""
    spec = _get_dataset_spec(dataset_key)
    model_dirs = _filter_model_dirs(
        list_model_dirs(root), include_models, exclude_models,
    )
    flat_files = list_flat_per_instance_files(root) if _is_tau2_dataset(dataset_key) else []

    if not model_dirs and not flat_files:
        eprint("[ERROR] No model folders or tau2 per-instance files found after include/exclude filtering.")
        sys.exit(2)

    models: List[ModelResults] = []
    missing = bad = 0

    model_sources: List[Tuple[str, Path]] = []
    for mdir in model_dirs:
        json_path = _resolve_dataset_json_path(mdir, dataset_arg, dataset_key)
        if not json_path.exists():
            eprint(f"[WARN] Missing dataset file for model '{mdir.name}': {json_path}")
            missing += 1
            continue
        model_sources.append((mdir.name, json_path))

    for json_path in flat_files:
        model_name = _derive_per_instance_model_name(json_path)
        if include_models:
            inc = {m.strip() for m in include_models.split(",") if m.strip()}
            if model_name not in inc:
                continue
        if exclude_models:
            exc = {m.strip() for m in exclude_models.split(",") if m.strip()}
            if model_name in exc:
                continue
        model_sources.append((model_name, json_path))

    for model_name, json_path in model_sources:
        try:
            top_level, details, correctness = _load_single_model(
                json_path, spec,
                scibench_atol=scibench_atol,
                scibench_rtol=scibench_rtol,
            )

            # Compute or read model-level accuracy
            total = len(correctness)
            correct = sum(1 for v in correctness.values() if v)
            computed_frac = correct / total if total else 0.0

            top_acc = _normalize_accuracy_value(top_level.get("accuracy"))
            if top_acc is None and _is_tau2_dataset(dataset_key):
                top_acc = _normalize_accuracy_value(top_level.get("acc"))
            if top_acc is not None:
                acc_frac = top_acc
                from_top = True
            else:
                acc_frac = computed_frac
                from_top = False

            models.append(ModelResults(
                model_name=model_name,
                json_path=json_path,
                details=details,
                correctness=correctness,
                score=acc_frac,
                metrics_from_top_level=from_top,
                accuracy_pct=acc_frac * 100.0,
            ))
        except Exception as e:
            eprint(f"[WARN] Skipping bad json for model '{model_name}': {json_path} ({e})")
            bad += 1

    if not models:
        eprint("[ERROR] No valid model JSON files found.")
        sys.exit(2)

    eprint(f"[INFO] Loaded {len(models)} model JSON files. (missing={missing}, bad={bad})")
    return models


# ── Model Selection ───────────────────────────────────────────────────

def filter_models_by_performance(
    models: List[ModelResults], min_acc: float, max_acc: float,
) -> List[ModelResults]:
    """Filter models whose score is within [min_acc, max_acc]."""
    filtered = [m for m in models if min_acc <= m.score <= max_acc]
    eprint(
        f"[INFO] Filtered models by accuracy [{min_acc:.3f}, {max_acc:.3f}]: "
        f"{len(filtered)}/{len(models)} remain."
    )
    if not filtered:
        eprint("[ERROR] No models remain after performance filtering.")
        sys.exit(2)
    return filtered


def _weighted_sample_no_replace(
    items: List[ModelResults],
    weights: List[float],
    k: int,
    rng: random.Random,
) -> List[ModelResults]:
    """Weighted sampling without replacement."""
    if k > len(items):
        raise ValueError(f"Cannot sample k={k} from n={len(items)} items")

    pool = list(zip(items, weights))
    selected: List[ModelResults] = []

    for _ in range(k):
        total_w = sum(w for _, w in pool)
        if total_w <= 0:
            idx = rng.randrange(len(pool))
        else:
            idx = rng.choices(
                range(len(pool)),
                weights=[w / total_w for _, w in pool],
                k=1,
            )[0]
        selected.append(pool.pop(idx)[0])

    return selected


def select_models(
    models: List[ModelResults],
    k: int,
    weak_boost: float,
    rng: random.Random,
) -> List[ModelResults]:
    """Select k models via random (optionally weighted) sampling."""
    if len(models) < k:
        eprint(f"[ERROR] Not enough models: have {len(models)}, need {k}")
        sys.exit(2)

    if weak_boost == 1.0:
        selected = rng.sample(models, k)
    else:
        # weight = (1 - acc) ^ weak_boost  =>  higher weight for weaker models
        weights = [
            max(0.0, min(1.0, 1.0 - m.score)) ** weak_boost
            for m in models
        ]
        selected = _weighted_sample_no_replace(models, weights, k, rng)

    _log_model_list("Selected models:", selected)
    return selected


# ── Voting ────────────────────────────────────────────────────────────

def _vote_majority(votes: List[bool], rng: random.Random) -> Tuple[bool, bool]:
    """
    Majority vote on True/False list.
    Returns (chosen_value, is_tie).
    """
    if not votes:
        return False, False
    counts = Counter(votes)
    max_ct = max(counts.values())
    top_vals = [val for val, ct in counts.items() if ct == max_ct]
    tie = len(top_vals) > 1
    return (rng.choice(top_vals) if tie else top_vals[0]), tie


def _qid_sort_key(qid: str) -> Tuple[int, Any]:
    """Sort key: numeric IDs first (by value), then strings alphabetically."""
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def _build_ordered_ids(selected: List[ModelResults]) -> List[str]:
    """
    Build a deterministic ordering of all question IDs across selected models.
    Uses the first model's key order as base, then appends extras sorted.
    """
    union_ids = {qid for m in selected for qid in m.details}

    # Use first non-empty model's order as base
    base_order: List[str] = []
    for m in selected:
        if m.details:
            base_order = list(m.details.keys())
            break

    seen: set = set()
    ordered: List[str] = []

    for qid in base_order:
        if qid in union_ids and qid not in seen:
            ordered.append(qid)
            seen.add(qid)

    extras = sorted(
        (qid for qid in union_ids if qid not in seen), key=_qid_sort_key,
    )
    ordered.extend(extras)
    return ordered


def build_ensemble_results(
    selected: List[ModelResults], rng: random.Random,
) -> Dict[str, Any]:
    """
    Build ensemble results via majority vote on correctness.

    Output format (compatible with batch script metric reading):
      { "is_correct": <frac>, "accuracy": <pct>, "details": { ... } }
    """
    all_ids = _build_ordered_ids(selected)
    details_out: Dict[str, Dict[str, Any]] = {}
    total = correct = ties = missing_votes = 0

    for qid in all_ids:
        votes: List[bool] = []
        first_record: Optional[Dict[str, Any]] = None

        for m in selected:
            rec = m.details.get(qid)
            if first_record is None and isinstance(rec, dict):
                first_record = rec
            c = m.correctness.get(qid)
            if c is not None:
                votes.append(c)

        if not votes:
            missing_votes += 1
            continue

        chosen, tie = _vote_majority(votes, rng)
        ties += tie
        total += 1
        correct += chosen  # bool(True) == 1

        base_rec = copy.deepcopy(first_record) if first_record else {}
        base_rec["voted_correctness"] = chosen
        details_out[qid] = base_rec

    acc_frac = correct / total if total else 0.0
    acc_pct = acc_frac * 100.0

    eprint(
        f"[INFO] Ensemble built: {len(details_out)} items, "
        f"ties={ties}, ids_with_no_votes={missing_votes}"
    )
    eprint(f"[INFO] Ensemble metrics: accuracy: {acc_pct:.6f} (fraction={acc_frac:.6f})")

    # Compatible with the batch script's metric reader:
    # - commonsense_qa / scibench etc. read the top-level "accuracy" (percentage)
    # - simpleqa etc. read the top-level "is_correct" (0-1)
    return {
        "is_correct": acc_frac,
        "accuracy": acc_pct,
        "details": details_out,
    }


# ── Argument Parsing ──────────────────────────────────────────────────

def _parse_selected_identifiers(raw_args: Optional[List[str]]) -> List[str]:
    """Parse --selected-identifiers (supports multiple invocations + comma separation)."""
    if not raw_args:
        return []
    out: List[str] = []
    for raw in raw_args:
        if raw is None:
            continue
        out.extend(p.strip() for p in str(raw).split(",") if p.strip())
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Offline voting on correctness (True/False) over OpenCompass results JSONs.",
    )
    ap.add_argument("--root", required=True, help="Root directory containing per-model folders.")
    ap.add_argument(
        "--dataset", required=True,
        help=f"Dataset name. Supported: {', '.join(sorted(DATASET_SPECS))}",
    )
    ap.add_argument("--k", type=int, default=None, help="Number of models to select for voting.")
    ap.add_argument("--out", default=None, help="Output results JSON path.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--include-models", default="", help="Comma-separated model names to include.")
    ap.add_argument("--exclude-models", default="", help="Comma-separated model names to exclude.")
    ap.add_argument("--min-acc", type=float, default=0.0, help="Min accuracy threshold (0-1).")
    ap.add_argument("--max-acc", type=float, default=1.0, help="Max accuracy threshold (0-1).")
    ap.add_argument(
        "--weak-boost", type=float, default=1.0,
        help="Weight exponent on (1-acc). >1 boosts weaker, =1 uniform, <1 boosts stronger.",
    )
    ap.add_argument(
        "--print-candidates", action="store_true",
        help="Print filtered candidate model names to stdout and exit.",
    )
    ap.add_argument(
        "--scibench-atol", type=float, default=SCIBENCH_ATOL_DEFAULT,
        help="SciBench numeric absolute tolerance.",
    )
    ap.add_argument(
        "--scibench-rtol", type=float, default=SCIBENCH_RTOL_DEFAULT,
        help="SciBench numeric relative tolerance.",
    )
    ap.add_argument(
        "--selected-identifiers", action="append", default=None,
        help=(
            "Fixed model names for voting (overrides random selection). "
            "Can be provided multiple times and/or comma-separated."
        ),
    )
    return ap.parse_args()


# ── Fixed Selection Logic ─────────────────────────────────────────────

def _resolve_fixed_selection(
    selected_ids: List[str],
    k: int,
    filtered: List[ModelResults],
) -> List[ModelResults]:
    """Resolve --selected-identifiers to ModelResults, with validation."""
    if len(selected_ids) != k:
        eprint(
            f"[ERROR] --selected-identifiers count ({len(selected_ids)}) "
            f"must equal --k ({k})."
        )
        sys.exit(2)
    if len(set(selected_ids)) != len(selected_ids):
        eprint("[ERROR] --selected-identifiers contains duplicates.")
        sys.exit(2)

    by_name = {m.model_name: m for m in filtered}
    missing = [name for name in selected_ids if name not in by_name]
    if missing:
        eprint("[ERROR] Selected identifiers not in filtered candidates:")
        for name in missing:
            eprint(f"  - {name}")
        sys.exit(2)

    selected = [by_name[name] for name in selected_ids]
    _log_model_list(f"Using fixed selection of {len(selected)} models:", selected)
    return selected


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    dataset_key = args.dataset.strip().lower()
    dataset_arg = args.dataset.strip()
    if dataset_key not in DATASET_SPECS and not _is_tau2_dataset(dataset_key):
        eprint(
            f"[ERROR] Unsupported dataset '{args.dataset}'. "
            f"Supported: {sorted(DATASET_SPECS)} plus tau2_bench_* aliases"
        )
        sys.exit(2)

    rng = random.Random(args.seed)
    root = Path(args.root)

    models = build_models_info(
        root, dataset_key, dataset_arg,
        args.include_models, args.exclude_models,
        scibench_atol=args.scibench_atol,
        scibench_rtol=args.scibench_rtol,
    )

    _log_model_list(
        "Model performance summary:",
        sorted(models, key=lambda x: -x.score),
    )

    filtered = filter_models_by_performance(models, args.min_acc, args.max_acc)

    # --print-candidates mode: output names and exit
    if args.print_candidates:
        for m in filtered:
            print(m.model_name)
        return

    # Validate required args for voting mode
    if not args.k or args.k <= 0:
        eprint("[ERROR] --k is required and must be positive (unless --print-candidates).")
        sys.exit(2)
    if not args.out:
        eprint("[ERROR] --out is required (unless --print-candidates).")
        sys.exit(2)

    # Resolve model selection
    selected_ids = _parse_selected_identifiers(args.selected_identifiers)
    if selected_ids:
        selected = _resolve_fixed_selection(selected_ids, args.k, filtered)
    else:
        selected = select_models(filtered, args.k, args.weak_boost, rng)

    # Build and write ensemble
    out_obj = build_ensemble_results(selected, rng)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=4)
    eprint(f"[INFO] Wrote ensemble results to {out_path}")


if __name__ == "__main__":
    main()
