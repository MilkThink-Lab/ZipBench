#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing script (JSON version): aggregate the results JSON records of multiple models on a test set into the pkl structure required downstream.

Input directory layout (aligned with vote_offline_results_json.py):
root_dir_1/
  model_A/
    simpleqa.json
    commonsense_qa.json
  model_B/
    simpleqa.json
    commonsense_qa.json
root_dir_2/
  model_C/
    commonsense_qa.json
...

Each `<dataset_name>.json` may take one of two forms:
1) OpenCompass results-like:
   {
     "accuracy_given_attempted": ...,
     "f1": ...,
     "details": { "<id>": { "prediction": "...", ... }, ... }
   }
2) Raw mapping:
   { "<id>": { "prediction": "...", ... }, ... }

This script maps records to correctness with the following rules:
  - SimpleQA: parse A/B/C from the prediction (case-insensitive, regex)
    - A => correctness = 1
    - B/C (including NOT_ATTEMPTED) => correctness = 0
  - CommonsenseQA: parse A/B/C/D/E from prediction/reference (case-insensitive, regex)
    - prediction == reference => correctness = 1
    - prediction != reference => correctness = 0
- If the reference is missing/unparseable => the id counts as wrong (correctness = 0)
  - If the prediction is missing/unparseable => counts as wrong by default (correctness = 0)
  - If a record carries an explicit correctness field (e.g. voted_correctness / any_correct / is_correct),
    that field takes precedence and is mapped to {1,0}, so template prediction/reference values cannot mislead.
  - Humaneval_plus: use the per-record is_correct (true/false)
  - Mbpp_plus: use the per-record any_correct (true/false)
  - Scibench: numeric comparison with tolerance (default atol=1e-3, rtol=1e-2, aligned with
    OpenCompass NumericAccEvaluator); falls back to normalized text comparison when no number can be parsed
  - A model missing an id => correctness = 0 (missing counts as wrong)

Output pkl structure (identical to the legacy aggregate_results.py):
{
  "models": [model_name_0, ...],
  "data": {
    dataset_name: {
      "correctness": np.ndarray(shape=(num_questions, num_models), dtype=int)
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


META_DETAIL_KEYS = {"type", "meta", "metadata"}
PER_INSTANCE_SUFFIX = ".per_instance.json"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    choices: Tuple[str, ...]
    prediction_keys: Tuple[str, ...]
    reference_keys: Tuple[str, ...]
    default_choice: Optional[str]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "simpleqa": DatasetSpec(
        name="simpleqa",
        choices=("A", "B", "C"),
        prediction_keys=("prediction", "predictions"),
        reference_keys=("gold",),
        default_choice="C",
    ),
    "commonsense_qa": DatasetSpec(
        name="commonsense_qa",
        choices=("A", "B", "C", "D", "E"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        default_choice=None,
    ),
    "openbookqa": DatasetSpec(
        name="openbookqa",
        choices=("A", "B", "C", "D"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        default_choice=None,
    ),
    "openbookqa_fact": DatasetSpec(
        name="openbookqa_fact",
        choices=("A", "B", "C", "D"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        default_choice=None,
    ),
    "c3": DatasetSpec(
        name="c3",
        choices=("A", "B", "C", "D"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        default_choice=None,
    ),
    "longbenchv2_0shot": DatasetSpec(
        name="longbenchv2_0shot",
        choices=("A", "B", "C", "D"),
        prediction_keys=("predictions", "prediction"),
        reference_keys=("references", "reference"),
        default_choice=None,
    ),
}


CHOICE_RE: Dict[str, re.Pattern[str]] = {
    name: re.compile(r"\b(" + "|".join(spec.choices) + r")\b", re.IGNORECASE)
    for name, spec in DATASET_SPECS.items()
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate model results JSON records into a pkl correctness matrix")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        nargs="+",
        help="Root directory containing one sub-folder per model (multiple roots may be given and are merged)",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (corresponds to <dataset_name>.json)")
    parser.add_argument("--output_path", type=str, default=None, help="Output pkl file path")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuous mode: read the float score in details[qid].is_correct ([0,1]) as-is, without binarization, and emit a float matrix",
    )
    parser.add_argument(
        "--scibench-atol",
        type=float,
        default=SCIBENCH_ATOL_DEFAULT,
        help="Absolute tolerance for Scibench numeric comparison (default 1e-3)",
    )
    parser.add_argument(
        "--scibench-rtol",
        type=float,
        default=SCIBENCH_RTOL_DEFAULT,
        help="Relative tolerance for Scibench numeric comparison (default 1e-2)",
    )
    return parser.parse_args()


def _normalize_dataset_name(name: str) -> str:
    return str(name).strip().lower()


def _normalize_is_correct(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _normalize_is_correct(value[0])
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


def _extract_explicit_correctness(record: Dict[str, Any]) -> Optional[bool]:
    """
    Extract an explicit correctness boolean from a single record, if present.

    Supported sources:
    - vote_offline_results_json_correctness.py: voted_correctness
    - code evaluators or other scripts: any_correct / is_correct
    - MMStar/VLMEvalKit evaluation details: hit / if_right
    """
    if not isinstance(record, dict):
        return None
    for key in ("voted_correctness", "any_correct", "is_correct", "hit", "if_right"):
        if key in record:
            val = _normalize_is_correct(record.get(key))
            if val is not None:
                return bool(val)
    lower_record = {str(k).strip().lower(): v for k, v in record.items()}
    for key in ("voted_correctness", "any_correct", "is_correct", "hit", "if_right"):
        if key in lower_record:
            val = _normalize_is_correct(lower_record.get(key))
            if val is not None:
                return bool(val)
    return None


def _normalize_simplevqa_judge(value: Any) -> Optional[bool]:
    """Map SimpleVQA judge labels to correctness."""
    if isinstance(value, dict):
        if "model_response" in value:
            return _normalize_simplevqa_judge(value.get("model_response"))
        return None
    if isinstance(value, (list, tuple)):
        return _normalize_simplevqa_judge(value[0]) if value else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return None

    text = re.sub(r"\s+", "", str(value).strip().lower())
    if not text:
        return None
    # The SimpleVQA judge emits Chinese verdicts (正确 = correct, 错误/不正确 = wrong,
    # 未尝试 = not attempted); these literals must stay to match the raw records.
    if text in {"正确", "correct", "true", "1", "yes", "y"}:
        return True
    if text in {
        "错误", "不正确", "未尝试", "incorrect", "wrong",
        "not_attempted", "notattempted", "false", "0", "no", "n",
    }:
        return False
    if "未尝试" in text or "错误" in text or "不正确" in text:
        return False
    if "正确" in text or "correct" in text:
        return True
    return None


def _is_simplevqa_correct(record: Dict[str, Any]) -> Optional[bool]:
    explicit = _extract_explicit_correctness(record)
    if explicit is not None:
        return explicit
    if "judge_res" in record:
        parsed = _normalize_simplevqa_judge(record.get("judge_res"))
        if parsed is not None:
            return parsed
    return _normalize_simplevqa_judge(record.get("judge_res.model_response"))


def _record_key(rec: Dict[str, Any], fallback_index: int) -> str:
    if "example_abbr" in rec:
        return str(rec.get("example_abbr"))
    for key in ("instance_id", "id", "index", "task_id", "example_id", "reference"):
        if key in rec:
            val = rec.get(key)
            if isinstance(val, (list, tuple)) and val:
                return str(val[0])
            if val is not None:
                return str(val)
    return str(fallback_index)


def _details_from_results_json(payload: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize input JSON into details mapping id->record dict."""
    if not isinstance(payload, dict):
        raise TypeError("Top-level JSON must be an object/dict.")

    if "details" in payload and isinstance(payload["details"], dict):
        raw_details = payload["details"]
    else:
        raw_details = payload

    details: Dict[str, Dict[str, Any]] = {}
    for k, v in raw_details.items():
        key = str(k)
        if key.lower() in META_DETAIL_KEYS:
            continue
        if isinstance(v, dict):
            details[key] = v
        else:
            # fallback: wrap non-dict records
            details[key] = {"prediction": v}
    return details


def _id_to_correctness_from_details_list(details: List[Any]) -> Dict[str, int]:
    id_to_correctness: Dict[str, int] = {}
    for idx, rec in enumerate(details):
        if not isinstance(rec, dict):
            continue
        qid = _record_key(rec, idx)
        is_corr = _extract_explicit_correctness(rec)
        if is_corr is None:
            continue
        id_to_correctness[str(qid)] = 1 if is_corr else 0
    return id_to_correctness


def _id_to_explicit_correctness_from_payload(payload: Any) -> Dict[str, int]:
    """Extract correctness from payloads that already carry hit/is_correct/voted_correctness.

    Supports three on-disk shapes:
      - top-level list of records (e.g. some MMStar/VLMEvalKit dumps)
      - { "details": [ ... ], ... }  (e.g. TheoremQA real records)
      - { "details": { id: rec, ... }, ... } or raw { id: rec, ... }
    """
    if isinstance(payload, list):
        return _id_to_correctness_from_details_list(payload)

    if isinstance(payload, dict) and isinstance(payload.get("details"), list):
        return _id_to_correctness_from_details_list(payload["details"])

    details = _details_from_results_json(payload)
    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details.items():
        if not isinstance(rec, dict):
            continue
        explicit = _extract_explicit_correctness(rec)
        if explicit is None:
            continue
        id_to_correctness[str(qid)] = 1 if explicit else 0
    return id_to_correctness


def _extract_last_number(value: Any) -> Optional[float]:
    """Extract the last numeric value from a string (for Scibench).

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
    """Normalize free-text answer for comparison (for Scibench fallback)."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    # Remove LaTeX boxed wrapper
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'`.,;:!?")
    return s or None


def _id_to_correctness_scibench(
    details: Dict[str, Dict[str, Any]],
    atol: float,
    rtol: float,
) -> Dict[str, int]:
    """
    Extract correctness for Scibench dataset.
    Priority: explicit correctness > numeric comparison > free-text comparison.
    """
    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details.items():
        if not isinstance(rec, dict):
            continue

        # 1) Try explicit correctness field first
        explicit = _extract_explicit_correctness(rec)
        if explicit is not None:
            id_to_correctness[str(qid)] = 1 if explicit else 0
            continue

        # 2) Try numeric comparison
        pred_val = rec.get("predictions") or rec.get("prediction")
        ref_val = rec.get("references") or rec.get("reference") or rec.get("answer") or rec.get("gold")

        if pred_val is None or ref_val is None:
            continue

        pred_num = _extract_last_number(pred_val)
        ref_num = _extract_last_number(ref_val)

        if pred_num is not None and ref_num is not None:
            is_close = _numeric_close(pred_num, ref_num, atol=atol, rtol=rtol)
            id_to_correctness[str(qid)] = 1 if is_close else 0
            continue

        # 3) Fallback: normalized free-text comparison
        pred_text = _normalize_free_text(pred_val)
        ref_text = _normalize_free_text(ref_val)
        if pred_text is not None and ref_text is not None:
            id_to_correctness[str(qid)] = 1 if pred_text == ref_text else 0

    return id_to_correctness


def _id_to_correctness_from_details_map_any_correct(
    details: Dict[str, Dict[str, Any]],
    key: str = "any_correct",
) -> Dict[str, int]:
    """mbpp_plus: real records usually carry any_correct; synthetic records prefer voted_correctness."""
    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details.items():
        if not isinstance(rec, dict):
            continue
        explicit = _extract_explicit_correctness(rec)
        if explicit is not None:
            id_to_correctness[str(qid)] = 1 if explicit else 0
            continue
        v = _normalize_is_correct(rec.get(key))
        if v is None:
            continue
        id_to_correctness[str(qid)] = 1 if v else 0
    return id_to_correctness


def _id_to_osworld_verified_correctness(payload: Any) -> Dict[str, int]:
    """Extract OSWorld-Verified correctness after threshold binarization.

    Raw OSWorld-Verified records use continuous per-task `is_correct` scores.
    To avoid silently mapping every non-zero score to True, only
    `voted_correctness` and `binary_is_correct` are accepted here.
    """
    details = _details_from_results_json(payload)
    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details.items():
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
        id_to_correctness[str(qid)] = 1 if flag else 0
    return id_to_correctness


def _id_to_ocrbench_v2_correctness(payload: Any) -> Dict[str, int]:
    """Extract OCRBench_v2 correctness after per-task threshold binarization."""
    details = _details_from_results_json(payload)
    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details.items():
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
        id_to_correctness[str(qid)] = 1 if flag else 0
    return id_to_correctness


def _id_to_terminal_bench_correctness(payload: Any) -> Dict[str, int]:
    """Extract Terminal-Bench correctness after threshold binarization.

    Raw Terminal-Bench records use continuous per-task `correct` scores. To
    avoid silently mapping every non-zero score to True, only
    `voted_correctness` and `binary_is_correct` are accepted here.
    """
    if isinstance(payload, dict) and isinstance(payload.get("details"), list):
        details_iter = (
            (str(_record_key(rec, idx)), rec)
            for idx, rec in enumerate(payload["details"])
            if isinstance(rec, dict)
        )
    else:
        details = _details_from_results_json(payload)
        details_iter = details.items()

    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details_iter:
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
        id_to_correctness[str(qid)] = 1 if flag else 0
    return id_to_correctness


def _id_to_scicode_correctness(payload: Any) -> Dict[str, int]:
    """Extract SciCode correctness from raw id -> 0/1 maps or generated vote JSON."""
    if isinstance(payload, dict) and "details" not in payload:
        id_to_correctness: Dict[str, int] = {}
        for qid, value in payload.items():
            if str(qid).lower() in META_DETAIL_KEYS:
                continue
            if isinstance(value, dict):
                explicit = _extract_explicit_correctness(value)
                if explicit is not None:
                    id_to_correctness[str(qid)] = 1 if explicit else 0
                continue
            flag = _normalize_is_correct(value)
            id_to_correctness[str(qid)] = 1 if flag else 0
        return id_to_correctness
    return _id_to_explicit_correctness_from_payload(payload)


def _id_to_tau2_correctness(payload: Any) -> Dict[str, int]:
    """Extract tau2_bench correctness from flat per-instance files or vote JSON."""
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        id_to_correctness: Dict[str, int] = {}
        for idx, rec in enumerate(payload["results"]):
            if not isinstance(rec, dict):
                continue
            qid = _record_key(rec, idx)
            flag = _normalize_is_correct(rec.get("correct"))
            if flag is None:
                flag = _normalize_is_correct(rec.get("is_correct"))
            if flag is None:
                flag = _normalize_is_correct(rec.get("any_correct"))
            if flag is None:
                continue
            id_to_correctness[str(qid)] = 1 if flag else 0
        return id_to_correctness
    return _id_to_explicit_correctness_from_payload(payload)


def _is_tau2_dataset(dataset_key: str) -> bool:
    return dataset_key == "tau2_bench" or dataset_key.startswith("tau2_bench_")


def _derive_per_instance_model_name(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(PER_INSTANCE_SUFFIX):
        return name[: -len(PER_INSTANCE_SUFFIX)]
    return os.path.splitext(name)[0]


def _is_countbenchqa_correct(record: Dict[str, Any]) -> Optional[bool]:
    """CountBenchQA official evaluation: answer string is a substring of prediction."""
    explicit = _extract_explicit_correctness(record)
    if explicit is not None:
        return explicit
    prediction = record.get("prediction")
    answer = record.get("answer")
    if prediction is None or answer is None:
        return None
    pred_str = str(prediction)
    ans_str = str(answer)
    if not pred_str or not ans_str:
        return None
    return ans_str in pred_str


def _id_to_countbenchqa_correctness(payload: Any) -> Dict[str, int]:
    """Extract CountBenchQA correctness from real xlsx-converted JSON or vote JSON."""
    if isinstance(payload, list):
        details_iter = ((str(_record_key(rec, idx)), rec) for idx, rec in enumerate(payload) if isinstance(rec, dict))
    else:
        details = _details_from_results_json(payload)
        details_iter = details.items()

    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details_iter:
        if not isinstance(rec, dict):
            continue
        is_corr = _is_countbenchqa_correct(rec)
        if is_corr is None:
            continue
        id_to_correctness[str(qid)] = 1 if is_corr else 0
    return id_to_correctness


def _id_to_simplevqa_correctness(payload: Any) -> Dict[str, int]:
    """Extract SimpleVQA correctness from judged eval JSON or vote JSON."""
    if isinstance(payload, list):
        details_iter = (
            (str(_record_key(rec, idx)), rec)
            for idx, rec in enumerate(payload)
            if isinstance(rec, dict)
        )
    else:
        details = _details_from_results_json(payload)
        details_iter = details.items()

    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details_iter:
        if not isinstance(rec, dict):
            continue
        is_corr = _is_simplevqa_correct(rec)
        if is_corr is None:
            continue
        id_to_correctness[str(qid)] = 1 if is_corr else 0
    return id_to_correctness


_CHOICE_LETTER_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def _extract_mathvista_choice_letter(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) == 1 and text.isalpha():
        return text.upper()
    match = _CHOICE_LETTER_RE.search(text)
    return match.group(1).upper() if match else None


def _compare_normalized_text(prediction: Any, answer: Any) -> Optional[bool]:
    if prediction is None or answer is None:
        return None
    pred = re.sub(r"\s+", " ", str(prediction).strip().lower())
    ans = re.sub(r"\s+", " ", str(answer).strip().lower())
    if not pred or not ans:
        return None
    return pred == ans or ans in pred or pred in ans


def _is_mathvista_correct(record: Dict[str, Any]) -> Optional[bool]:
    explicit = _extract_explicit_correctness(record)
    if explicit is not None:
        return explicit

    response = record.get("res")
    if response is None:
        response = record.get("prediction")

    question_type = str(record.get("question_type") or "").strip().lower()
    answer_type = str(record.get("answer_type") or "").strip().lower()

    if question_type == "multi_choice":
        pred_letter = _extract_mathvista_choice_letter(response)
        ans_letter = _extract_mathvista_choice_letter(record.get("answer_option"))
        if pred_letter is None or ans_letter is None:
            return False
        return pred_letter == ans_letter

    if answer_type == "integer":
        try:
            return int(float(str(response).strip())) == int(float(str(record.get("answer")).strip()))
        except (OverflowError, TypeError, ValueError):
            return False

    if answer_type == "float":
        try:
            return float(str(response).strip()) == float(str(record.get("answer")).strip())
        except (OverflowError, TypeError, ValueError):
            return False

    return bool(_compare_normalized_text(response, record.get("answer")))


def _id_to_mathvista_correctness(payload: Any) -> Dict[str, int]:
    """Extract MathVista correctness from xlsx-converted real JSON or vote JSON."""
    if isinstance(payload, list):
        details_iter = (
            (str(_record_key(rec, idx)), rec)
            for idx, rec in enumerate(payload)
            if isinstance(rec, dict)
        )
    else:
        details = _details_from_results_json(payload)
        details_iter = details.items()

    id_to_correctness: Dict[str, int] = {}
    for qid, rec in details_iter:
        if not isinstance(rec, dict):
            continue
        is_corr = _is_mathvista_correct(rec)
        if is_corr is None:
            continue
        id_to_correctness[str(qid)] = 1 if is_corr else 0
    return id_to_correctness


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _extract_choice_letter(value: Any, pattern: re.Pattern[str], default: Optional[str]) -> Optional[str]:
    if value is None:
        return default
    val = _unwrap_value(value)
    if val is None:
        return default
    match = pattern.search(str(val))
    return match.group(1).upper() if match else default


def _get_field_value(record: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record.get(key)
    return None


def load_model_json_continuous(model_json_path: str, dataset_name: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Continuous mode: read the float score in details[qid].is_correct ([0,1]) as-is, without binarization.

    Note: this does NOT go through _extract_explicit_correctness (which prefers voted_correctness/any_correct;
    any_correct is a per-attempt max and is wrong under continuous semantics). The is_correct field must be read exactly.

    Returns:
      - id_to_score: id -> float in [0,1]
      - metrics: num_samples, num_correct (float total score), accuracy (mean)
    """
    with open(model_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    details = _details_from_results_json(payload)
    id_to_score: Dict[str, float] = {}
    for qid, rec in details.items():
        if not isinstance(rec, dict) or "is_correct" not in rec:
            continue
        try:
            id_to_score[str(qid)] = float(rec["is_correct"])
        except (TypeError, ValueError):
            continue

    num_samples = len(id_to_score)
    total = float(sum(id_to_score.values()))
    metrics = {
        "num_samples": int(num_samples),
        "num_correct": total,
        "accuracy": total / num_samples if num_samples else 0.0,
    }
    return id_to_score, metrics


def load_model_json(
    model_json_path: str,
    dataset_name: str,
    atol: float = SCIBENCH_ATOL_DEFAULT,
    rtol: float = SCIBENCH_RTOL_DEFAULT,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Returns:
      - id_to_correctness: id -> {1,0}
      - metrics: num_samples, num_correct, accuracy
    """
    with open(model_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    dataset_key = _normalize_dataset_name(dataset_name)

    if dataset_key == "humaneval_plus":
        if isinstance(payload, dict) and isinstance(payload.get("details"), list):
            id_to_correctness = _id_to_correctness_from_details_list(payload["details"])
        elif isinstance(payload, list):
            id_to_correctness = _id_to_correctness_from_details_list(payload)
        else:
            id_to_correctness = {}
    elif dataset_key in {"swe_bench_verified", "swe_bench_multilingual", "swe_bench_pro", "livecodebench", "arc_challenge", "hellaswag"}:
        # SWE-bench-style records: details is a dict id ->
        # {"is_correct": 0/1, ...}.
        # For voted ensemble files, prefer voted_correctness over the
        # inherited is_correct/any_correct from a source model.
        id_to_correctness = _id_to_explicit_correctness_from_payload(payload)
    elif dataset_key == "osworld_verified":
        # OSWorld-Verified must be binarized from continuous per-task scores
        # before aggregation. See binarize_continuous_records.py.
        id_to_correctness = _id_to_osworld_verified_correctness(payload)
    elif dataset_key == "ocrbench_v2":
        # OCRBench_v2 must be binarized from continuous per-item scores
        # before aggregation. See binarize_continuous_records.py.
        id_to_correctness = _id_to_ocrbench_v2_correctness(payload)
    elif dataset_key == "toolathlon":
        # Toolathlon converted records have the same explicit correctness
        # fields as SWE-bench-style agentic records.
        id_to_correctness = _id_to_explicit_correctness_from_payload(payload)
    elif dataset_key == "terminal_bench":
        # Terminal-Bench must be binarized from continuous per-task scores
        # before aggregation. See binarize_continuous_records.py.
        id_to_correctness = _id_to_terminal_bench_correctness(payload)
    elif dataset_key in {"omnidocbench_text_edit", "omnidocbench_tab_edit", "omnidocbench_tab_teds", "omnidocbench_formula_edit", "omnidocbench_formula_cdm", "omnidocbench_read_order_edit", "arenahard"}:
        # OmniDocBench subsets and ArenaHard must be binarized from continuous
        # per-item scores before (binary) aggregation, see
        # binarize_continuous_records.py. Only binary/voted fields are trusted.
        id_to_correctness = _id_to_terminal_bench_correctness(payload)
    elif _is_tau2_dataset(dataset_key):
        id_to_correctness = _id_to_tau2_correctness(payload)
    elif dataset_key in {"mmstar", "mmmu_dev_val", "mmmu_pro_10c", "mmmu_pro_v", "logicvista", "realworldqa", "mmvp", "spatialeval"}:
        # MMStar / MMMU_DEV_VAL / MMMU_Pro_10c / MMMU_Pro_V / LogicVista / RealWorldQA / MMVP / SpatialEval
        # Real records are usually list[record] with hit/if_right on each entry;
        # synthetic records are vote JSON whose details carry voted_correctness.
        id_to_correctness = _id_to_explicit_correctness_from_payload(payload)
    elif dataset_key == "theoremqa":
        # TheoremQA: real records are {"score": ..., "details": [ {"example_abbr", "is_correct": [...]}, ... ]};
        # synthetic records are vote JSON whose details is a dict with voted_correctness on each entry.
        id_to_correctness = _id_to_explicit_correctness_from_payload(payload)
    elif dataset_key in {"scicode", "scicode_with_background"}:
        # SciCode real records are raw {subproblem_id: 0/1}; synthetic records
        # are vote JSON with per-item voted_correctness.
        id_to_correctness = _id_to_scicode_correctness(payload)
    elif dataset_key == "countbenchqa":
        # CountBenchQA real records are xlsx-converted lists with question / answer
        # / index / prediction and no judge model; synthetic records are vote JSON
        # with voted_correctness.
        id_to_correctness = _id_to_countbenchqa_correctness(payload)
    elif dataset_key == "simplevqa":
        # SimpleVQA real records are timestamped judged eval JSON lists with
        # judge_res.model_response in Chinese: 正确 / 错误 / 未尝试.
        # Synthetic records carry voted_correctness.
        id_to_correctness = _id_to_simplevqa_correctness(payload)
    elif dataset_key == "scibench":
        # Scibench: explicit correctness field first; otherwise numeric comparison with tolerance (aligned with
        # OpenCompass NumericAccEvaluator), falling back to normalized text comparison when no number can be parsed.
        id_to_correctness = _id_to_correctness_scibench(
            _details_from_results_json(payload), atol=atol, rtol=rtol
        )
    elif dataset_key == "mbpp_plus":
        # mbpp_plus: real records use any_correct; synthetic records carry voted_correctness.
        id_to_correctness = _id_to_correctness_from_details_map_any_correct(
            _details_from_results_json(payload), key="any_correct"
        )
    elif dataset_key == "mathvista_mini":
        # MathVista evaluated xlsx files store extracted answers in `res`;
        # synthetic records are vote JSON with per-item voted_correctness.
        id_to_correctness = _id_to_mathvista_correctness(payload)
    elif dataset_key in DATASET_SPECS:
        spec = DATASET_SPECS[dataset_key]
        choice_re = CHOICE_RE[dataset_key]
        details = _details_from_results_json(payload)
        id_to_correctness = {}
        for qid, rec in details.items():
            if not isinstance(rec, dict):
                rec = {"prediction": rec}
            # If the record already carries an explicit field such as the voted correctness, prefer it:
            # this guarantees voted_correctness==true always maps to 1 and is not affected by
            # mismatched template prediction/reference values.
            explicit = _extract_explicit_correctness(rec)
            if explicit is not None:
                id_to_correctness[str(qid)] = 1 if explicit else 0
                continue
            pred_val = _get_field_value(rec, spec.prediction_keys)
            if dataset_key == "simpleqa":
                pred_letter = _extract_choice_letter(pred_val, choice_re, spec.default_choice)
                # Prediction unparseable (or missing) => counts as wrong by default
                if pred_letter is None:
                    id_to_correctness[str(qid)] = 0
                else:
                    id_to_correctness[str(qid)] = 1 if pred_letter == "A" else 0
            else:
                ref_val = _get_field_value(rec, spec.reference_keys)
                ref_letter = _extract_choice_letter(ref_val, choice_re, None)
                # Reference missing/unparseable => counts as wrong (0 in the aggregated matrix)
                if ref_letter is None:
                    id_to_correctness[str(qid)] = 0
                    continue
                pred_letter = _extract_choice_letter(pred_val, choice_re, None)
                # Prediction missing/unparseable => counts as wrong by default
                if pred_letter is None:
                    id_to_correctness[str(qid)] = 0
                else:
                    id_to_correctness[str(qid)] = 1 if pred_letter == ref_letter else 0
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_samples = int(len(id_to_correctness))
    num_correct = int(sum(id_to_correctness.values()))
    accuracy = float(num_correct) / num_samples if num_samples else 0.0
    metrics = {"num_samples": num_samples, "num_correct": num_correct, "accuracy": accuracy}

    return id_to_correctness, metrics


def _safe_sort_ids(ids: List[str]) -> List[str]:
    try:
        return sorted(ids, key=lambda x: float(x))
    except Exception:
        return sorted(ids, key=lambda x: str(x))


def _dedup_model_name(model_dir: str, root_dir: str, seen: Dict[str, int]) -> str:
    """
    Resolve name clashes between same-named model directories under different roots:
    - no clash: use model_dir as-is
    - clash: append a root identifier and a counter to keep names unique
    """
    if model_dir not in seen:
        seen[model_dir] = 1
        return model_dir
    seen[model_dir] += 1
    tag = os.path.basename(os.path.normpath(root_dir)) or "root"
    return f"{model_dir}@@{tag}@@{seen[model_dir]}"


def _iter_model_json_files(root_dir: str, dataset_name: str) -> List[Tuple[str, str]]:
    """Yield (model_name, json_path) from directory layout and tau2 flat files."""
    entries: List[Tuple[str, str]] = []
    try:
        names = _safe_sort_ids(os.listdir(root_dir))
    except OSError:
        return entries

    for model_dir in names:
        model_path = os.path.join(root_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        json_path = os.path.join(model_path, f"{dataset_name}.json")
        if not os.path.exists(json_path):
            target_lower = f"{_normalize_dataset_name(dataset_name)}.json"
            for file_name in _safe_sort_ids(os.listdir(model_path)):
                if file_name.lower() == target_lower:
                    json_path = os.path.join(model_path, file_name)
                    break
        if not os.path.exists(json_path):
            print(f"Warning: model {model_dir} is missing file: {json_path}")
            continue
        entries.append((model_dir, json_path))

    dataset_key = _normalize_dataset_name(dataset_name)
    if _is_tau2_dataset(dataset_key):
        for file_name in names:
            if not file_name.endswith(PER_INSTANCE_SUFFIX):
                continue
            json_path = os.path.join(root_dir, file_name)
            if os.path.isfile(json_path):
                entries.append((_derive_per_instance_model_name(json_path), json_path))

    return entries


def _ocrbench_v2_record_correctness(record: Dict[str, Any]) -> Optional[int]:
    """Read OCRBench_v2 correctness after binarization or voting.

    OCRBench_v2 raw scores are continuous, so aggregation must not use
    `is_correct` directly unless it comes from a voted/binarized record.
    """
    if not isinstance(record, dict):
        return None
    for key in ("voted_correctness", "binary_is_correct"):
        if key in record:
            flag = _normalize_is_correct(record.get(key))
            if flag is not None:
                return 1 if flag else 0
    return None


def _ocrbench_v2_record_metric(rec: Dict[str, Any]) -> str:
    """Scenario key: `metric`, else `type` / `task` (convert_continuous_native_records.py writes all three)."""
    for key in ("metric", "type", "task"):
        if rec.get(key) not in (None, ""):
            return str(rec[key])
    return "unknown"


def load_ocrbench_v2_metric_json(
    model_json_path: str,
    continuous: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], List[str]]:
    """Load OCRBench_v2 JSON as metric -> id -> correctness.

    Binary mode reads only voted/binarized fields. Continuous mode reads the
    float `is_correct` as-is and skips records flagged `ignored` (their ids
    are returned so the caller can drop the union across models).
    """
    with open(model_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    details = _details_from_results_json(payload)
    metric_maps: Dict[str, Dict[str, float]] = {}
    ignored: List[str] = []
    num_samples = 0
    num_correct = 0.0
    for qid, rec in details.items():
        if not isinstance(rec, dict):
            continue
        if continuous:
            if bool(rec.get("ignored", False)):
                ignored.append(str(qid))
                continue
            if "is_correct" not in rec:
                continue
            try:
                corr: float = float(rec["is_correct"])
            except (TypeError, ValueError):
                continue
        else:
            flag = _ocrbench_v2_record_correctness(rec)
            if flag is None:
                continue
            corr = int(flag)
        metric = _ocrbench_v2_record_metric(rec)
        metric_maps.setdefault(metric, {})[str(qid)] = corr
        num_samples += 1
        num_correct += corr

    metrics = {
        "num_samples": int(num_samples),
        "num_correct": num_correct if continuous else int(num_correct),
        "accuracy": float(num_correct) / num_samples if num_samples else 0.0,
    }
    return metric_maps, metrics, ignored


def aggregate_ocrbench_v2_by_metric(
    root_dirs: List[str],
    dataset_name: str,
    continuous: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Aggregate OCRBench_v2 into one pkl scenario per item `metric` (== `type`).

    With ``continuous=True`` the matrices are float and every id flagged
    `ignored` in any model is dropped from every scenario (union), mirroring
    binarize_continuous_records.py.
    """
    model_list: List[str] = []
    performance_stats: List[Dict[str, Any]] = []
    seen_names: Dict[str, int] = {}

    model_metric_maps: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_ids_by_metric: Dict[str, set[str]] = {}
    ignored_union: set[str] = set()

    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            print(f"Warning: root directory does not exist or is not a directory, skipping: {root_dir}")
            continue

        for model_dir, json_path in _iter_model_json_files(root_dir, dataset_name):
            print(f"Processing file: {json_path}")
            try:
                metric_maps, metrics, ignored = load_ocrbench_v2_metric_json(json_path, continuous=continuous)
            except Exception as e:
                print(f"Error while processing file {json_path}: {str(e)}")
                continue

            unique_name = _dedup_model_name(model_dir, root_dir, seen_names)
            if unique_name != model_dir:
                print(f"Warning: duplicate model directory name, renamed: {model_dir} -> {unique_name}")

            model_list.append(unique_name)
            model_metric_maps[unique_name] = metric_maps
            ignored_union.update(ignored)
            for metric, mapping in metric_maps.items():
                all_ids_by_metric.setdefault(metric, set()).update(mapping.keys())

            performance_stats.append(
                {
                    "model_name": unique_name,
                    "file_path": json_path,
                    "accuracy": metrics["accuracy"],
                    "num_correct": metrics["num_correct"],
                    "num_samples": metrics["num_samples"],
                }
            )

    if not model_list:
        raise ValueError(f"No valid {dataset_name}.json found under root_dir={root_dirs}")
    if ignored_union:
        print(f"Dropping {len(ignored_union)} id(s) flagged `ignored` in at least one model (union)")

    data: Dict[str, Dict[str, np.ndarray]] = {}
    num_models = len(model_list)
    dtype = float if continuous else int
    for metric in _safe_sort_ids(list(all_ids_by_metric.keys())):
        sorted_ids = _safe_sort_ids([qid for qid in all_ids_by_metric[metric] if qid not in ignored_union])
        if not sorted_ids:
            continue
        id_to_idx = {qid: i for i, qid in enumerate(sorted_ids)}
        correctness_array = np.full((len(sorted_ids), num_models), 0, dtype=dtype)
        for m_idx, model_name in enumerate(model_list):
            mapping = model_metric_maps.get(model_name, {}).get(metric, {})
            for qid, corr in mapping.items():
                row_idx = id_to_idx.get(qid)
                if row_idx is not None:
                    correctness_array[row_idx, m_idx] = dtype(corr)
        data[metric] = {"correctness": correctness_array}

    result = {"models": model_list, "data": data}
    return result, performance_stats


def aggregate_results(
    root_dirs: List[str],
    dataset_name: str,
    continuous: bool = False,
    atol: float = SCIBENCH_ATOL_DEFAULT,
    rtol: float = SCIBENCH_RTOL_DEFAULT,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if _normalize_dataset_name(dataset_name) == "ocrbench_v2":
        # per-metric grouped path; --continuous keeps the float scores and
        # drops the `ignored` union (test split of the continuous pipeline)
        return aggregate_ocrbench_v2_by_metric(root_dirs, dataset_name, continuous=continuous)

    model_list: List[str] = []
    performance_stats: List[Dict[str, Any]] = []

    all_ids: set[str] = set()
    model_maps: Dict[str, Dict[str, int]] = {}
    seen_names: Dict[str, int] = {}

    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            print(f"Warning: root directory does not exist or is not a directory, skipping: {root_dir}")
            continue

        for model_dir, json_path in _iter_model_json_files(root_dir, dataset_name):
            print(f"Processing file: {json_path}")
            try:
                if continuous:
                    id_to_corr, metrics = load_model_json_continuous(json_path, dataset_name)
                else:
                    id_to_corr, metrics = load_model_json(json_path, dataset_name, atol=atol, rtol=rtol)
            except Exception as e:
                print(f"Error while processing file {json_path}: {str(e)}")
                continue

            unique_name = _dedup_model_name(model_dir, root_dir, seen_names)
            if unique_name != model_dir:
                print(f"Warning: duplicate model directory name, renamed: {model_dir} -> {unique_name}")

            model_list.append(unique_name)
            model_maps[unique_name] = id_to_corr
            all_ids.update(id_to_corr.keys())

            performance_stats.append(
                {
                    "model_name": unique_name,
                    "file_path": json_path,
                    "accuracy": metrics["accuracy"],
                    "num_correct": metrics["num_correct"],
                    "num_samples": metrics["num_samples"],
                }
            )

    if not model_list:
        raise ValueError(f"No valid {dataset_name}.json found under root_dir={root_dirs}")

    sorted_ids = _safe_sort_ids(list(all_ids))
    num_questions = len(sorted_ids)
    num_models = len(model_list)

    id_to_idx = {qid: i for i, qid in enumerate(sorted_ids)}
    correctness_array = np.full((num_questions, num_models), 0, dtype=(float if continuous else int))

    for m_idx, m_name in enumerate(model_list):
        mapping = model_maps[m_name]
        for qid, corr in mapping.items():
            row_idx = id_to_idx.get(qid)
            if row_idx is None:
                continue
            correctness_array[row_idx, m_idx] = float(corr) if continuous else int(corr)

    result = {
        "models": model_list,
        "data": {dataset_name: {"correctness": correctness_array}},
    }
    return result, performance_stats


def _default_output_path(dataset_name: str) -> str:
    return os.path.join("Zoom", f"aggregated_{dataset_name}.pkl")


def main() -> None:
    args = parse_args()

    # args.root_dir: List[str]
    valid_roots = [d for d in args.root_dir if os.path.isdir(d)]
    if not valid_roots:
        print(f"Error: no valid root directory found: {args.root_dir}")
        return

    dataset_key = _normalize_dataset_name(args.dataset_name)
    if dataset_key not in DATASET_SPECS and dataset_key not in {"humaneval_plus", "mbpp_plus", "scibench", "swe_bench_verified", "swe_bench_multilingual", "swe_bench_pro", "livecodebench", "arc_challenge", "hellaswag", "osworld_verified", "ocrbench_v2", "toolathlon", "terminal_bench", "omnidocbench_text_edit", "omnidocbench_tab_edit", "omnidocbench_tab_teds", "omnidocbench_formula_edit", "omnidocbench_formula_cdm", "omnidocbench_read_order_edit", "arenahard", "mmstar", "mmmu_dev_val", "mmmu_pro_10c", "mmmu_pro_v", "logicvista", "realworldqa", "mmvp", "spatialeval", "theoremqa", "scicode", "scicode_with_background", "countbenchqa", "mathvista_mini", "simplevqa"} and not _is_tau2_dataset(dataset_key):
        print(f"Error: only the following datasets are supported: simpleqa / simplevqa / commonsense_qa / openbookqa / openbookqa_fact / c3 / longbenchv2_0shot / humaneval_plus / mbpp_plus / scibench / swe_bench_verified / swe_bench_multilingual / swe_bench_pro / LiveCodeBench / OSWorld_verified / OCRBench_v2 / Toolathlon / Terminal-Bench / OmniDocBench_text_edit / OmniDocBench_tab_edit / OmniDocBench_formula_edit / OmniDocBench_formula_cdm / OmniDocBench_read_order_edit / ArenaHard / tau2_bench / MMStar / MMMU_DEV_VAL / MMMU_Pro_10c / MMMU_Pro_V / LogicVista / RealWorldQA / MMVP / SpatialEval / theoremqa / SciCode / CountBenchQA / MathVista_MINI, got: {args.dataset_name}")
        return

    result, performance_stats = aggregate_results(
        valid_roots,
        args.dataset_name,
        continuous=args.continuous,
        atol=args.scibench_atol,
        rtol=args.scibench_rtol,
    )

    output_path = args.output_path or _default_output_path(args.dataset_name)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    try:
        with open(output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results saved to: {output_path}")
        print(f"Aggregated {len(result['models'])} model record(s)")
        if args.dataset_name in result["data"]:
            shape = result["data"][args.dataset_name]["correctness"].shape
            print(f"Dataset {args.dataset_name} contains {shape[0]} question(s)")
            print(f"Numpy array shape: {shape}")
        else:
            total_questions = sum(
                scenario["correctness"].shape[0]
                for scenario in result["data"].values()
            )
            print(f"Dataset {args.dataset_name} contains {len(result['data'])} scenario(s)")
            print(f"Total questions: {total_questions}")
            for scenario_name, scenario in sorted(result["data"].items()):
                print(f"  {scenario_name}: {scenario['correctness'].shape}")

        if performance_stats:
            print("\nPer-model performance (sorted by accuracy, descending):")
            for stat in sorted(performance_stats, key=lambda x: x["accuracy"], reverse=True):
                acc_pct = f"{stat['accuracy'] * 100:.2f}%"
                print(
                    f"  {stat['model_name']}: correct {stat['num_correct']}/{stat['num_samples']} (accuracy={acc_pct})"
                )
    except Exception as e:
        print(f"Error while saving file: {str(e)}")


if __name__ == "__main__":
    main()
