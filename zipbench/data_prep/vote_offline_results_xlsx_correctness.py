#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline voting on *correctness* (True/False) over multimodal benchmark XLSX result files.

Same core logic as the text benchmark version (vote_offline_results_json_correctness.py);
the main difference is reading xlsx instead of JSON:
- if the xlsx has a hit column (0/1), use it directly as correctness
- otherwise compare the prediction and answer columns (note: a rough heuristic, may be inaccurate)

Expected directory layout:
root_dir/
  model_folder_A/
    <inner_folder>/
      <model_name>_<benchmark>[_<judge>].xlsx
  model_folder_B/
    ...

Output (JSON, compatible with the batch script):
{
  "is_correct": 0.755,     # fraction correct after voting (0-1)
  "accuracy": 75.5,        # accuracy after voting (percentage)
  "details": {
    "v1_0": { ... "voted_correctness": true },
    ...
  }
}
"""

from __future__ import annotations

import argparse
import csv
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

try:
    import openpyxl
except ImportError:
    print(
        "[ERROR] openpyxl is required. Install with: pip install openpyxl",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Logging ───────────────────────────────────────────────────────────

def eprint(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def _log_model_list(header: str, models: List[ModelResults]) -> None:
    eprint(f"[INFO] {header}")
    for m in models:
        eprint(f"  - {m.model_name}: accuracy={m.accuracy_pct:.4f}")


# ── XLSX Loading ─────────────────────────────────────────────────────

def load_xlsx_records(
    xlsx_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Load an xlsx file and return (lowercase_column_names, {row_id: record_dict}).

    Row IDs come from the ``id`` column, falling back to ``index``.
    """
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return [], {}

    headers = [
        str(h).strip().lower() if h is not None else f"col_{i}"
        for i, h in enumerate(rows[0])
    ]

    records: Dict[str, Dict[str, Any]] = {}
    for row_data in rows[1:]:
        rec: Dict[str, Any] = {}
        for i, val in enumerate(row_data):
            if i < len(headers):
                rec[headers[i]] = val
        # Determine row ID
        row_id = rec.get("id")
        if row_id is None:
            row_id = rec.get("index")
        if row_id is None:
            continue
        records[str(row_id)] = rec

    return headers, records


def load_tsv_records(
    tsv_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Load a TSV file, preserving quoted multiline fields via csv.DictReader."""
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        raw_headers = reader.fieldnames or []
        headers = [
            str(h).strip().lower() if h is not None else f"col_{i}"
            for i, h in enumerate(raw_headers)
        ]

        records: Dict[str, Dict[str, Any]] = {}
        for fallback_idx, row in enumerate(reader):
            rec: Dict[str, Any] = {}
            for i, raw_header in enumerate(raw_headers):
                if raw_header is None:
                    continue
                rec[headers[i]] = row.get(raw_header)
            row_id = rec.get("id")
            if row_id is None:
                row_id = rec.get("index")
            if row_id is None:
                row_id = fallback_idx
            records[str(row_id)] = rec

    return headers, records


def load_eval_json_records(
    json_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Load judged eval JSON records, including SimpleVQA timestamp outputs."""
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("details"), list):
        items = payload["details"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("details"), dict):
        items = []
        for key, value in payload["details"].items():
            if isinstance(value, dict):
                rec = dict(value)
                rec.setdefault("id", key)
                items.append(rec)
    elif isinstance(payload, dict):
        items = []
        for key, value in payload.items():
            if isinstance(value, dict):
                rec = dict(value)
                rec.setdefault("id", key)
                items.append(rec)
    else:
        items = []

    records: Dict[str, Dict[str, Any]] = {}
    headers: set[str] = set()
    for fallback_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        rec = dict(item)
        row_id = rec.get("id")
        if row_id is None:
            row_id = rec.get("index")
        if row_id is None:
            row_id = rec.get("question_id")
        if row_id is None:
            row_id = fallback_idx
        headers.update(str(k).strip().lower() for k in rec)
        if isinstance(rec.get("judge_res"), dict):
            headers.add("judge_res.model_response")
        records[str(row_id)] = rec

    return sorted(headers), records


def load_result_records(
    result_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    suffix = result_path.suffix.lower()
    if suffix == ".json":
        return load_eval_json_records(result_path)
    if suffix == ".tsv":
        return load_tsv_records(result_path)
    return load_xlsx_records(result_path)


# ── XLSX File Discovery ──────────────────────────────────────────────

def _model_token_from_result_name(path: Path, dataset: str) -> str:
    marker = f"_{dataset}".lower()
    name = path.name
    idx = name.lower().find(marker)
    return name[:idx].lower() if idx >= 0 else path.stem.lower()


def _split_judges(judge: Optional[str]) -> List[str]:
    if not judge:
        return []
    return [p.strip() for p in str(judge).split(",") if p.strip()]


def _match_sort_key(model_dir: Path, dataset: str, path: Path) -> Tuple[int, int, int, str]:
    """Prefer candidates whose file/parent name semantically matches model_dir."""
    outer = model_dir.name.lower()
    parent = path.parent.name.lower()
    token = _model_token_from_result_name(path, dataset)
    semantic_match = bool(
        token and (
            token in outer or outer in token or parent in outer or outer in parent
        )
    )
    parent_matches_file = bool(token and parent == token)
    return (
        0 if semantic_match else 1,
        0 if parent_matches_file else 1,
        len(path.relative_to(model_dir).parts),
        str(path).lower(),
    )

def _find_xlsx_in_model_dir(
    model_dir: Path, dataset: str, judge: Optional[str],
) -> Optional[Path]:
    """
    Find a matching xlsx file within *model_dir* (searches recursively).

    Matching rules (case-insensitive on the filename):
    - With judge:    ``*_<dataset>_<judge>.xlsx`` or
                     ``*_<dataset>_<judge>_result.xlsx``
    - With judge=any/auto/*: ``*_<dataset>_*_result.xlsx``
    - Without judge: ``*_<dataset>.xlsx``  (will NOT match ``*_<dataset>_<judge>.xlsx``)
    """
    judge_values = _split_judges(judge)
    judge_keys = {j.lower() for j in judge_values}
    suffix_priority = None

    if judge_keys & {"any", "auto", "*"}:
        dataset_token = f"_{dataset}".lower()

        def is_match(path: Path) -> bool:
            name = path.name.lower()
            return f"{dataset_token}_" in name and name.endswith("_result.xlsx")
    elif judge_values:
        target_suffixes = []
        for one_judge in judge_values:
            target_suffixes.extend([
                f"_{dataset}_{one_judge}.xlsx",
                f"_{dataset}_{one_judge}_result.xlsx",
            ])

        target_lowers = tuple(s.lower() for s in target_suffixes)

        def is_match(path: Path) -> bool:
            return path.name.lower().endswith(target_lowers)

        def suffix_priority(path: Path) -> int:
            name = path.name.lower()
            for idx, suffix in enumerate(target_lowers):
                if name.endswith(suffix):
                    return idx
            return len(target_lowers)
    else:
        target_suffixes = (f"_{dataset}.xlsx",)

        target_lowers = tuple(s.lower() for s in target_suffixes)

        def is_match(path: Path) -> bool:
            return path.name.lower().endswith(target_lowers)

    matches = [
        xlsx_path
        for xlsx_path in model_dir.rglob("*.xlsx")
        if is_match(xlsx_path)
    ]
    if matches:
        return sorted(
            matches,
            key=lambda p: (
                *_match_sort_key(model_dir, dataset, p)[:3],
                suffix_priority(p) if suffix_priority else 0,
                _match_sort_key(model_dir, dataset, p)[3],
            ),
        )[0]

    if str(dataset).strip().lower() == "simplevqa":
        json_match = _find_simplevqa_eval_json_in_model_dir(model_dir, judge)
        if json_match is not None:
            return json_match

    tsv_matches = [p for p in model_dir.rglob("detail.tsv") if p.is_file()]
    if tsv_matches:
        return sorted(
            tsv_matches,
            key=lambda p: (
                len(p.relative_to(model_dir).parts),
                str(p).lower(),
            ),
        )[0]

    return None


def _find_simplevqa_eval_json_in_model_dir(model_dir: Path, judge: Optional[str]) -> Optional[Path]:
    """Find SimpleVQA judged eval JSON inside timestamped model folders."""
    judge_values = _split_judges(judge)
    judge_keys = {j.lower() for j in judge_values}

    def is_match(path: Path) -> bool:
        name = path.name.lower()
        if name == "model_eval.json":
            return True
        if judge_keys & {"any", "auto", "*"}:
            return name.endswith("_eval.json")
        if judge_values:
            return any(name == f"{j.lower()}_eval.json" for j in judge_values)
        return name.endswith("_eval.json")

    matches = [p for p in model_dir.rglob("*.json") if is_match(p)]
    if not matches:
        return None

    def sort_key(path: Path) -> Tuple[int, str, int, str]:
        timestamp = path.parent.name if path.parent.name.startswith("T") else ""
        return (
            0 if timestamp else 1,
            "".join(chr(255 - ord(ch)) for ch in timestamp),
            len(path.relative_to(model_dir).parts),
            str(path).lower(),
        )

    return sorted(
        matches,
        key=sort_key,
    )[0]


# ── Correctness Computation ──────────────────────────────────────────

def _compute_correctness_from_hit(record: Dict[str, Any]) -> Optional[bool]:
    """Extract correctness from the ``hit`` column (0/1)."""
    hit_val = record.get("hit")
    if hit_val is None:
        return None
    try:
        return bool(int(float(hit_val)))
    except (TypeError, ValueError):
        pass
    if isinstance(hit_val, str):
        s = hit_val.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def _compute_correctness_from_if_right(record: Dict[str, Any]) -> Optional[bool]:
    """Extract correctness from VLMEvalKit-style ``if_right`` columns."""
    if_right = record.get("if_right")
    if if_right is None:
        return None
    if isinstance(if_right, bool):
        return if_right
    if isinstance(if_right, (int, float)):
        return bool(if_right)
    if isinstance(if_right, str):
        s = if_right.strip().lower()
        if s in {"true", "1", "yes", "y", "correct"}:
            return True
        if s in {"false", "0", "no", "n", "incorrect"}:
            return False
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


def _compute_simplevqa_correctness(record: Dict[str, Any]) -> Optional[bool]:
    if "voted_correctness" in record:
        parsed = _normalize_simplevqa_judge(record.get("voted_correctness"))
        if parsed is not None:
            return parsed
    if "judge_res" in record:
        parsed = _normalize_simplevqa_judge(record.get("judge_res"))
        if parsed is not None:
            return parsed
    return _normalize_simplevqa_judge(record.get("judge_res.model_response"))


def _compare_prediction_answer(prediction: Any, answer: Any) -> Optional[bool]:
    """
    Compare *prediction* vs *answer* when the ``hit`` column is unavailable.

    This is a rough heuristic:
    1. Exact match (case-insensitive, whitespace-normalised).
    2. For short answers (<=10 chars): word-boundary search in the last 300
       characters of the prediction (final answer usually appears at the end).
    3. For longer answers: substring containment check.
    """
    if prediction is None or answer is None:
        return None
    pred_str = str(prediction).strip()
    ans_str = str(answer).strip()
    if not pred_str or not ans_str:
        return None

    pred_lower = pred_str.lower()
    ans_lower = ans_str.lower()

    # 1) Exact match
    if pred_lower == ans_lower:
        return True

    # 2) Normalised exact match
    pred_norm = re.sub(r"\s+", " ", pred_lower).strip()
    ans_norm = re.sub(r"\s+", " ", ans_lower).strip()
    if pred_norm == ans_norm:
        return True

    # 3) Short answer: look near the end of prediction
    if len(ans_str) <= 10:
        try:
            pattern = re.compile(
                r"(?:^|\b)" + re.escape(ans_lower) + r"(?:\b|$)",
                re.IGNORECASE,
            )
        except re.error:
            return False
        tail = pred_str[-300:] if len(pred_str) > 300 else pred_str
        if pattern.search(tail):
            return True
    elif ans_lower in pred_lower:
        # 4) Longer answer: containment
        return True

    return False


def _compare_countbenchqa_prediction_answer(
    prediction: Any, answer: Any,
) -> Optional[bool]:
    """CountBenchQA official evaluation: answer string is a substring of prediction."""
    if prediction is None or answer is None:
        return None
    pred_str = str(prediction)
    ans_str = str(answer)
    if not pred_str or not ans_str:
        return None
    return ans_str in pred_str


_CHOICE_LETTER_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def _extract_choice_letter(value: Any) -> Optional[str]:
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


def _compute_mathvista_correctness(record: Dict[str, Any]) -> Optional[bool]:
    """MathVista evaluated XLSX files store extracted answers in ``res``."""
    response = record.get("res")
    if response is None:
        response = record.get("prediction")

    question_type = str(record.get("question_type") or "").strip().lower()
    answer_type = str(record.get("answer_type") or "").strip().lower()

    if question_type == "multi_choice":
        pred_letter = _extract_choice_letter(response)
        ans_letter = _extract_choice_letter(record.get("answer_option"))
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


def compute_correctness_for_record(
    record: Dict[str, Any], has_hit_col: bool, dataset: Optional[str] = None,
) -> Optional[bool]:
    """Judge a single record's correctness."""
    if str(dataset or "").strip().lower() == "simplevqa":
        return _compute_simplevqa_correctness(record)
    if "if_right" in record:
        explicit = _compute_correctness_from_if_right(record)
        if explicit is not None:
            return explicit
    if has_hit_col:
        return _compute_correctness_from_hit(record)
    if str(dataset or "").strip().lower() == "mathvista_mini":
        return _compute_mathvista_correctness(record)
    if str(dataset or "").strip().lower() == "countbenchqa":
        return _compare_countbenchqa_prediction_answer(
            record.get("prediction"), record.get("answer"),
        )
    return _compare_prediction_answer(
        record.get("prediction"), record.get("answer"),
    )


# ── Model Data ────────────────────────────────────────────────────────

@dataclass
class ModelResults:
    model_name: str
    xlsx_path: Path
    details: Dict[str, Dict[str, Any]]  # id -> record
    correctness: Dict[str, bool]  # id -> True/False
    score: float  # accuracy as fraction [0, 1]
    accuracy_pct: float  # accuracy as percentage [0, 100]


def list_model_dirs(root: Path) -> List[Path]:
    """List immediate subdirectories of *root*, sorted by name."""
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda x: x.name,
    )


def _filter_model_dirs(
    all_dirs: List[Path], include_models: str, exclude_models: str,
) -> List[Path]:
    selected = list(all_dirs)
    if include_models:
        inc = {m.strip() for m in include_models.split(",") if m.strip()}
        selected = [d for d in selected if d.name in inc]
    if exclude_models:
        exc = {m.strip() for m in exclude_models.split(",") if m.strip()}
        selected = [d for d in selected if d.name not in exc]
    return selected


def build_models_info(
    root: Path,
    dataset: str,
    judge: Optional[str],
    include_models: str,
    exclude_models: str,
) -> List[ModelResults]:
    """Load all model results from *root* directory."""
    model_dirs = _filter_model_dirs(
        list_model_dirs(root), include_models, exclude_models,
    )
    if not model_dirs:
        eprint("[ERROR] No model folders found after include/exclude filtering.")
        sys.exit(2)

    models: List[ModelResults] = []
    missing = bad = 0

    for mdir in model_dirs:
        result_path = _find_xlsx_in_model_dir(mdir, dataset, judge)
        if result_path is None:
            eprint(
                f"[WARN] No matching result file for model '{mdir.name}' "
                f"(dataset={dataset}, judge={judge})"
            )
            missing += 1
            continue

        try:
            headers, details = load_result_records(result_path)
            has_hit = "hit" in headers
            has_if_right = "if_right" in headers

            if not has_hit and not has_if_right:
                dataset_key = str(dataset).strip().lower()
                if dataset_key == "simplevqa":
                    eprint(
                        f"[INFO] No 'hit' column in {result_path.name}; "
                        "using SimpleVQA judge_res.model_response correctness."
                    )
                elif dataset_key == "mathvista_mini":
                    eprint(
                        f"[INFO] No 'hit' column in {result_path.name}; "
                        "using MathVista res/answer_option correctness."
                    )
                else:
                    eprint(
                        f"[WARN] No 'hit'/'if_right' column in {result_path.name}; "
                        "using prediction vs answer comparison (may be inaccurate)."
                    )

            correctness: Dict[str, bool] = {}
            for qid, rec in details.items():
                result = compute_correctness_for_record(rec, has_hit, dataset)
                if result is not None:
                    correctness[qid] = result

            total = len(correctness)
            correct = sum(1 for v in correctness.values() if v)
            acc_frac = correct / total if total else 0.0

            models.append(
                ModelResults(
                    model_name=mdir.name,
                    xlsx_path=result_path,
                    details=details,
                    correctness=correctness,
                    score=acc_frac,
                    accuracy_pct=acc_frac * 100.0,
                )
            )
        except Exception as e:
            eprint(
                f"[WARN] Skipping bad result file for model '{mdir.name}': "
                f"{result_path} ({e})"
            )
            bad += 1

    if not models:
        eprint("[ERROR] No valid model result files found.")
        sys.exit(2)

    eprint(
        f"[INFO] Loaded {len(models)} model result files. "
        f"(missing={missing}, bad={bad})"
    )
    return models


# ── Model Selection ───────────────────────────────────────────────────

def filter_models_by_performance(
    models: List[ModelResults], min_acc: float, max_acc: float,
) -> List[ModelResults]:
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
    if len(models) < k:
        eprint(f"[ERROR] Not enough models: have {len(models)}, need {k}")
        sys.exit(2)

    if weak_boost == 1.0:
        selected = rng.sample(models, k)
    else:
        weights = [
            max(0.0, min(1.0, 1.0 - m.score)) ** weak_boost for m in models
        ]
        selected = _weighted_sample_no_replace(models, weights, k, rng)

    _log_model_list("Selected models:", selected)
    return selected


# ── Voting ────────────────────────────────────────────────────────────

def _vote_majority(
    votes: List[bool], rng: random.Random,
) -> Tuple[bool, bool]:
    """Majority vote on True/False. Returns (chosen_value, is_tie)."""
    if not votes:
        return False, False
    counts = Counter(votes)
    max_ct = max(counts.values())
    top_vals = [val for val, ct in counts.items() if ct == max_ct]
    tie = len(top_vals) > 1
    return (rng.choice(top_vals) if tie else top_vals[0]), tie


def _qid_sort_key(qid: str) -> Tuple[int, Any]:
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def _build_ordered_ids(selected: List[ModelResults]) -> List[str]:
    """Deterministic ordering of all question IDs across selected models."""
    union_ids = {qid for m in selected for qid in m.details}

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
    """Majority vote on correctness, output compatible with batch script."""
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
    eprint(
        f"[INFO] Ensemble metrics: accuracy: {acc_pct:.6f} "
        f"(fraction={acc_frac:.6f})"
    )

    return {
        "is_correct": acc_frac,
        "accuracy": acc_pct,
        "details": details_out,
    }


# ── Argument Parsing ──────────────────────────────────────────────────

def _parse_selected_identifiers(raw_args: Optional[List[str]]) -> List[str]:
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
        description=(
            "Offline voting on correctness over multimodal benchmark "
            "XLSX result files."
        ),
    )
    ap.add_argument(
        "--root", required=True,
        help="Root directory containing per-model folders.",
    )
    ap.add_argument(
        "--dataset", required=True,
        help="Dataset/benchmark name (e.g., LogicVista).",
    )
    ap.add_argument(
        "--judge", default=None,
        help=(
            "Judge model name suffix (e.g., qwen3-30b). "
            "Comma-separated values are accepted (e.g., qwen3-30b,deepseek-v3). "
            "Use any/auto/* to accept any judged *_result.xlsx. "
            "If omitted, uses non-judged xlsx files."
        ),
    )
    ap.add_argument(
        "--k", type=int, default=None,
        help="Number of models to select for voting.",
    )
    ap.add_argument("--out", default=None, help="Output results JSON path.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument(
        "--include-models", default="",
        help="Comma-separated model folder names to include.",
    )
    ap.add_argument(
        "--exclude-models", default="",
        help="Comma-separated model folder names to exclude.",
    )
    ap.add_argument(
        "--min-acc", type=float, default=0.0,
        help="Min accuracy threshold (0-1).",
    )
    ap.add_argument(
        "--max-acc", type=float, default=1.0,
        help="Max accuracy threshold (0-1).",
    )
    ap.add_argument(
        "--weak-boost", type=float, default=1.0,
        help=(
            "Weight exponent on (1-acc). "
            ">1 boosts weaker, =1 uniform, <1 boosts stronger."
        ),
    )
    ap.add_argument(
        "--print-candidates", action="store_true",
        help="Print filtered candidate model names to stdout and exit.",
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
    _log_model_list(
        f"Using fixed selection of {len(selected)} models:", selected,
    )
    return selected


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)

    models = build_models_info(
        root, args.dataset, args.judge,
        args.include_models, args.exclude_models,
    )

    _log_model_list(
        "Model performance summary:",
        sorted(models, key=lambda x: -x.score),
    )

    filtered = filter_models_by_performance(models, args.min_acc, args.max_acc)

    # --print-candidates mode
    if args.print_candidates:
        for m in filtered:
            print(m.model_name)
        return

    if not args.k or args.k <= 0:
        eprint(
            "[ERROR] --k is required and must be positive "
            "(unless --print-candidates)."
        )
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
