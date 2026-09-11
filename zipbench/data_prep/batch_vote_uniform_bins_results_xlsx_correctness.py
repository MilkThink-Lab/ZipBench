#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch voting driver that uniformly covers an accuracy range with bins (XLSX results version, based on exhaustive combinations).

Version for the xlsx result files of multimodal benchmarks,
derived from the text version batch_vote_uniform_bins_results_json_multiqa.py.

Core mechanism, same as the text version:
1) Split [acc_min, acc_max] into m equal-width bins;
2) The candidate pool comes from the filtered output of vote_offline_results_xlsx_correctness.py
   (obtained as a list of model_name via --print-candidates);
3) Enumerate all combinations for each k (order controlled by --k-order, default 2..N, i.e. start from the smallest ensembles), calling the vote script once per combination to produce an ensemble JSON;
4) The top-level is_correct (0-1) of the ensemble JSON decides the bin;
5) After all k are traversed, the accepted records are emitted.

Main differences:
- Candidate model result files are in xlsx format
- --judge optionally names the judge model, used to pick the right xlsx file
- Correctness: use the hit column when present, otherwise compare prediction with answer
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import openpyxl
except ImportError:
    print(
        "[ERROR] openpyxl is required. Install with: pip install openpyxl",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Constants & Configuration ─────────────────────────────────────────

DEFAULT_VOTE_SCRIPT = Path(__file__).with_name(
    "vote_offline_results_xlsx_correctness.py"
)

FORBIDDEN_FORWARD_ARGS = frozenset({
    "--out", "--seed", "--root", "--dataset", "--judge",
    "--k", "--print-candidates", "--selected-identifiers",
})


@dataclass(frozen=True)
class MetricConfig:
    """Per-dataset metric configuration."""
    kind: str   # "accuracy" (0-100) or "is_correct" (0-1)
    fmt: str    # format spec for output filenames


_ACC_CFG = MetricConfig("accuracy", ".2f")
_IC_CFG = MetricConfig("is_correct", ".6f")

# By default all multimodal benchmarks use is_correct (fraction [0, 1]).
# Add dataset-specific overrides here if needed.
METRIC_BY_DATASET: Dict[str, MetricConfig] = {}


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class CandidateRecord:
    json_path: Path
    accuracy: float   # metric value
    source: str       # e.g. generated_k3


# ── Low-level Helpers ─────────────────────────────────────────────────

def _get_metric_config(dataset_key: str) -> MetricConfig:
    return METRIC_BY_DATASET.get(dataset_key, _IC_CFG)


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
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
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


# ── Bin Management ────────────────────────────────────────────────────

def build_bins(
    m: int, acc_min: float, acc_max: float,
) -> List[Tuple[float, float]]:
    if m <= 0:
        raise ValueError("--m must be positive")
    if acc_min == acc_max:
        return [(acc_min, acc_max)] * m
    delta = (acc_max - acc_min) / m
    return [
        (
            acc_min + i * delta,
            acc_max if i == m - 1 else acc_min + (i + 1) * delta,
        )
        for i in range(m)
    ]


def find_bin_index(
    acc: float, bins: Sequence[Tuple[float, float]],
) -> Optional[int]:
    for i, (low, high) in enumerate(bins):
        if (low <= acc < high) if i < len(bins) - 1 else (low <= acc <= high):
            return i
    return None


def select_best_from_bucket(
    bucket: Sequence[CandidateRecord], low: float, high: float,
) -> Optional[CandidateRecord]:
    """Select the record closest to the bin centre."""
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
    """Find matching xlsx within *model_dir* (same logic as vote script)."""
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


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


# ── Metric Reading from Output JSON ──────────────────────────────────

def read_metric_from_results_json(
    path: Path, dataset_key: str,
) -> Optional[float]:
    """Read the primary metric from the ensemble output JSON."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    cfg = _get_metric_config(dataset_key)

    if cfg.kind == "accuracy":
        try:
            return float(data["accuracy"])
        except (KeyError, TypeError, ValueError):
            pass
    else:
        if "is_correct" in data:
            try:
                return float(data["is_correct"])
            except (TypeError, ValueError):
                pass
    return None


# ── Accuracy Inference from XLSX ──────────────────────────────────────

def _is_countbenchqa_correct(prediction: Any, answer: Any) -> Optional[bool]:
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


def _is_mathvista_correct(record: Dict[str, Any]) -> Optional[bool]:
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


def _load_tabular_rows(result_path: Path) -> Optional[List[Tuple[Any, ...]]]:
    if result_path.suffix.lower() == ".tsv":
        try:
            with result_path.open("r", encoding="utf-8", newline="") as f:
                return [tuple(row) for row in csv.reader(f, delimiter="\t")]
        except Exception:
            return None

    try:
        wb = openpyxl.load_workbook(
            str(result_path), read_only=True, data_only=True,
        )
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
    except Exception:
        return None
    return rows


def _compute_accuracy_from_xlsx(xlsx_path: Path, dataset: str) -> Optional[float]:
    """Compute accuracy fraction [0, 1] from a single xlsx/detail.tsv file."""
    if xlsx_path.suffix.lower() == ".json":
        if str(dataset).strip().lower() != "simplevqa":
            return None
        try:
            with xlsx_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("details"), list):
            items = payload["details"]
        elif isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict) and isinstance(payload.get("details"), dict):
            items = list(payload["details"].values())
        else:
            items = []
        total = correct = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            parsed = _normalize_simplevqa_judge(item.get("judge_res"))
            if parsed is None:
                parsed = _normalize_simplevqa_judge(item.get("judge_res.model_response"))
            if parsed is None:
                continue
            total += 1
            if parsed:
                correct += 1
        return correct / total if total else 0.0

    rows = _load_tabular_rows(xlsx_path)

    if not rows:
        return None

    headers = [
        str(h).strip().lower() if h is not None else ""
        for h in rows[0]
    ]
    dataset_key = str(dataset).strip().lower()
    has_hit = "hit" in headers
    has_if_right = "if_right" in headers

    if dataset_key == "mathvista_mini":
        total = correct = 0
        for row in rows[1:]:
            rec = {
                headers[i]: row[i]
                for i in range(min(len(headers), len(row)))
            }
            is_correct = _is_mathvista_correct(rec)
            if is_correct is not None:
                total += 1
                if is_correct:
                    correct += 1
        return correct / total if total else 0.0

    if has_if_right or has_hit:
        hit_idx = headers.index("if_right") if has_if_right else headers.index("hit")
        total = correct = 0
        for row in rows[1:]:
            if hit_idx < len(row):
                val = row[hit_idx]
                if val is not None:
                    parsed = _normalize_is_correct(val)
                    if parsed is None:
                        continue
                    total += 1
                    if parsed:
                        correct += 1
        return correct / total if total else 0.0
    else:
        # Fallback: compare prediction and answer (exact match only)
        pred_idx = (
            headers.index("prediction") if "prediction" in headers else -1
        )
        ans_idx = headers.index("answer") if "answer" in headers else -1
        if pred_idx < 0 or ans_idx < 0:
            return None
        total = correct = 0
        for row in rows[1:]:
            if pred_idx < len(row) and ans_idx < len(row):
                pred = row[pred_idx]
                ans = row[ans_idx]
                if pred is not None and ans is not None:
                    total += 1
                    if str(dataset).strip().lower() == "countbenchqa":
                        is_correct = _is_countbenchqa_correct(pred, ans)
                    else:
                        is_correct = str(pred).strip().lower() == str(ans).strip().lower()
                    if is_correct:
                        correct += 1
        return correct / total if total else 0.0


def infer_acc_range(
    root_dir: Path, dataset: str, judge: Optional[str],
) -> Tuple[List[Tuple[str, float]], float, float]:
    """Infer (per_model_metrics, acc_min, acc_max) from xlsx files."""
    per_model: List[Tuple[str, float]] = []
    for mdir in list_model_dirs(root_dir):
        xlsx_path = _find_xlsx_in_model_dir(mdir, dataset, judge)
        if xlsx_path is None:
            continue
        acc = _compute_accuracy_from_xlsx(xlsx_path, dataset)
        if acc is not None:
            per_model.append((mdir.name, float(acc)))

    if not per_model:
        raise ValueError(
            "Cannot infer accuracy range (no readable xlsx files). "
            "Please provide --min-final-acc and --max-final-acc explicitly."
        )
    values = [a for _, a in per_model]
    return per_model, min(values), max(values)


# ── Subprocess Wrappers ───────────────────────────────────────────────

def _build_vote_cmd(
    vote_script: Path,
    root_dir: Path,
    dataset: str,
    judge: Optional[str],
    extra_args: Sequence[str],
) -> List[str]:
    cmd = [
        sys.executable, str(vote_script),
        "--root", str(root_dir),
        "--dataset", str(dataset),
    ]
    if judge:
        cmd.extend(["--judge", str(judge)])
    cmd.extend(extra_args)
    return cmd


def run_vote_once(
    vote_script: Path,
    root_dir: Path,
    dataset: str,
    judge: Optional[str],
    vote_args: Sequence[str],
    out_path: Path,
    seed: int,
) -> subprocess.CompletedProcess:
    cmd = _build_vote_cmd(vote_script, root_dir, dataset, judge, [
        "--out", str(out_path), "--seed", str(seed), *vote_args,
    ])
    return subprocess.run(cmd)


def run_vote_print_candidates(
    vote_script: Path,
    root_dir: Path,
    dataset: str,
    judge: Optional[str],
    vote_args: Sequence[str],
) -> List[str]:
    cmd = _build_vote_cmd(vote_script, root_dir, dataset, judge, [
        "--print-candidates", *vote_args,
    ])
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"vote script --print-candidates failed "
            f"(rc={completed.returncode}). stderr:\n{completed.stderr}"
        )
    return [ln for ln in (completed.stdout or "").splitlines() if ln.strip()]


# ── Argument Parsing ──────────────────────────────────────────────────

def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "XLSX results version: enumerate k-combinations in --k-order (default 2..N) to build ensembles, "
            "and uniformly cover the accuracy range with bins."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--m", type=int, required=True,
        help="Number of bins (at most m records are selected in the end).",
    )
    parser.add_argument(
        "--root", required=True,
        help="Root directory containing one sub-folder per model.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset/benchmark name (e.g. LogicVista).",
    )
    parser.add_argument(
        "--judge", default=None,
        help="Judge model name suffix (e.g. qwen3-30b); comma-separated list allowed, e.g. qwen3-30b,deepseek-v3; any/auto/* accepts any *_result.xlsx.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory (generated/, selected_list, etc.).",
    )
    parser.add_argument(
        "--min-final-acc", type=float, default=None,
        help="Lower bound (inclusive) of the accuracy range; inferred from the candidate pool if omitted.",
    )
    parser.add_argument(
        "--max-final-acc", type=float, default=None,
        help="Upper bound (inclusive) of the accuracy range; inferred from the candidate pool if omitted.",
    )
    parser.add_argument(
        "--vote-script", default=str(DEFAULT_VOTE_SCRIPT),
        help="Path to vote_offline_results_xlsx_correctness.py.",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=0,
        help="Global maximum number of attempts (0 = unlimited).",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="Initial seed.",
    )
    parser.add_argument(
        "--seed-step", type=int, default=1,
        help="Seed increment after each call.",
    )
    parser.add_argument(
        "--strict-bins", action="store_true",
        help="Strict mode: exit with non-zero status if any bin is left empty.",
    )
    parser.add_argument(
        "--k-order",
        choices=("asc", "desc"),
        default="asc",
        help="Traversal order of k: asc=2..N (default; start from small ensembles and grow k only when bins stay empty); desc=N..2 (prefer the majority consensus of large ensembles).",
    )

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
            "Do not set "
            f"{', '.join(sorted(conflicts))} in the pass-through arguments; this script manages them automatically."
        )
    return forwarded


# ── Combination Attempt ───────────────────────────────────────────────

def _try_combination(
    *,
    vote_script: Path,
    root_dir: Path,
    dataset: str,
    judge: Optional[str],
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
    temp_out = (
        generated_root / f"temp_{attempt_index:06d}_k_{k}_seed_{seed}.json"
    )
    try:
        vote_args = vote_forwarded + [
            "--k", str(k),
            "--selected-identifiers", ",".join(combo),
        ]
        completed = run_vote_once(
            vote_script, root_dir, dataset, judge,
            vote_args, temp_out, seed,
        )
        if completed.returncode != 0:
            print(
                f"[WARN] Vote script call #{attempt_index} failed: "
                f"k={k}, seed={seed}, rc={completed.returncode}",
                file=sys.stderr,
            )
            return None

        acc = read_metric_from_results_json(temp_out, dataset_key)
        if acc is None:
            print(
                f"[WARN] Could not read "
                f"{metric_cfg.kind} from JSON generated at attempt #{attempt_index}: k={k}, seed={seed}; discarding.",
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
        final_path = (
            generated_root
            / f"acc_{acc:{metric_cfg.fmt}}_k_{k}_seed_{seed}.json"
        )
        temp_out.rename(final_path)
        return bin_idx, CandidateRecord(
            json_path=final_path,
            accuracy=acc,
            source=f"generated_k{k}",
        )
    finally:
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
            print(
                "[ERROR] --strict-bins is enabled; exiting because some bins are empty.",
                file=sys.stderr,
            )
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
            fh_txt.write(
                f"{i}\t{low:.6f}\t{high:.6f}\t"
                f"{rec.accuracy:.6f}\t{rel_path}\n"
            )
            fh_json.write(
                json.dumps(
                    {
                        "bin_index": i,
                        "bin_low": low,
                        "bin_high": high,
                        "accuracy": rec.accuracy,
                        "json_path": str(rec.json_path),
                        "source": rec.source,
                        "metric": metric_name,
                        "dataset": dataset,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

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
        print(
            f"[ERROR] vote script not found: {vote_script}", file=sys.stderr,
        )
        return 2

    root_dir = Path(args.root).resolve()
    if not root_dir.is_dir():
        print(
            f"[ERROR] --root not found or not a directory: {root_dir}",
            file=sys.stderr,
        )
        return 2

    dataset = str(args.dataset)
    dataset_key = dataset.strip().lower()
    judge = args.judge
    metric_cfg = _get_metric_config(dataset_key)

    # ── Infer accuracy range ──
    per_model_acc: List[Tuple[str, float]] = []
    if args.min_final_acc is None or args.max_final_acc is None:
        try:
            per_model_acc, acc_min_auto, acc_max_auto = infer_acc_range(
                root_dir, dataset, judge,
            )
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 2
        acc_min = (
            acc_min_auto
            if args.min_final_acc is None
            else float(args.min_final_acc)
        )
        acc_max = (
            acc_max_auto
            if args.max_final_acc is None
            else float(args.max_final_acc)
        )
    else:
        acc_min = float(args.min_final_acc)
        acc_max = float(args.max_final_acc)

    if acc_min > acc_max:
        print(
            "[ERROR] --min-final-acc must not be greater than --max-final-acc",
            file=sys.stderr,
        )
        return 2

    print(
        f"[INFO] Global accuracy range ({metric_cfg.kind}): "
        f"[{acc_min:.6f}, {acc_max:.6f}]",
        file=sys.stderr,
    )

    bins = build_bins(args.m, acc_min, acc_max)

    # ── Print per-model bin statistics ──
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
            vote_script, root_dir, dataset, judge, vote_forwarded,
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
        f"[INFO] Candidate models available for voting: N={n_candidates}; "
        f"enumerating combinations with k={'N..2' if args.k_order == 'desc' else '2..N'} ({args.k_order}).",
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
                judge=judge,
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
            continue
        break

    return _write_results(
        output_dir, bins, buckets, dataset, metric_cfg.kind, args.strict_bins,
    )


if __name__ == "__main__":
    sys.exit(main())
