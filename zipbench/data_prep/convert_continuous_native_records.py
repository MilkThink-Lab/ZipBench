#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert native continuous-metric results into the flat ZipBench record schema.

One model at a time: read the harness-native result of ``--source`` and write
``<output-root>/<model>/<dataset>.json`` in the unified continuous schema::

    {
      "dataset": "<dataset>", "model": "<model>", "source": "<source>",
      "is_correct": <mean of per-item is_correct>,     # continuous accuracy
      "accuracy":   <same value>,
      "details": {
        "<qid>": {
          "is_correct": <float in [0, 1], higher is better>,
          "task":       "<grouping key used by binarize_continuous_records.py>",
          "missing":    <bool, item absent from the native result -> worst score>,
          "raw":        {...native values, kept for auditing...},
          # ocrbench_v2 only: "type", "metric" (== type) and "ignored"
        }, ...
      },
      "meta": {"source", "direction", "n_items", "n_missing", ...}
    }

This script only unifies the format: direction flip (edit distances become
``1 - edit``), worst score for missing items, and the OCRBench_v2 ``ignored``
flag. No thresholding happens here -- that is ``binarize_continuous_records.py``
(train records) or ``aggregate_results_json_multiqa.py --continuous`` (test).

Sources, accepted inputs (harness-native, or the per-model records hosted in
the ``zooer/ZipBench`` dataset) and the ``task`` they emit:

* ``arenahard``        opencompass judged ``arenahard.json`` (``{idx: {gold,
                       prediction}}``, ``[[A>>B]]`` verdicts) or the question list
                       ``[{question_id, capability, pairs: [{answer1, answer2,
                       judge_label}]}]``; 2 games / question folded to a soft win
                       rate; task = ``capability`` (kept, not used for grouping).
* ``osworld_verified`` ``all_result.json`` / ``<domain>/<uuid>/result.txt`` tree
                       (agent adapter) or a record ``{details: {uuid:
                       {application, is_correct}}}``; task = domain.
* ``terminal_bench``   harbor job/trial ``result.json`` (agent adapter, per-trial
                       ``reward >= 0.5`` then mean) or a record ``{details:
                       [{instance_id, correct}]}``; task = ``all``.
* ``omnidocbench``     the official OmniDocBench v1.5 end2end evaluator output
                       (driven by ``VLMEvalKit/omnidocbench_v15.py eval``, not
                       vlmeval ``run.py``): ``<model>/official_eval/result/
                       *_per_page_edit.json`` / ``*_per_table_TEDS.json`` /
                       ``*_per_sample_CDM.json``, or per-page ``{page:
                       {teds_mean | cdm_mean}}`` (``--subset``); task = ``all``.
                       The item universe is always the packaged ``ids.json``;
                       result keys outside it are dropped with a warning.
                       tab_teds items are table instances (``<page>_[idx]``);
                       a per-page file spreads the page mean over its instances.
                       formula_cdm items are pages: score = sum of the page's
                       CDM instances / GT instance count from ``counts.json``
                       (``--counts``; the official file merges matches into
                       ``_[i, j]`` keys, so its own count is not the GT count).
* ``ocrbench_v2``      VLMEvalKit ``*_per_question.json`` (list of ``{index, type,
                       score, ignored}``) or ``*_per_item_scores.jsonl`` (``{index,
                       task, metric, score, ignored}``); task = ``type``/``task``,
                       ``metric`` kept as the aggregation scenario.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCES = ("arenahard", "osworld_verified", "terminal_bench", "omnidocbench", "ocrbench_v2")

# --- OmniDocBench sub-metrics (mirrors VLMEvalKit/omnidocbench_v15/zip_subset.py) ---
# kind: "edit" (per-page, lower is better -> 1 - edit), "teds_instance"
# (per-instance, higher is better), "cdm" (per-instance file, folded per page
# as sum / GT instance count from counts.json, higher is better; mirrors
# zip_subset._aggregate_micro).
OMNIDOC_SUBSETS: Dict[str, Dict[str, Any]] = {
    "text_edit": dict(kind="edit", result_suffix="_text_block_per_page_edit.json"),
    "tab_edit": dict(kind="edit", result_suffix="_table_per_page_edit.json"),
    "tab_teds": dict(kind="teds_instance", result_suffix="_table_per_table_TEDS.json", per_page_file="table_teds_per_page.json"),
    "formula_edit": dict(kind="edit", result_suffix="_display_formula_per_page_edit.json"),
    "formula_cdm": dict(kind="cdm", result_suffix="_display_formula_per_sample_CDM.json", per_page_file="formula_cdm_per_page.json"),
    "read_order_edit": dict(kind="edit", result_suffix="_reading_order_per_page_edit.json"),
}
# CDM merges some matches, so the index part can be a list: "_[30, 32, 127]".
_INSTANCE_KEY_RE = re.compile(r"^(?P<page>.+)_\[[\d,\s]*\]$")

# --- ArenaHard verdict table (mirrors opencompass/zipbench/adapters/arenahard.py) ---
_VERDICT_RE = re.compile(r"\[\[([AB<>=]+)\]\]")
_SCORE_FOR_A = {"A>>B": 1.0, "A>B": 0.75, "A=B": 0.5, "B>A": 0.25, "B>>A": 0.0}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ids(path: Optional[str]) -> Optional[List[str]]:
    """Read an id universe: a JSON list, a JSON {id: ...} map, or a JSONL of
    ``{"question_id"|"id"|"index": ...}`` records. None if no path."""
    if not path:
        return None
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        ids: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("question_id", "id", "index"):
                if key in rec:
                    ids.append(str(rec[key]))
                    break
            else:
                raise ValueError(f"{path}: record without question_id/id/index: {rec!r}")
        return ids
    data = json.loads(text)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        return [str(k) for k in data]
    raise ValueError(f"{path}: expected a JSON list/dict or a JSONL file")


def _agent_adapter(name: str):
    """Import the agent adapter (installed package or ``<repo>/agent``)."""
    try:
        from zipbench_agent.adapters import get_adapter  # type: ignore
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT / "agent"))
        from zipbench_agent.adapters import get_adapter  # type: ignore
    return get_adapter(name)


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def _detail(score: float, task: str, raw: Dict[str, Any], missing: bool = False) -> Dict[str, Any]:
    return {"is_correct": _clamp01(score), "task": str(task), "missing": bool(missing), "raw": raw}


def _fill_missing(
    details: Dict[str, Dict[str, Any]],
    ids: Optional[Iterable[str]],
    worst: float,
    task_of: Any,
) -> List[str]:
    """Add every id from the universe that is absent, with the worst score."""
    if ids is None:
        return []
    missing: List[str] = []
    for qid in ids:
        qid = str(qid)
        if qid not in details:
            details[qid] = _detail(worst, task_of(qid), {}, missing=True)
            missing.append(qid)
    return missing


# ---------------------------------------------------------------------------
# per-source converters: each returns (details, meta)
# ---------------------------------------------------------------------------


def convert_arenahard(input_path: Path, ids: Optional[List[str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    path = Path(input_path)
    if path.is_dir():
        path = path / "arenahard.json"
    payload = _load_json(path)

    games: Dict[str, List[Optional[float]]] = {}
    capability: Dict[str, str] = {}
    n_unparsed = 0

    def add_game(qid: str, verdict: Optional[str], model_is_a: bool) -> None:
        nonlocal n_unparsed
        if verdict not in _SCORE_FOR_A:
            games.setdefault(qid, []).append(None)
            n_unparsed += 1
            return
        s = _SCORE_FOR_A[verdict]
        games.setdefault(qid, []).append(s if model_is_a else 1.0 - s)

    if isinstance(payload, dict) and payload:
        # opencompass judged output: {idx: {gold: {...}, prediction: "...[[A>>B]]..."}}
        for sample in payload.values():
            if not isinstance(sample, dict) or "gold" not in sample:
                continue
            gold = sample["gold"]
            qid = str(gold["question_id"])
            capability.setdefault(qid, str(gold.get("capability", "")))
            match = _VERDICT_RE.search(str(sample.get("prediction", "")))
            base_models = gold.get("base_models") or []
            add_game(qid, match.group(1) if match else None,
                     gold.get("answer1") not in base_models)
    elif isinstance(payload, list) and payload:
        # question list: [{question_id, capability, target_model, pairs: [{answer1, answer2, judge_label}]}]
        for q in payload:
            if not isinstance(q, dict) or "pairs" not in q:
                continue
            qid = str(q["question_id"])
            capability.setdefault(qid, str(q.get("capability", "")))
            target = q.get("target_model")
            for pair in q.get("pairs") or []:
                label = pair.get("judge_label")
                if label is None and pair.get("judge_raw"):
                    match = _VERDICT_RE.search(str(pair["judge_raw"]))
                    label = match.group(1) if match else None
                if target is not None:
                    model_is_a = pair.get("answer1") == target
                elif pair.get("target_model_pos") in ("A", "B"):
                    model_is_a = pair["target_model_pos"] == "A"
                else:
                    raise ValueError(f"{path}: question {qid}: cannot tell which side is the evaluated model")
                add_game(qid, label, model_is_a)
    else:
        raise ValueError(f"{path}: expected the opencompass judged dict {{idx: {{gold, prediction}}}} "
                         f"or a question list with 'pairs'")

    details: Dict[str, Dict[str, Any]] = {}
    n_no_parsed = 0
    for qid, scores in games.items():
        parsed = [s for s in scores if s is not None]
        if parsed:
            q_score = sum(parsed) / len(parsed)
        else:
            q_score = 0.0
            n_no_parsed += 1
        details[qid] = _detail(
            q_score, capability[qid],
            {"game_scores": scores, "num_games": len(scores), "num_parsed": len(parsed),
             "capability": capability[qid]},
        )
    missing = _fill_missing(details, ids, 0.0, lambda q: "")
    meta = {
        "direction": "higher_is_better",
        "score": "mean soft win rate over parsed games (A>>B=1 .. B>>A=0 from the evaluated side); "
                 "a question with no parseable verdict scores 0",
        "num_games": sum(len(v) for v in games.values()),
        "num_unparsed_games": n_unparsed,
        "num_questions_without_parsed_game": n_no_parsed,
        "missing_ids": missing,
    }
    return details, meta


def _record_details(input_path: Path) -> Optional[Any]:
    """``details`` of an already-converted record file, else None."""
    p = Path(input_path)
    if not p.is_file() or p.suffix != ".json":
        return None
    try:
        payload = _load_json(p)
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("details"), (dict, list)):
        return payload["details"]
    return None


def convert_osworld_verified(input_path: Path, ids: Optional[List[str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    adapter = _agent_adapter("osworld_verified")
    domains = adapter._domain_map()
    universe = ids if ids is not None else sorted(domains)

    def task_of(qid: str) -> str:
        return str(domains.get(qid, "unknown"))

    details: Dict[str, Dict[str, Any]] = {}
    errored: List[str] = []
    record = _record_details(input_path)
    if isinstance(record, dict):
        # converted record: {uuid: {application, is_correct, attempts...}}
        for qid, rec in record.items():
            qid = str(rec.get("instance_id", qid)) if isinstance(rec, dict) else str(qid)
            score = rec.get("is_correct") if isinstance(rec, dict) else rec
            task = str(rec.get("application") or task_of(qid)) if isinstance(rec, dict) else task_of(qid)
            if score is None:
                errored.append(qid)
                details[qid] = _detail(0.0, task, {"errored": True, "domain": task}, missing=True)
            else:
                details[qid] = _detail(float(score), task, {"score": float(score), "domain": task})
        how = "record is_correct"
    else:
        res = adapter.parse_results(input_path)
        for qid, score in res.scores.items():
            details[str(qid)] = _detail(score, task_of(str(qid)), {"score": float(score), "domain": task_of(str(qid))})
        for qid in sorted(res.errored):
            details[str(qid)] = _detail(0.0, task_of(str(qid)), {"errored": True, "domain": task_of(str(qid))}, missing=True)
        errored = sorted(res.errored)
        how = "harness reward clamped to [0, 1], several runs averaged (agent adapter parse_results)"
    missing = _fill_missing(details, universe, 0.0, task_of)
    meta = {
        "direction": "higher_is_better",
        "score": how,
        "errored_ids": errored,
        "missing_ids": missing,
    }
    return details, meta


def convert_terminal_bench(input_path: Path, ids: Optional[List[str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    details: Dict[str, Dict[str, Any]] = {}
    errored: List[str] = []
    record = _record_details(input_path)
    if record is not None:
        # converted record: details = [{instance_id, correct}] or {task: {correct|is_correct}}
        items = record.items() if isinstance(record, dict) else enumerate(record)
        for key, rec in items:
            if not isinstance(rec, dict):
                continue
            qid = str(rec.get("instance_id", rec.get("task_name", key)))
            score = rec.get("correct", rec.get("is_correct"))
            if score is None:
                errored.append(qid)
                details[qid] = _detail(0.0, "all", {"errored": True}, missing=True)
            else:
                details[qid] = _detail(float(score), "all", {"score": float(score)})
        how = "record correct (mean over trials of the 0/1 task reward)"
    else:
        res = _agent_adapter("terminal_bench").parse_results(input_path)
        for task, score in res.scores.items():
            details[str(task)] = _detail(score, "all", {"score": float(score)})
        for task in sorted(res.errored):
            details[str(task)] = _detail(0.0, "all", {"errored": True}, missing=True)
        errored = sorted(res.errored)
        how = "mean over trials of (reward >= 0.5); harbor terminal-bench@2.0 rewards are 0/1 already"
    missing = _fill_missing(details, ids, 0.0, lambda q: "all")
    meta = {
        "direction": "higher_is_better",
        "score": how,
        "errored_ids": errored,
        "missing_ids": missing,
    }
    return details, meta


def default_omnidoc_ids_path(subset: str) -> Path:
    return REPO_ROOT / "VLMEvalKit" / "vlmeval" / "zipbench" / "subsets" / f"OmniDocBench_{subset}" / "ids.json"


def default_omnidoc_counts_path(subset: str) -> Path:
    return default_omnidoc_ids_path(subset).with_name("counts.json")


def load_counts(path: Path) -> Dict[str, int]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON {{page: gt_instance_count}} map")
    return {str(k): int(v) for k, v in data.items()}


def _resolve_omnidoc_result_file(input_path: Path, suffix: str, subset: str) -> Path:
    if input_path.is_file():
        return input_path
    patterns = [f"*{suffix}", OMNIDOC_SUBSETS[subset].get("per_page_file", f"{subset}_per_page.json")]
    candidates: List[str] = []
    for pat in patterns:
        candidates = glob.glob(os.path.join(str(input_path), pat)) or \
            glob.glob(os.path.join(str(input_path), "**", pat), recursive=True)
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"no *{suffix} (or {patterns[1]}) under {input_path}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return Path(candidates[0])


def _instance_value(value: Any) -> float:
    return float(value["TEDS"]) if isinstance(value, dict) else float(value)


def _page_of(item_id: str) -> str:
    m = _INSTANCE_KEY_RE.match(item_id)
    return m.group("page") if m else item_id


def _per_page_mean(payload: Dict[str, Any], mean_key: str, count_key: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Per-page ``{page: {<mean_key>, <count_key>}}`` dicts (zooer/ZipBench layout), else None."""
    if payload and all(isinstance(v, dict) and mean_key in v for v in payload.values()):
        return {str(k): {"mean": float(v[mean_key]), "count": int(v.get(count_key, 0))} for k, v in payload.items()}
    return None


def convert_omnidocbench(
    input_path: Path,
    subset: str,
    ids: Optional[List[str]],
    counts_path: Optional[str] = None,
    ids_path: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if subset not in OMNIDOC_SUBSETS:
        raise ValueError(f"--subset must be one of {sorted(OMNIDOC_SUBSETS)}, got {subset!r}")
    cfg = OMNIDOC_SUBSETS[subset]
    ids_source = "explicit"
    if ids is None:
        ids_source = "default"
        ids = load_ids(str(default_omnidoc_ids_path(subset)))
    counts: Optional[Dict[str, int]] = None
    counts_file: Optional[Path] = None
    if cfg["kind"] == "cdm":
        # GT-side instance count per page: --counts, else counts.json next to
        # an explicit --ids, else the packaged counts.json.
        if counts_path:
            counts_file = Path(counts_path)
        elif ids_path and Path(ids_path).with_name("counts.json").is_file():
            counts_file = Path(ids_path).with_name("counts.json")
        else:
            counts_file = default_omnidoc_counts_path(subset)
        if not counts_file.is_file():
            raise FileNotFoundError(f"formula_cdm needs a GT counts.json (--counts); not found: {counts_file}")
        counts = load_counts(counts_file)
        no_count = sorted(q for q in ids if q not in counts)
        if no_count:
            raise ValueError(f"{counts_file}: no GT instance count for {len(no_count)} id(s), e.g. {no_count[:3]}")
    result_file = _resolve_omnidoc_result_file(Path(input_path), cfg["result_suffix"], subset)
    payload = _load_json(result_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{result_file}: expected a dict")

    details: Dict[str, Dict[str, Any]] = {}
    if cfg["kind"] == "edit":
        direction = "lower_is_better (stored as 1 - edit)"
        worst = 0.0
        for qid in ids:
            if qid in payload:
                edit = float(payload[qid])
                details[qid] = _detail(1.0 - edit, "all", {"edit": edit})
        missing = _fill_missing(details, ids, worst, lambda q: "all")
        for qid in missing:
            details[qid]["raw"] = {"edit": 1.0}
    elif cfg["kind"] == "teds_instance":
        direction = "higher_is_better"
        per_page = _per_page_mean(payload, "teds_mean", "table_count")
        if per_page is not None:
            # page-level TEDS means only: the item unit stays the table
            # instance (as in the evaluation), each instance of a page gets
            # the page mean; table_count must match the instances in ids
            n_inst: Dict[str, int] = {}
            for i in ids:
                n_inst[_page_of(i)] = n_inst.get(_page_of(i), 0) + 1
            bad = sorted(pg for pg, v in per_page.items() if pg in n_inst and v["count"] != n_inst[pg])
            if bad:
                raise ValueError(f"{result_file}: table_count differs from the instances in ids for "
                                 f"{len(bad)} page(s), e.g. {bad[:3]}")
            for qid in ids:
                pg = _page_of(qid)
                if pg in per_page:
                    details[qid] = _detail(per_page[pg]["mean"], "all",
                                           {"teds": per_page[pg]["mean"], "from_page_mean": True})
        else:
            for qid in ids:
                if qid in payload:
                    v = _instance_value(payload[qid])
                    details[qid] = _detail(v, "all", {"teds": v})
        missing = _fill_missing(details, ids, 0.0, lambda q: "all")
    else:  # cdm: one item per page = sum(instance CDM) / GT instance count
        direction = "higher_is_better"
        assert counts is not None
        per_page = _per_page_mean(payload, "cdm_mean", "formula_count")
        sums: Dict[str, Tuple[float, int]] = {}
        if per_page is not None:
            for pg, v in per_page.items():
                sums[pg] = (v["mean"] * v["count"], v["count"])
        else:
            by_page: Dict[str, List[float]] = {}
            for key, value in payload.items():
                m = _INSTANCE_KEY_RE.match(str(key))
                if not m:
                    raise ValueError(f"{result_file}: per-instance key {key!r} does not end in _[idx]")
                by_page.setdefault(m.group("page"), []).append(float(value))
            for pg, vals in by_page.items():
                sums[pg] = (sum(vals), len(vals))
        count_mismatch: List[str] = []
        for qid in ids:
            if qid in sums:
                cdm_sum, n_result = sums[qid]
                gt_count = counts[qid]
                if n_result != gt_count:
                    count_mismatch.append(qid)
                details[qid] = _detail(cdm_sum / gt_count if gt_count else 0.0, "all",
                                       {"cdm_sum": cdm_sum, "cdm_count": n_result, "gt_count": gt_count})
        missing = _fill_missing(details, ids, 0.0, lambda q: "all")
        for qid in missing:
            details[qid]["raw"] = {"cdm_sum": 0.0, "cdm_count": 0, "gt_count": counts[qid]}

    # a result key is "extra" if neither it nor its page is in ids (cdm keys
    # are instances of page ids; teds ids are instances themselves)
    id_set = set(ids) | {_page_of(i) for i in ids}
    extra = sorted(k for k in map(str, payload.keys()) if k not in id_set and _page_of(k) not in id_set)
    if extra:
        print(f"[WARN] omnidocbench/{subset}: {len(extra)} result key(s) not in ids.json dropped, "
              f"e.g. {extra[:3]}", file=sys.stderr)
    meta = {
        "direction": direction,
        "subset": subset,
        "result_file": str(result_file),
        "ids_source": ids_source,
        "missing_ids": missing,
        "num_result_keys_not_in_ids": len(extra),
        "result_keys_not_in_ids": extra[:20],
    }
    if cfg["kind"] == "cdm":
        meta["counts_file"] = str(counts_file)
        meta["cdm_count_mismatch_ids"] = count_mismatch
    return details, meta


def convert_ocrbench_v2(input_path: Path, ids: Optional[List[str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    path = Path(input_path)
    if path.is_dir():
        cands = sorted(glob.glob(os.path.join(str(path), "*_per_question.json"))
                       + glob.glob(os.path.join(str(path), "*_per_item_scores.jsonl")),
                       key=os.path.getmtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"no *_per_question.json / *_per_item_scores.jsonl under {path}")
        path = Path(cands[0])
    if path.suffix == ".jsonl":
        payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected the VLMEvalKit per_question list or a per_item_scores jsonl")

    details: Dict[str, Dict[str, Any]] = {}
    types: Dict[str, str] = {}
    n_ignored = n_score_missing = 0
    for pos, rec in enumerate(payload):
        if not isinstance(rec, dict):
            continue
        qid = str(rec.get("subset_index", rec.get("index", pos)))
        if qid in details:
            raise ValueError(f"{path}: duplicate index {qid!r}")
        typ = str(rec.get("type") or rec.get("task") or "unknown")
        metric = str(rec.get("metric") or typ)
        types[qid] = typ
        ignored = bool(rec.get("ignored", False))
        score = rec.get("score")
        score_missing = score is None
        if score_missing:
            n_score_missing += 1
        d = _detail(0.0 if score_missing else float(score), typ,
                    {"score": score, "type": typ, "metric": metric, "index": rec.get("index"),
                     "ignored": ignored, "score_missing": score_missing})
        d["type"] = typ
        d["metric"] = metric
        d["ignored"] = ignored
        n_ignored += ignored
        details[qid] = d
    missing = _fill_missing(details, ids, 0.0, lambda q: types.get(q, "unknown"))
    for qid in missing:
        details[qid]["type"] = details[qid]["metric"] = details[qid]["task"]
        details[qid]["ignored"] = False
    meta = {
        "direction": "higher_is_better",
        "result_file": str(path),
        "num_ignored": n_ignored,
        "num_score_missing": n_score_missing,
        "missing_ids": missing,
    }
    return details, meta


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def convert(
    source: str,
    input_path: str,
    ids_path: Optional[str] = None,
    subset: Optional[str] = None,
    counts_path: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    ids = load_ids(ids_path)
    p = Path(input_path)
    if source == "arenahard":
        return convert_arenahard(p, ids)
    if source == "osworld_verified":
        return convert_osworld_verified(p, ids)
    if source == "terminal_bench":
        return convert_terminal_bench(p, ids)
    if source == "omnidocbench":
        if not subset:
            raise ValueError("--subset is required for --source omnidocbench")
        return convert_omnidocbench(p, subset, ids, counts_path=counts_path, ids_path=ids_path)
    if source == "ocrbench_v2":
        return convert_ocrbench_v2(p, ids)
    raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


def build_record(source: str, model: str, dataset: str, details: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    if not details:
        raise ValueError("conversion produced no items")
    scores = [float(d["is_correct"]) for d in details.values()]
    mean = sum(scores) / len(scores)
    meta = dict(meta)
    meta.update({
        "source": source,
        "n_items": len(details),
        "n_missing": sum(1 for d in details.values() if d.get("missing")),
        "schema": "continuous_v1",
    })
    return {
        "dataset": dataset,
        "model": model,
        "source": source,
        "is_correct": mean,
        "accuracy": mean,
        "details": details,
        "meta": meta,
    }


def write_record(record: Dict[str, Any], output_root: str, model: str, dataset: str, overwrite: bool = False) -> Path:
    out_dir = Path(output_root) / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--input", required=True, help="Native result file or directory (see module docstring).")
    ap.add_argument("--model", required=True, help="Model name; becomes the output sub-directory.")
    ap.add_argument("--dataset", required=True, help="Dataset name; becomes <model>/<dataset>.json.")
    ap.add_argument("--output-root", required=True, help="Record root to write <model>/<dataset>.json under.")
    ap.add_argument("--subset", choices=sorted(OMNIDOC_SUBSETS), default=None, help="omnidocbench sub-metric.")
    ap.add_argument("--ids", default=None,
                    help="Optional id universe (JSON list / JSON map / JSONL with question_id). Items absent "
                         "from the native result get the worst score. omnidocbench defaults to the packaged "
                         "VLMEvalKit ids.json; osworld_verified defaults to the packaged task_domains.json.")
    ap.add_argument("--counts", default=None,
                    help="omnidocbench formula_cdm only: GT instance counts {page: n} (counts.json). Defaults "
                         "to counts.json next to --ids, else the packaged VLMEvalKit counts.json.")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing <model>/<dataset>.json.")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    details, meta = convert(args.source, args.input, args.ids, args.subset, counts_path=args.counts)
    record = build_record(args.source, args.model, args.dataset, details, meta)
    out_path = write_record(record, args.output_root, args.model, args.dataset, args.overwrite)
    print(f"source={args.source} model={args.model} dataset={args.dataset}")
    print(f"items={record['meta']['n_items']} missing={record['meta']['n_missing']} mean_is_correct={record['is_correct']:.6f}")
    print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
