"""ZipBench subset support for the OmniDocBench v1.5 standalone evaluator.

Six sub-metric specs (``vlmeval/zipbench/subsets/OmniDocBench_<sub>/``), one
inference run over the union of their pages, one official evaluation over the
subset GT, then a weighted aggregation (``zip_summarize``) on top of the
official per-page / per-instance result files:

* the four Edit metrics estimate the official page-average directly:
  ``weighted = sum(w_p * edit_p)``;
* table TEDS: the spec weights table INSTANCES (ids ``<page>_[gt_idx]``), so
  ``weighted = sum(w_i * TEDS_i)`` estimates the headline instance-micro
  number directly;
* formula CDM headline is an instance-micro number, estimated as a ratio
  ``sum(w_p * sum_p) / sum(w_p * count_p)`` from the per-instance file
  grouped by page.

Missing pages / instances (no prediction .md / skipped by the official
scorer) score worst (edit=1.0; teds=0.0; cdm sum=0 with the GT-side instance
count in the denominator) and are reported in ``secondary.missing_pages`` --
the denominator never silently shrinks.

Stdlib-only: the zipbench spec module is loaded by file path so this works in
any Python without importing the vlmeval package.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .common import _dump_json, _load_json, _write_csv_rows

REPO_ROOT = Path(__file__).resolve().parent.parent

SUBSET_CHOICES = ("full", "small", "tiny")

#: The six sub-metrics; order == score report order. ``result_suffix`` matches
#: the official result filename (prefix varies with the pred dir name, so we
#: glob). ``kind``: how the headline folds pages (page_avg vs instance micro).
OMNIDOC_METRICS = {
    "text_edit": dict(
        kind="edit", result_suffix="_text_block_per_page_edit.json",
        enters_overall=True, label="Text Edit_dist"),
    "tab_edit": dict(
        kind="edit", result_suffix="_table_per_page_edit.json",
        enters_overall=False, label="Table Edit_dist"),
    "tab_teds": dict(
        kind="teds_instance", result_suffix="_table_per_table_TEDS.json",
        enters_overall=True, label="Table TEDS"),
    "formula_edit": dict(
        kind="edit", result_suffix="_display_formula_per_page_edit.json",
        enters_overall=False, label="Formula Edit_dist"),
    "formula_cdm": dict(
        kind="cdm", result_suffix="_display_formula_per_sample_CDM.json",
        enters_overall=True, label="Formula CDM"),
    "read_order_edit": dict(
        kind="edit", result_suffix="_reading_order_per_page_edit.json",
        enters_overall=False, label="Reading Order Edit_dist"),
}

# CDM merges some matches, so the index part can be a list: "_[30, 32, 127]".
_INSTANCE_KEY_RE = re.compile(r"^(?P<page>.+)_\[[\d,\s]*\]$")


def _spec_module():
    path = REPO_ROOT / "vlmeval" / "zipbench" / "spec.py"
    spec = importlib.util.spec_from_file_location("_omnidoc_zipbench_spec", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_specs(subset: str) -> Dict[str, Dict[str, float]]:
    """-> {sub_metric: {item_id: weight}} for one subset name.

    Item ids are page basenames, except tab_teds whose spec weights table
    instances (``<page>_[gt_idx]``).
    """
    if subset not in SUBSET_CHOICES or subset == "full":
        raise ValueError(f"load_specs wants a non-full subset, got {subset!r}")
    spec_mod = _spec_module()
    out: Dict[str, Dict[str, float]] = {}
    for sub in OMNIDOC_METRICS:
        records = spec_mod.load_subset_spec(f"OmniDocBench_{sub}", subset)
        out[sub] = {str(rec["index"]): float(rec["weight"]) for rec in records}
    return out


def load_manifests() -> Dict[str, Dict[str, Any]]:
    spec_mod = _spec_module()
    return {sub: spec_mod.load_manifest(f"OmniDocBench_{sub}") for sub in OMNIDOC_METRICS}


def load_gt_instance_counts(sub: str) -> Dict[str, int]:
    """GT-side instance count per page (tab_teds / formula_cdm missing-page fill)."""
    spec_mod = _spec_module()
    path = os.path.join(spec_mod.subsets_dir(f"OmniDocBench_{sub}"), "counts.json")
    return {str(k): int(v) for k, v in _load_json(path).items()}


def union_pages(specs: Dict[str, Dict[str, float]]) -> Set[str]:
    """All pages the subset run must infer; instance ids fold to their page."""
    pages: Set[str] = set()
    for weights in specs.values():
        for item in weights:
            m = _INSTANCE_KEY_RE.match(item)
            pages.add(m.group("page") if m else item)
    return pages


def filter_gt_samples(samples: List[Dict[str, Any]], pages: Set[str]) -> List[Dict[str, Any]]:
    """Keep GT samples whose page basename is in ``pages``; loud on misses."""
    selected = [
        s for s in samples
        if os.path.basename(str(s["page_info"]["image_path"])) in pages
    ]
    found = {os.path.basename(str(s["page_info"]["image_path"])) for s in selected}
    missing = sorted(pages - found)
    if missing:
        raise ValueError(
            f"{len(missing)} subset pages are not in the GT JSON -- the spec is stale "
            f"or the GT drifted. First missing: {missing[:5]}")
    return selected


def run_dir_suffix(subset: Optional[str]) -> str:
    return "" if subset in (None, "full") else f"_ZIP_{subset}"


# ---------------------------------------------------------------------------
# Weighted aggregation over the official result files
# ---------------------------------------------------------------------------


def _find_result_file(result_dir: str, suffix: str) -> str:
    candidates = glob.glob(os.path.join(result_dir, f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"no *{suffix} in {result_dir}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _group_instances(per_instance: Dict[str, Any], value_key: Optional[str]) -> Dict[str, Tuple[float, int]]:
    """{page: (sum, count)} from a per-instance dict keyed ``<page>_[idx]``."""
    grouped: Dict[str, List[float]] = {}
    for key, value in per_instance.items():
        m = _INSTANCE_KEY_RE.match(key)
        if not m:
            raise ValueError(f"per-instance key {key!r} does not end in _[idx]")
        score = float(value[value_key]) if value_key else float(value)
        grouped.setdefault(m.group("page"), []).append(score)
    return {page: (sum(vals), len(vals)) for page, vals in grouped.items()}


def _aggregate_edit(weights: Dict[str, float], per_page: Dict[str, Any]) -> Dict[str, Any]:
    missing = sorted(p for p in weights if p not in per_page)
    weighted = sum(w * (float(per_page[p]) if p in per_page else 1.0)
                   for p, w in weights.items())
    return {
        "weighted": weighted,
        "missing_pages": missing,
        "n_pages": len(weights),
    }


def _instance_value(value: Any) -> float:
    return float(value["TEDS"]) if isinstance(value, dict) else float(value)


def _aggregate_instance(weights: Dict[str, float], per_instance: Dict[str, Any]) -> Dict[str, Any]:
    """Weighted mean over spec-selected instances; a missing instance scores 0."""
    missing = sorted(k for k in weights if k not in per_instance)
    weighted = sum(w * (_instance_value(per_instance[k]) if k in per_instance else 0.0)
                   for k, w in weights.items())
    return {
        "weighted": weighted,
        "missing_pages": missing,
        "n_pages": len(weights),
    }


def _aggregate_micro(
    weights: Dict[str, float],
    by_page: Dict[str, Tuple[float, int]],
    gt_counts: Dict[str, int],
) -> Dict[str, Any]:
    missing = sorted(p for p in weights if p not in by_page)
    num = den = page_avg = 0.0
    for p, w in weights.items():
        if p in by_page:
            s, c = by_page[p]
        else:
            s, c = 0.0, gt_counts[p]
        num += w * s
        den += w * c
        page_avg += w * (s / c if c else 0.0)
    return {
        "weighted": num / den if den else None,
        "weighted_page_avg": page_avg,
        "missing_pages": missing,
        "n_pages": len(weights),
    }


def zip_summarize(
    result_dir: str,
    out_dir: str,
    subset: str,
    official_summary: Dict[str, Any],
    score_json_path: str,
    score_csv_path: str,
) -> Dict[str, Any]:
    """Weighted zipbench scores from a subset run's official result files.

    Writes ``zip_score.json`` (zipbench standard shape), the weighted
    ``score.json``/``score.csv``, and a per-page detail tsv. The official
    unweighted summary must already have been written to
    ``score_unweighted.json`` by the caller and is embedded as diagnostics.
    """
    specs = load_specs(subset)
    manifests = load_manifests()

    per_metric: Dict[str, Dict[str, Any]] = {}
    detail_rows: List[Dict[str, Any]] = []
    for sub, cfg in OMNIDOC_METRICS.items():
        weights = specs[sub]
        path = _find_result_file(result_dir, cfg["result_suffix"])
        payload = _load_json(path)
        if cfg["kind"] == "edit":
            agg = _aggregate_edit(weights, payload)
            for p, w in sorted(weights.items()):
                present = p in payload
                detail_rows.append({
                    "metric": sub, "page": p, "zip_weight": w,
                    "value": float(payload[p]) if present else 1.0,
                    "instance_sum": None, "instance_count": None,
                    "missing": not present,
                })
        elif cfg["kind"] == "teds_instance":
            agg = _aggregate_instance(weights, payload)
            for k, w in sorted(weights.items()):
                present = k in payload
                detail_rows.append({
                    "metric": sub, "page": k, "zip_weight": w,
                    "value": _instance_value(payload[k]) if present else 0.0,
                    "instance_sum": None, "instance_count": None,
                    "missing": not present,
                })
        else:
            value_key = None
            by_page = _group_instances(payload, value_key)
            gt_counts = load_gt_instance_counts(sub)
            agg = _aggregate_micro(weights, by_page, gt_counts)
            for p, w in sorted(weights.items()):
                present = p in by_page
                s, c = by_page[p] if present else (0.0, gt_counts[p])
                detail_rows.append({
                    "metric": sub, "page": p, "zip_weight": w,
                    "value": (s / c if c else None),
                    "instance_sum": s, "instance_count": c,
                    "missing": not present,
                })
        agg["result_file"] = os.path.basename(path)
        per_metric[sub] = agg

    text_w = per_metric["text_edit"]["weighted"]
    teds_w = per_metric["tab_teds"]["weighted"]
    cdm_w = per_metric["formula_cdm"]["weighted"]
    overall_weighted = None
    if None not in (text_w, teds_w, cdm_w):
        overall_weighted = ((1.0 - text_w) * 100.0 + teds_w * 100.0 + cdm_w * 100.0) / 3.0

    anchor_mae = {sub: manifests[sub]["subsets"][subset].get("anchor_mae")
                  for sub in OMNIDOC_METRICS}

    zip_score = {
        "score": overall_weighted,
        "metric": (
            "Overall = ((1-text_edit)*100 + table_teds*100 + formula_cdm*100)/3, weighted "
            "zipbench estimate; edit metrics = sum(w_p * edit_p) (page-avg); "
            "table_teds = sum(w_i * TEDS_i) over spec table instances (instance-micro); "
            "cdm = instance-micro ratio estimate sum(w_p*sum_p)/sum(w_p*count_p). "
            "Missing items score worst (edit=1.0; teds=0.0; cdm sum=0, "
            "count=GT instance count)."
        ),
        "subset": subset,
        "secondary": {
            "weighted": {sub: per_metric[sub]["weighted"] for sub in OMNIDOC_METRICS},
            "weighted_page_avg": {
                sub: per_metric[sub]["weighted_page_avg"]
                for sub in ("formula_cdm",)
            },
            "unweighted_subset_official": official_summary.get("metrics"),
            "unweighted_subset_overall": official_summary.get("overall"),
            "missing_pages": {
                sub: per_metric[sub]["missing_pages"] for sub in OMNIDOC_METRICS
                if per_metric[sub]["missing_pages"]
            },
            "num_pages": {sub: per_metric[sub]["n_pages"] for sub in OMNIDOC_METRICS},
            "weight_sum": {sub: sum(specs[sub].values()) for sub in OMNIDOC_METRICS},
            "anchor_mae": anchor_mae,
            "result_files": {sub: per_metric[sub]["result_file"] for sub in OMNIDOC_METRICS},
        },
    }
    _dump_json(zip_score, os.path.join(out_dir, "zip_score.json"))

    weighted_summary = {
        "complete": overall_weighted is not None,
        "formula_metric": official_summary.get("formula_metric"),
        "overall": overall_weighted,
        "subset": subset,
        "weighting": "zipbench weighted estimate (see zip_score.json)",
        "metrics": {
            "text_edit": text_w,
            "table_teds": teds_w,
            "formula_cdm": cdm_w,
            "formula_edit": per_metric["formula_edit"]["weighted"],
            "table_edit": per_metric["tab_edit"]["weighted"],
            "reading_order_edit": per_metric["read_order_edit"]["weighted"],
        },
    }
    _dump_json(weighted_summary, score_json_path)

    rows = [{"Metric": "Overall-Weighted", "Value": overall_weighted,
             "Complete": overall_weighted is not None}]
    key_of = {"text_edit": "Text Edit_dist", "tab_teds": "Table TEDS",
              "formula_cdm": "Formula CDM", "formula_edit": "Formula Edit_dist",
              "tab_edit": "Table Edit_dist", "read_order_edit": "Reading Order Edit_dist"}
    for sub in ("text_edit", "tab_teds", "formula_cdm", "formula_edit", "tab_edit",
                "read_order_edit"):
        value = per_metric[sub]["weighted"]
        rows.append({"Metric": f"{key_of[sub]}-Weighted", "Value": value,
                     "Complete": value is not None})
    _write_csv_rows(rows, score_csv_path)

    detail_path = os.path.join(out_dir, "zip_per_page.tsv")
    with open(detail_path, "w", encoding="utf-8") as f:
        cols = ["metric", "page", "zip_weight", "value", "instance_sum",
                "instance_count", "missing"]
        f.write("\t".join(cols) + "\n")
        for row in detail_rows:
            f.write("\t".join("" if row[c] is None else str(row[c]) for c in cols) + "\n")

    return zip_score
