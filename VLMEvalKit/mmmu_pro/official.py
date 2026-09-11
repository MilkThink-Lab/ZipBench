"""Official MMMU repo integration: clone, JSONL export, official evaluate.py
invocation and score summarization."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import string
import subprocess
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd

from .common import (
    DOMAIN_CAT2SUB_CAT,
    EXPECTED_NUM_SAMPLES,
    ZIP_WEIGHT_COL,
    is_missing_prediction,
    load_zip_spec_weights,
    read_table,
    resolve_run_dir,
    write_table,
)

DEFAULT_OFFICIAL_REPO = "https://github.com/MMMU-Benchmark/MMMU.git"
OFFICIAL_METHOD_CHOICES = ("direct", "cot")


def ensure_official_mmmu_pro(
    work_dir: str,
    repo_url: Optional[str] = None,
    ref: Optional[str] = None,
) -> str:
    """Clone / reuse official MMMU repo under work_dir/third_party/MMMU.

    Returns path to mmmu-pro/ directory.
    """
    repo_url = repo_url or DEFAULT_OFFICIAL_REPO
    third_party = os.path.join(work_dir, "third_party")
    repo_dir = os.path.join(third_party, "MMMU")
    mmmu_pro_dir = os.path.join(repo_dir, "mmmu-pro")

    if not os.path.isdir(repo_dir):
        os.makedirs(third_party, exist_ok=True)
        print(f"Cloning official MMMU repo -> {repo_dir}")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", repo_url, repo_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if ref:
        print(f"Checking out official ref: {ref}")
        subprocess.check_call(
            ["git", "-C", repo_dir, "fetch", "--depth", "1", "origin", ref],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "-C", repo_dir, "checkout", ref],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if not os.path.isdir(mmmu_pro_dir):
        raise RuntimeError(
            f"Expected mmmu-pro/ directory at {mmmu_pro_dir} but not found."
        )
    return mmmu_pro_dir


def resolve_official_method(args: argparse.Namespace) -> str:
    """Resolve the official MMMU-Pro method tag used in JSONL filenames.

    Explicit --official-method wins; otherwise prompt versions containing
    "cot" map to "cot" and everything else maps to "direct" (the official
    evaluate.py filename regex only accepts these two tags).
    """
    official_method = getattr(args, "official_method", None)
    if official_method:
        return official_method
    prompt_version = getattr(args, "prompt_version", "direct") or "direct"
    return "cot" if "cot" in prompt_version else "direct"


def strip_thinking_for_eval(text: str, all_choices: list, index2ans: dict) -> str:
    """Pre-extract the answer from full text (including thinking), then return a
    clean response for the official eval parser.

    Uses the same rfind-based logic as the official parser on the FULL text
    (where thinking helps rather than hurts), then returns just "Answer: X" so
    the official parser gets an unambiguous input.  Falls back to the post-think
    text when no answer letter can be extracted.
    """
    text = str(text)

    # --- Try to extract answer letter from FULL text (same logic as official parser) ---
    last_pos = text.rfind("Answer:")
    if last_pos != -1:
        answer_str = text[last_pos + len("Answer:"):].strip()
        matching = [c for c in all_choices if c in answer_str]
        if len(matching) == 1:
            return f"Answer: {matching[0]}"

    # Fallback pattern matching on full text (mirrors official parser fallback)
    search_text = text
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        search_text = search_text.strip(char)
    search_text = " " + search_text + " "

    candidates = []
    ans_with_brack = False
    for choice in all_choices:
        if f"({choice})" in search_text:
            candidates.append(choice)
            ans_with_brack = True
    if not candidates:
        for choice in all_choices:
            if f"{choice} " in search_text:
                candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f"{choice}." in search_text:
                candidates.append(choice)

    if len(candidates) == 1:
        return f"Answer: {candidates[0]}"
    elif len(candidates) > 1:
        if ans_with_brack:
            start_indexes = [search_text.rfind(f"({c})") for c in candidates]
        else:
            start_indexes = [search_text.rfind(f" {c} ") for c in candidates]
        import numpy as np
        best = candidates[np.argmax(start_indexes)]
        return f"Answer: {best}"

    # Cannot extract — return post-think text for content-based matching.
    # Split on the closing tag only: Qwen-style thinking models often emit
    # "</think>" without an opening "<think>" (it lives in the chat template).
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def export_predictions_to_official_jsonl(
    pred_df: pd.DataFrame,
    dataset: str,
    model_name: str,
    output_dir: str,
    official_method: str,
) -> str:
    """Convert prediction TSV to official-format JSONL.

    Returns path to generated JSONL file.
    """
    setting = "standard" if dataset == "MMMU_Pro_10c" else "vision"
    fname = f"{model_name}_{setting}_{official_method}.jsonl"
    jsonl_path = os.path.join(output_dir, fname)
    os.makedirs(output_dir, exist_ok=True)

    option_cols = list(string.ascii_uppercase[:10])  # A-J
    records = []
    for i in range(len(pred_df)):
        row = pred_df.iloc[i]
        # Build options list from A-J columns
        options = []
        for c in option_cols:
            if c in row and pd.notna(row[c]):
                options.append(str(row[c]))
        prediction = row.get("prediction", "")
        if is_missing_prediction(prediction):
            prediction = ""
        # Build choice info for answer extraction
        start_chr = "A"
        all_choices = [chr(ord(start_chr) + j) for j in range(len(options))]
        index2ans = {chr(ord(start_chr) + j): opt for j, opt in enumerate(options)}
        # Pre-extract answer from full text (including thinking) for clean eval
        response_for_eval = strip_thinking_for_eval(str(prediction), all_choices, index2ans)
        records.append(
            {
                "id": str(row.get("id", "")),
                "response": response_for_eval,
                "answer": str(row.get("answer", "")),
                "options": str(options),
                "subdomain": str(row.get("category", "")),
            }
        )

    # Official evaluate.py requires exactly EXPECTED_NUM_SAMPLES records
    if len(records) != EXPECTED_NUM_SAMPLES:
        print(
            f"WARNING: {len(records)} records found, but official evaluate.py "
            f"expects {EXPECTED_NUM_SAMPLES}. Padding or truncating may affect results."
        )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Exported {len(records)} records to {jsonl_path}")
    return jsonl_path


def run_official_eval_inprocess(
    mmmu_pro_dir: str,
    eval_output_dir: str,
) -> None:
    """Score every JSONL in *eval_output_dir* with the official per-record logic.

    The official evaluate.py hard-skips any file whose record count differs
    from 1730 (NUM gate in check_files), so ZipBench subsets cannot go through
    the subprocess path. Instead we import the official script as a module and
    apply its own ``mmmu_process_results`` record by record — byte-identical
    parsing/judging, minus the count gate — then write ``pred_indexs`` /
    ``if_right`` back into the JSONL exactly like the official main loop does.
    """
    eval_script = os.path.join(mmmu_pro_dir, "evaluate.py")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"Official evaluate.py not found at {eval_script}")

    spec = importlib.util.spec_from_file_location("_mmmu_pro_official_evaluate", eval_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # guarded by __main__, so nothing runs

    for fname in sorted(os.listdir(eval_output_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(eval_output_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        processed = [module.mmmu_process_results(rec) for rec in records]
        right = sum(1 for rec in processed if rec["if_right"])
        acc = right / len(processed) * 100 if processed else 0.0
        print(
            f"In-process official scoring: {fname} — {right}/{len(processed)} "
            f"correct ({acc:.2f}%, unweighted)"
        )
        with open(path, "w", encoding="utf-8") as f:
            for rec in processed:
                f.write(json.dumps(rec) + "\n")


def run_official_eval(
    mmmu_pro_dir: str,
    eval_output_dir: str,
) -> None:
    """Run official evaluate.py with cwd pointing to a directory containing output/."""
    eval_script = os.path.join(mmmu_pro_dir, "evaluate.py")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"Official evaluate.py not found at {eval_script}")

    # The official script expects to be run from a directory with ./output/*.jsonl
    # We copy the script to the eval output root and run from there
    eval_root = os.path.dirname(eval_output_dir)  # parent of output/
    script_copy = os.path.join(eval_root, "evaluate.py")
    shutil.copy2(eval_script, script_copy)

    print(f"Running official evaluate.py in {eval_root}")
    result = subprocess.run(
        [sys.executable, script_copy],
        cwd=eval_root,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"WARNING: official evaluate.py exited with code {result.returncode}")


def summarize_official_eval_output(
    jsonl_path: str,
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read back official JSONL (with pred_indexs / if_right) and produce summary.

    Returns (score_df, detail_df).
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # Build id -> index mapping from pred_df
    id_to_index = {}
    if "index" in pred_df.columns and "id" in pred_df.columns:
        for i in range(len(pred_df)):
            id_to_index[str(pred_df.iloc[i]["id"])] = pred_df.iloc[i]["index"]

    # Build detail
    detail_rows = []
    for rec in records:
        row_id = rec.get("id", "")
        detail_rows.append(
            {
                "id": row_id,
                "index": id_to_index.get(str(row_id), ""),
                "category": rec.get("subdomain", ""),
                "answer": rec.get("answer", ""),
                "prediction": rec.get("response", ""),
                "pred_indexs": rec.get("pred_indexs", ""),
                "if_right": rec.get("if_right", False),
            }
        )
    detail_df = pd.DataFrame(detail_rows)

    # Build score by category
    cat_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for rec in records:
        cat = rec.get("subdomain", "Unknown")
        cat_stats[cat]["total"] += 1
        if rec.get("if_right", False):
            cat_stats[cat]["correct"] += 1

    # Aggregate by domain
    score_rows = []
    for domain, subcats in DOMAIN_CAT2SUB_CAT.items():
        domain_total = 0
        domain_correct = 0
        for sc in subcats:
            if sc in cat_stats:
                s = cat_stats[sc]
                domain_total += s["total"]
                domain_correct += s["correct"]
                score_rows.append(
                    {
                        "Category": sc,
                        "Domain": domain,
                        "Total": s["total"],
                        "Correct": s["correct"],
                        "Accuracy": round(s["correct"] / s["total"] * 100, 2)
                        if s["total"] > 0
                        else 0.0,
                    }
                )
        if domain_total > 0:
            score_rows.append(
                {
                    "Category": f"Overall-{domain}",
                    "Domain": domain,
                    "Total": domain_total,
                    "Correct": domain_correct,
                    "Accuracy": round(domain_correct / domain_total * 100, 2),
                }
            )

    # Overall
    all_total = sum(s["total"] for s in cat_stats.values())
    all_correct = sum(s["correct"] for s in cat_stats.values())
    score_rows.append(
        {
            "Category": "Overall",
            "Domain": "All",
            "Total": all_total,
            "Correct": all_correct,
            "Accuracy": round(all_correct / all_total * 100, 2)
            if all_total > 0
            else 0.0,
        }
    )

    score_df = pd.DataFrame(score_rows)
    return score_df, detail_df


def weighted_zip_summary(
    detail_df: pd.DataFrame,
    score_df: pd.DataFrame,
    dataset: str,
    subset: str,
    run_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ZipBench weighted aggregation on top of the official per-item results.

    Joins the subset spec weights onto ``detail_df`` by ``index``, computes the
    weighted micro accuracy (the subset's estimate of the official full-set
    Overall), appends an ``Overall-Weighted`` row to ``score_df`` and writes
    ``run_dir/zip_score.json``. Per-category rows stay unweighted diagnostics.
    """
    from vlmeval.zipbench.report import (
        group_breakdown,
        plain_mean,
        weighted_mean,
        zip_result,
    )
    from vlmeval.zipbench.spec import load_manifest

    weights = load_zip_spec_weights(dataset, subset)
    manifest = load_manifest(dataset)
    subset_info = manifest["subsets"][subset]

    frame = detail_df.copy()
    missing = [i for i in frame["index"] if i not in weights]
    if missing:
        raise ValueError(
            f"{len(missing)} scored items are not in the '{subset}' spec "
            f"(first: {missing[:5]}) — the prediction file does not match this subset."
        )
    if len(frame) != len(weights):
        raise ValueError(
            f"Scored {len(frame)} items but the '{subset}' spec has {len(weights)} — "
            f"a partial evaluation would bias the weighted estimate."
        )
    frame[ZIP_WEIGHT_COL] = [weights[i] for i in frame["index"]]
    frame["_hit"] = frame["if_right"].astype(bool).astype(float)

    # Domain column for the diagnostic breakdown
    cat2domain = {
        sc: domain for domain, subcats in DOMAIN_CAT2SUB_CAT.items() for sc in subcats
    }
    frame["domain"] = [cat2domain.get(c, "Unknown") for c in frame["category"]]

    weighted_acc = weighted_mean(frame, "_hit") * 100
    unweighted_acc = plain_mean(frame, "_hit") * 100

    metric = (
        "micro_accuracy — weighted version of the official MMMU-Pro Overall "
        f"(sum(w*if_right)/sum(w)) on the '{subset}' subset; per-item scoring "
        "is the official MMMU repo evaluate.py"
    )
    report = zip_result(
        weighted_acc,
        metric,
        {
            "weighted_accuracy": weighted_acc,
            "accuracy_unweighted": unweighted_acc,
            "num_samples": int(len(frame)),
            "weight_sum": float(frame[ZIP_WEIGHT_COL].sum()),
            "subset": subset,
            "dataset": dataset,
            "dataset_size": manifest.get("dataset_size"),
            "ratio": subset_info.get("ratio"),
            "anchor_mae": subset_info.get("anchor_mae"),
            "per_group": group_breakdown(frame, "_hit", groups=("domain", "category")),
        },
    )
    zip_path = os.path.join(run_dir, "zip_score.json")
    with open(zip_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved weighted summary to {zip_path}")

    weighted_row = pd.DataFrame(
        [
            {
                "Category": "Overall-Weighted",
                "Domain": "All",
                "Total": int(len(frame)),
                "Correct": int(frame["_hit"].sum()),
                "Accuracy": round(weighted_acc, 2),
            }
        ]
    )
    score_df = pd.concat([score_df, weighted_row], ignore_index=True)

    detail_df = detail_df.copy()
    detail_df[ZIP_WEIGHT_COL] = frame[ZIP_WEIGHT_COL].values
    return score_df, detail_df, report


def evaluate_mmmu_pro(args: argparse.Namespace) -> None:
    if not args.work_dir:
        raise ValueError("--work-dir is required for eval")

    prompt_version = getattr(args, "prompt_version", "direct")
    # API runs derive max_tokens from the model config (set by the backend as
    # max_tokens_resolved); otherwise fall back to the CLI --max-tokens value.
    max_tokens = getattr(args, "max_tokens_resolved", None)
    if max_tokens is None:
        max_tokens = getattr(args, "max_tokens", 2048)
    subset = getattr(args, "subset", "full") or "full"
    run_dir = resolve_run_dir(
        args.work_dir, args.dataset, args.model_name, prompt_version, max_tokens,
        subset=subset,
        run_tag=getattr(args, "run_tag", ""),
    )

    # Resolve input: explicit --input wins, otherwise infer from run_dir
    input_path = getattr(args, "input", None)
    if not input_path:
        input_path = os.path.join(run_dir, "pred.tsv")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Prediction file not found: {input_path}")

    pred_df = read_table(input_path)
    if "prediction" not in pred_df.columns:
        raise ValueError("Input file must contain a 'prediction' column")
    print(f"Loaded {len(pred_df)} predictions from {input_path}")

    if subset != "full":
        spec_weights = load_zip_spec_weights(args.dataset, subset)
        if len(pred_df) != len(spec_weights):
            raise ValueError(
                f"Prediction file has {len(pred_df)} rows but the '{subset}' spec "
                f"of {args.dataset} has {len(spec_weights)} — wrong subset or "
                f"incomplete inference."
            )

    # Ensure official repo
    mmmu_pro_dir = ensure_official_mmmu_pro(
        args.work_dir,
        getattr(args, "official_repo", None),
        getattr(args, "official_ref", None),
    )

    # Create eval workspace inside run_dir
    eval_root = os.path.join(run_dir, "official_eval")
    eval_output_dir = os.path.join(eval_root, "output")
    os.makedirs(eval_output_dir, exist_ok=True)
    official_method = resolve_official_method(args)

    # Export to official JSONL
    jsonl_path = export_predictions_to_official_jsonl(
        pred_df, args.dataset, args.model_name, eval_output_dir, official_method
    )

    # Run official eval. Subset JSONLs would be skipped by the official
    # script's NUM==1730 gate, so they are scored in-process with the same
    # official per-record function instead.
    if subset == "full":
        run_official_eval(mmmu_pro_dir, eval_output_dir)
    else:
        run_official_eval_inprocess(mmmu_pro_dir, eval_output_dir)

    # Summarize
    score_df, detail_df = summarize_official_eval_output(jsonl_path, pred_df)

    if subset != "full":
        score_df, detail_df, zip_report = weighted_zip_summary(
            detail_df, score_df, args.dataset, subset, run_dir
        )
        print(
            f"\nZipBench weighted Overall ({subset}): "
            f"{zip_report['score']:.2f} "
            f"(unweighted on subset: {zip_report['secondary']['accuracy_unweighted']:.2f})"
        )

    # Resolve output paths – default to run_dir
    score_path = getattr(args, "output_score", None) or os.path.join(
        run_dir, "score.csv"
    )
    detail_path = getattr(args, "output_detail", None) or os.path.join(
        run_dir, "detail.tsv"
    )

    write_table(score_df, score_path)
    write_table(detail_df, detail_path)

    print(f"\nSaved score to {score_path}")
    print(f"Saved detail to {detail_path}")
    print("\n" + score_df.to_string(index=False))

    # Cleanup
    if not getattr(args, "keep_official_output", False):
        shutil.rmtree(eval_root, ignore_errors=True)
