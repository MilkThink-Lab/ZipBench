"""Official OmniDocBench v1.5 evaluation: repo management, end2end config,
Docker/local scorer invocation, and score summarization."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

from .common import (
    _abspath,
    _dump_json,
    _ensure_dir,
    _load_json,
    _output_dir,
    _write_csv_rows,
    load_gt_samples,
)


DEFAULT_OFFICIAL_REPO = "https://github.com/opendatalab/OmniDocBench.git"
DEFAULT_OFFICIAL_REF = "v1_5"
DEFAULT_OFFICIAL_DIRNAME = "OmniDocBench_v1_5"
DEFAULT_DOCKER_IMAGE = "sunyuefeng/omnidocbench-env:v1.5"


def _default_official_dir(work_dir: str) -> str:
    return os.path.join(work_dir, "third_party", DEFAULT_OFFICIAL_DIRNAME)


def ensure_official_repo(
    work_dir: str,
    official_dir: Optional[str] = None,
    repo_url: Optional[str] = None,
    ref: Optional[str] = None,
    fatal: bool = True,
) -> Optional[str]:
    """Clone/reuse the official OmniDocBench repo and checkout v1_5."""
    repo_url = repo_url or DEFAULT_OFFICIAL_REPO
    ref = ref or DEFAULT_OFFICIAL_REF
    repo_dir = _abspath(official_dir or _default_official_dir(work_dir))

    try:
        if not os.path.isdir(repo_dir):
            os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
            print(f"Cloning official OmniDocBench repo -> {repo_dir}")
            subprocess.check_call(
                ["git", "clone", "--depth", "1", "--branch", ref, repo_url, repo_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif os.path.isdir(os.path.join(repo_dir, ".git")):
            current_ref = subprocess.run(
                ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if current_ref != ref:
                print(f"Checking out official OmniDocBench ref: {ref}")
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

        pdf_validation = os.path.join(repo_dir, "pdf_validation.py")
        if not os.path.isfile(pdf_validation):
            raise RuntimeError(f"Official pdf_validation.py not found at {pdf_validation}")
        return repo_dir
    except Exception as exc:
        if fatal:
            raise RuntimeError(f"Failed to prepare official OmniDocBench repo: {exc}") from exc
        print(f"Warning: failed to prepare official OmniDocBench repo: {exc}")
        return None


# ---------------------------------------------------------------------------
# Official evaluation
# ---------------------------------------------------------------------------


def _subset_gt_for_eval(
    gt_json: str,
    limit: Optional[int],
    official_eval_dir: str,
    subset: str = "full",
) -> str:
    zip_active = subset not in (None, "full")
    if limit is None and not zip_active:
        return _abspath(gt_json)
    samples = load_gt_samples(gt_json, limit)
    if zip_active:
        from . import zip_subset

        pages = zip_subset.union_pages(zip_subset.load_specs(subset))
        samples = zip_subset.filter_gt_samples(samples, pages)
        subset_path = os.path.join(official_eval_dir, f"gt_zip_{subset}.json")
    else:
        subset_path = os.path.join(official_eval_dir, f"gt_limit_{limit}.json")
    _dump_json(samples, subset_path)
    return subset_path


def _make_end2end_config(
    gt_path: str,
    pred_md_dir: str,
    formula_metric: str,
    match_method: str,
) -> Dict[str, Any]:
    return {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": ["Edit_dist", formula_metric]},
                "table": {"metric": ["TEDS", "Edit_dist"]},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": gt_path},
                "prediction": {"data_path": pred_md_dir},
                "match_method": match_method,
            },
        }
    }


def _write_yaml_config(config: Dict[str, Any], path: str) -> None:
    import yaml

    os.makedirs(os.path.dirname(_abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def _docker_volume(host_path: str, container_path: str, readonly: bool = False) -> List[str]:
    spec = f"{_abspath(host_path)}:{container_path}"
    if readonly:
        spec += ":ro"
    return ["-v", spec]


def _docker_eval_command(
    args: argparse.Namespace,
    official_repo_dir: str,
    official_eval_dir: str,
    gt_json_for_eval: str,
    pred_md_dir: str,
    config_path: str,
) -> List[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    cmd += _docker_volume(official_repo_dir, "/workspace/OmniDocBench", readonly=True)
    cmd += _docker_volume(official_eval_dir, "/workspace/official_eval", readonly=False)
    cmd += _docker_volume(pred_md_dir, "/workspace/pred_md", readonly=True)
    if os.path.abspath(gt_json_for_eval).startswith(os.path.abspath(official_eval_dir) + os.sep):
        container_gt = "/workspace/official_eval/" + os.path.relpath(gt_json_for_eval, official_eval_dir)
    else:
        cmd += _docker_volume(gt_json_for_eval, "/workspace/input/OmniDocBench.json", readonly=True)
        container_gt = "/workspace/input/OmniDocBench.json"

    # The config path is mounted through official_eval_dir.
    container_config = "/workspace/official_eval/" + os.path.relpath(config_path, official_eval_dir)
    # The image ships a conda env "OmniDocBench" (py3.10) and CDM toolchain
    # (nodejs/imagemagick/texlive) under /root/cdm_lib. The default `python`
    # in PATH is /usr/bin/python (2.7) so we explicitly activate the env and
    # prepend CDM tool paths before invoking pdf_validation.py.
    # The image's conda + texlive expect root, so we run as root then chown
    # outputs back to the host UID/GID to keep the result tree user-writable
    # (otherwise rmtree on the next run hits PermissionError).
    host_uid = os.getuid()
    host_gid = os.getgid()
    inner_cmd = (
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate OmniDocBench && "
        "export PATH=/root/cdm_lib/nodejs/bin:/root/cdm_lib/magick/bin:"
        "/root/cdm_lib/texlive/bin/x86_64-linux:$PATH && "
        "export LD_LIBRARY_PATH=/root/cdm_lib/magick/lib/:${LD_LIBRARY_PATH:-} && "
        f"{{ python /workspace/OmniDocBench/pdf_validation.py --config {container_config}; rc=$?; "
        f"chown -R {host_uid}:{host_gid} /workspace/official_eval; exit $rc; }}"
    )
    cmd += [
        "-w",
        "/workspace/official_eval",
        args.docker_image,
        "bash", "-c", inner_cmd,
    ]

    # Store these resolved container paths so the generated YAML matches the command.
    args._container_gt_path = container_gt
    args._container_pred_md_dir = "/workspace/pred_md"
    return cmd


def _run_command(cmd: List[str], cwd: str, env: Optional[Dict[str, str]] = None) -> None:
    print("Running:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited with code {result.returncode}")


def _find_metric_result(result_dir: str, match_method: str) -> str:
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Official result directory not found: {result_dir}")
    suffix = f"_{match_method}_metric_result.json"
    candidates = [
        os.path.join(result_dir, name)
        for name in os.listdir(result_dir)
        if name.endswith(suffix) or name.endswith("_metric_result.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No official metric_result.json found in {result_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _get_nested(data: Dict[str, Any], path: Sequence[str]) -> Optional[float]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)):
        if isinstance(cur, float) and (math.isnan(cur) or math.isinf(cur)):
            return None
        return float(cur)
    return None


def _first_present(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def _summarize_scores(
    metric_json_path: str,
    formula_metric: str,
    score_json_path: str,
    score_csv_path: str,
) -> Dict[str, Any]:
    metric_result = _load_json(metric_json_path)

    text_edit = _first_present(
        _get_nested(metric_result, ["text_block", "all", "Edit_dist", "ALL_page_avg"]),
        _get_nested(metric_result, ["text_block", "page", "Edit_dist", "ALL"]),
    )
    table_teds = _first_present(
        _get_nested(metric_result, ["table", "all", "TEDS", "all"]),
        _get_nested(metric_result, ["table", "page", "TEDS", "ALL"]),
    )
    formula_cdm = _first_present(
        _get_nested(metric_result, ["display_formula", "all", "CDM", "all"]),
        _get_nested(metric_result, ["display_formula", "page", "CDM", "ALL"]),
    )
    formula_edit = _first_present(
        _get_nested(metric_result, ["display_formula", "all", "Edit_dist", "ALL_page_avg"]),
        _get_nested(metric_result, ["display_formula", "page", "Edit_dist", "ALL"]),
    )
    table_edit = _first_present(
        _get_nested(metric_result, ["table", "all", "Edit_dist", "ALL_page_avg"]),
        _get_nested(metric_result, ["table", "page", "Edit_dist", "ALL"]),
    )
    reading_order_edit = _first_present(
        _get_nested(metric_result, ["reading_order", "all", "Edit_dist", "ALL_page_avg"]),
        _get_nested(metric_result, ["reading_order", "page", "Edit_dist", "ALL"]),
    )

    complete = formula_metric == "CDM" and all(
        value is not None for value in (text_edit, table_teds, formula_cdm)
    )
    if formula_metric == "CDM" and not complete:
        raise RuntimeError(
            "Official eval completed but CDM leaderboard metrics were missing. "
            "Check the Docker CDM runtime and official metric_result.json."
        )

    overall = None
    if complete:
        overall = ((1.0 - text_edit) * 100.0 + table_teds * 100.0 + formula_cdm * 100.0) / 3.0

    summary = {
        "complete": complete,
        "formula_metric": formula_metric,
        "overall": overall,
        "metrics": {
            "text_edit": text_edit,
            "table_teds": table_teds,
            "formula_cdm": formula_cdm,
            "formula_edit": formula_edit,
            "table_edit": table_edit,
            "reading_order_edit": reading_order_edit,
        },
        "official_metric_result": metric_json_path,
    }
    if formula_metric == "CDM_plain":
        summary["note"] = (
            "Incomplete leaderboard score: CDM_plain only exports CDM matching pairs "
            "and does not compute Formula CDM or Overall."
        )

    rows = [
        {"Metric": "Overall", "Value": overall, "Complete": complete},
        {"Metric": "Text Edit_dist", "Value": text_edit, "Complete": text_edit is not None},
        {"Metric": "Table TEDS", "Value": table_teds, "Complete": table_teds is not None},
        {"Metric": "Formula CDM", "Value": formula_cdm, "Complete": formula_cdm is not None},
        {"Metric": "Formula Edit_dist", "Value": formula_edit, "Complete": formula_edit is not None},
        {"Metric": "Table Edit_dist", "Value": table_edit, "Complete": table_edit is not None},
        {
            "Metric": "Reading Order Edit_dist",
            "Value": reading_order_edit,
            "Complete": reading_order_edit is not None,
        },
    ]
    _dump_json(summary, score_json_path)
    _write_csv_rows(rows, score_csv_path)
    return summary


def evaluate_omnidocbench(args: argparse.Namespace) -> None:
    args.work_dir = _abspath(args.work_dir)
    out_dir = _output_dir(args)
    official_eval_dir = _ensure_dir(os.path.join(out_dir, "official_eval"))
    result_dir = os.path.join(official_eval_dir, "result")

    gt_json = _abspath(args.gt_json)
    pred_md_dir = _abspath(args.pred_md_dir or os.path.join(out_dir, "pred_md"))
    if not os.path.isdir(pred_md_dir):
        raise FileNotFoundError(f"Prediction markdown directory not found: {pred_md_dir}")

    if not args.keep_official_output and os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    # cal_metric.py writes to ./result/<file>.json directly (no mkdir), so
    # the directory must exist before invoking pdf_validation.py.
    os.makedirs(result_dir, exist_ok=True)

    official_repo_dir = ensure_official_repo(
        args.work_dir,
        args.official_dir,
        args.official_repo,
        args.official_ref,
        fatal=True,
    )
    assert official_repo_dir is not None

    gt_json_for_eval = _subset_gt_for_eval(
        gt_json, args.limit, official_eval_dir, getattr(args, "subset", "full"))

    host_config_path = os.path.join(official_eval_dir, "end2end.yaml")
    if args.eval_backend == "docker":
        docker_cmd = _docker_eval_command(
            args,
            official_repo_dir,
            official_eval_dir,
            gt_json_for_eval,
            pred_md_dir,
            host_config_path,
        )
        config = _make_end2end_config(
            args._container_gt_path,
            args._container_pred_md_dir,
            args.formula_metric,
            args.match_method,
        )
        _write_yaml_config(config, host_config_path)
        print(f"Wrote Docker official config to {host_config_path}")
        if args.dry_run:
            print("Dry run: not running Docker eval")
            print(" ".join(docker_cmd))
            return
        _run_command(docker_cmd, cwd=official_eval_dir)
    else:
        config = _make_end2end_config(
            gt_json_for_eval,
            pred_md_dir,
            args.formula_metric,
            args.match_method,
        )
        _write_yaml_config(config, host_config_path)
        print(f"Wrote local official config to {host_config_path}")
        cmd = [
            sys.executable,
            os.path.join(official_repo_dir, "pdf_validation.py"),
            "--config",
            host_config_path,
        ]
        if args.dry_run:
            print("Dry run: not running local eval")
            print(" ".join(cmd))
            return
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = official_repo_dir + os.pathsep + env.get("PYTHONPATH", "")
        _run_command(cmd, cwd=official_eval_dir, env=env)

    metric_json_path = _find_metric_result(result_dir, args.match_method)
    score_json_path = args.output_score_json or os.path.join(out_dir, "score.json")
    score_csv_path = args.output_score_csv or os.path.join(out_dir, "score.csv")
    subset = getattr(args, "subset", "full")
    if subset not in (None, "full"):
        # The unweighted official summary of the subset run must not sit in the
        # score.json slot -- score.json/score.csv carry the weighted estimate.
        unweighted_json = os.path.join(out_dir, "score_unweighted.json")
        unweighted_csv = os.path.join(out_dir, "score_unweighted.csv")
        summary = _summarize_scores(
            metric_json_path, args.formula_metric, unweighted_json, unweighted_csv)
        from . import zip_subset

        zip_score = zip_subset.zip_summarize(
            result_dir, out_dir, subset, summary, score_json_path, score_csv_path)
        print(f"Official result directory: {result_dir}")
        print(f"Official metric JSON: {metric_json_path}")
        print(f"Saved unweighted subset summary to {unweighted_json}")
        print(f"Saved weighted score JSON to {score_json_path}")
        print(f"Saved zipbench score to {os.path.join(out_dir, 'zip_score.json')}")
        if zip_score.get("score") is not None:
            print(f"Overall (weighted, subset={subset}): {zip_score['score']:.4f}")
        else:
            print("Overall (weighted): incomplete")
        return

    summary = _summarize_scores(
        metric_json_path,
        args.formula_metric,
        score_json_path,
        score_csv_path,
    )

    print(f"Official result directory: {result_dir}")
    print(f"Official metric JSON: {metric_json_path}")
    print(f"Saved score JSON to {score_json_path}")
    print(f"Saved score CSV to {score_csv_path}")
    if summary.get("overall") is not None:
        print(f"Overall: {summary['overall']:.4f}")
    else:
        print("Overall: incomplete")
