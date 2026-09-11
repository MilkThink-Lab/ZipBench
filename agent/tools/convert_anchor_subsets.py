#!/usr/bin/env python3
"""Convert internal anchor-subset JSON exports into the packaged subset data.

Reads the ``<dataset>_{small,tiny}_anchor_subset.json`` files produced by the
ZipBench compression pipeline and writes, per dataset::

    zipbench_agent/subsets_data/<dataset>/
        manifest.json   # metadata: sizes, metric convention, upstream pin
        small.jsonl     # one {"question_id", "weight"} per line, anchor order
        tiny.jsonl

Internal filesystem paths from the source files are stripped; only the anchor
pkl basename is kept for provenance.

Usage:
    python convert_anchor_subsets.py --src <dir-with-anchor-subset-jsons> \
        [--dst <repo>/agent/zipbench_agent/subsets_data]
"""

import argparse
import json
import re
from pathlib import Path

DATASETS = {
    "swe_bench_verified": {
        "display_name": "SWE-bench Verified",
        "id_kind": "instance_id, e.g. 'astropy__astropy-12907'",
        "official_metric": "resolved rate",
        "metric_note": (
            "s_i = 1 if the instance is resolved (member of resolved_ids in the "
            "swebench harness report), else 0."
        ),
        "upstream": {
            "harness": "swebench >= 4.1.0 (python -m swebench.harness.run_evaluation)",
            "dataset": "SWE-bench/SWE-bench_Verified (test split, 500 instances)",
            "task_filter_flag": "-i/--instance_ids",
        },
    },
    "swe_bench_multilingual": {
        "display_name": "SWE-bench Multilingual",
        "id_kind": "instance_id, e.g. 'apache__druid-13704'",
        "official_metric": "resolved rate",
        "metric_note": (
            "s_i = 1 if the instance is resolved (member of resolved_ids in the "
            "swebench harness report), else 0."
        ),
        "upstream": {
            "harness": "swebench >= 4.1.0 (python -m swebench.harness.run_evaluation)",
            "dataset": "SWE-bench/SWE-bench_Multilingual (test split, 300 instances)",
            "task_filter_flag": "-i/--instance_ids",
        },
    },
    "terminal_bench": {
        "display_name": "Terminal-Bench 2.0",
        "id_kind": "task name in the terminal-bench@2.0 dataset, e.g. 'gpt2-codegolf'",
        "official_metric": "task pass rate",
        "metric_note": (
            "s_i = 1 if the trial reward >= 0.5, else 0 (ZipBench binarisation "
            "convention; harbor rewards for terminal-bench@2.0 are effectively "
            "0/1 already). With several attempts per task, s_i is the mean of "
            "the binarised attempt rewards."
        ),
        "upstream": {
            "harness": "harbor >= 0.4.0 (terminal-bench 2.x harness)",
            "dataset": "terminal-bench@2.0 (89 tasks)",
            "task_filter_flag": "-i/--include-task-name (repeatable)",
        },
    },
    "tau2_telecom": {
        "display_name": "τ³-bench telecom",
        "id_kind": (
            "task id from the telecom 'base' split, e.g. "
            "'[mms_issue]airplane_mode_on|...[PERSONA:None]'"
        ),
        "official_metric": "pass^1 (average task success)",
        "metric_note": (
            "s_i = 1 if reward_info.reward == 1.0 (tau2 success criterion), else "
            "0. With several trials per task, s_i is the mean success over "
            "trials; simulations that died on infrastructure errors are excluded."
        ),
        "upstream": {
            "harness": "tau2 >= 1.0.0 (v1.0.0 = the τ³-bench release; tau2 run --domain telecom)",
            "dataset": "telecom domain, task split 'base' (114 tasks, the CLI default; τ³ task set)",
            "task_filter_flag": "--task-ids",
        },
    },
    "tau2_retail": {
        "display_name": "τ³-bench retail",
        "id_kind": "numeric task id string '0'..'113'",
        "official_metric": "pass^1 (average task success)",
        "metric_note": (
            "s_i = 1 if reward_info.reward == 1.0 (tau2 success criterion), else "
            "0. With several trials per task, s_i is the mean success over "
            "trials; simulations that died on infrastructure errors are excluded."
        ),
        "upstream": {
            "harness": "tau2 >= 1.0.0 (v1.0.0 = the τ³-bench release; tau2 run --domain retail)",
            "dataset": "retail domain (114 tasks, τ³ task set)",
            "task_filter_flag": "--task-ids",
        },
    },
    "tau2_airline": {
        "display_name": "τ³-bench airline",
        "id_kind": "numeric task id string '0'..'49'",
        "official_metric": "pass^1 (average task success)",
        "metric_note": (
            "s_i = 1 if reward_info.reward == 1.0 (tau2 success criterion), else "
            "0. With several trials per task, s_i is the mean success over "
            "trials; simulations that died on infrastructure errors are excluded."
        ),
        "upstream": {
            "harness": "tau2 >= 1.0.0 (v1.0.0 = the τ³-bench release; tau2 run --domain airline)",
            "dataset": "airline domain (50 tasks, τ³ task set)",
            "task_filter_flag": "--task-ids",
        },
    },
    "swe_bench_pro": {
        "display_name": "SWE-Bench Pro (public set)",
        "id_kind": (
            "instance_id, e.g. 'instance_<org>__<repo>-<commit-sha>[-v<sha|nan>]'"
        ),
        "official_metric": "resolved rate",
        "metric_note": (
            "s_i = 1 if eval_results.json marks the instance true (all "
            "FAIL_TO_PASS and PASS_TO_PASS tests passed), else 0."
        ),
        "upstream": {
            "harness": (
                "swe_bench_pro_eval.py (scaleapi/SWE-bench_Pro-os @ ca10a60)"
            ),
            "dataset": "ScaleAI/SWE-bench_Pro public split (731 instances)",
            "task_filter_flag": (
                "none — the harness evaluates exactly the instances in the "
                "--patch_path predictions JSON; restrict it with "
                "'tasks swe_bench_pro --format ids'"
            ),
        },
    },
    "osworld_verified": {
        "display_name": "OSWorld-Verified",
        "id_kind": (
            "example uuid, e.g. '94d95f96-9699-4208-98ba-3c3119edf9c2'"
        ),
        "official_metric": "average score",
        "metric_note": (
            "s_i is the harness's raw reward from result.txt / "
            "all_result.json, kept as a float in [0, 1] (no binarisation); "
            "with several runs per task, s_i is the mean reward."
        ),
        "upstream": {
            "harness": (
                "OSWorld run.py + show_result.py (xlang-ai/OSWorld @ 84aee65, "
                "OSWorld-Verified)"
            ),
            "dataset": (
                "OSWorld-Verified, evaluation_examples/test_nogdrive.json "
                "(361 tasks = 369 minus the 8 Google Drive tasks)"
            ),
            "task_filter_flag": (
                "--test_all_meta_path <json> — generate the file with "
                "'tasks osworld_verified --format meta'"
            ),
        },
    },
    "toolathlon": {
        "src_name": "Toolathlon",
        "display_name": "Toolathlon",
        "id_kind": "task name in tasks/finalpool, e.g. 'course-schedule'",
        "official_metric": "task pass rate",
        "metric_note": (
            "s_i = 1 if the task's eval_res.json has pass == true, else 0 "
            "(0/1, no partial credit); with several runs per task, s_i is "
            "the mean over runs."
        ),
        "upstream": {
            "harness": "run_parallel.py (hkust-nlp/Toolathlon)",
            "dataset": (
                "original 108-task release (pre-Verified); correctness "
                "records from HF Toolathlon-Trajectories"
            ),
            "task_filter_flag": (
                "--task_list <txt> — one task name per line, i.e. the "
                "'tasks toolathlon --format ids' output"
            ),
        },
    },
}

WEIGHTING = "score = sum(w_i * s_i); weights normalised so sum(w_i) == 1"


def parse_mae(anchor_pkl_basename):
    """Extract the anchor MAE from the source pkl filename, if present."""
    m = re.search(r"mae_?([0-9.]+?)(?:_|\.pkl)", anchor_pkl_basename)
    return float(m.group(1)) if m else None


def write_osworld_domains(test_meta_path, src_dir, out_dir):
    """Derive task_domains.json ({uuid: domain}) from an OSWorld meta file.

    The official meta file (e.g. evaluation_examples/test_nogdrive.json or
    test_all.json) has shape {domain: [uuid, ...]}; it is inverted and pruned
    to the 361 Verified ids in <src>/osworld_verified_ordered_question_ids.json
    (the ids pruned from test_all.json are the 8 Google Drive tasks).
    """
    meta = json.loads(Path(test_meta_path).read_text())
    assert isinstance(meta, dict) and all(
        isinstance(v, list) for v in meta.values()), (
        f"{test_meta_path} is not an OSWorld meta file ({{domain: [uuid]}})")
    uuid_to_domain = {}
    for domain, uuids in meta.items():
        for uuid in uuids:
            assert uuid not in uuid_to_domain, f"duplicate uuid {uuid}"
            uuid_to_domain[uuid] = domain

    ordered = json.loads(
        (src_dir / "osworld_verified_ordered_question_ids.json").read_text())
    missing = [u for u in ordered if u not in uuid_to_domain]
    assert not missing, f"{len(missing)} Verified ids missing from meta: {missing[:5]}"
    pruned = sorted(set(uuid_to_domain) - set(ordered))
    assert len(pruned) in (0, 8), (
        f"expected to prune exactly the 8 Google Drive tasks, got "
        f"{len(pruned)}: {pruned}")
    if pruned:
        print(f"osworld_verified: pruned {len(pruned)} Google Drive tasks: "
              f"{', '.join(pruned)}")
    domains = {u: uuid_to_domain[u] for u in ordered}
    (out_dir / "task_domains.json").write_text(
        json.dumps(domains, indent=2, ensure_ascii=False) + "\n")
    print(f"osworld_verified: task_domains.json covers {len(domains)} ids")


def convert(src_dir, dst_dir, datasets, osworld_test_meta=None):
    if "osworld_verified" in datasets and osworld_test_meta is None:
        raise SystemExit(
            "error: converting osworld_verified requires --osworld-test-meta "
            "<path to OSWorld evaluation_examples/test_nogdrive.json>")
    for dataset in datasets:
        meta = DATASETS[dataset]
        src_name = meta.get("src_name", dataset)
        out_dir = dst_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset": dataset,
            "display_name": meta["display_name"],
            "n_total_questions": None,
            "id_kind": meta["id_kind"],
            "official_metric": meta["official_metric"],
            "metric_note": meta["metric_note"],
            "weighting": WEIGHTING,
            "upstream": meta["upstream"],
            "subsets": {},
        }
        for size in ("small", "tiny"):
            src = src_dir / f"{src_name}_{size}_anchor_subset.json"
            data = json.loads(src.read_text())
            assert data["dataset"] == src_name and data["size"] == size, src
            n_total = data["n_total_questions"]
            if manifest["n_total_questions"] is None:
                manifest["n_total_questions"] = n_total
            assert manifest["n_total_questions"] == n_total, src

            anchors = data["anchors"]
            assert len(anchors) == data["n_anchors"], src
            weight_sum = sum(a["weight"] for a in anchors)
            assert abs(weight_sum - 1.0) < 1e-6, (src, weight_sum)
            ids = [a["question_id"] for a in anchors]
            assert len(set(ids)) == len(ids), f"duplicate question_id in {src}"

            with open(out_dir / f"{size}.jsonl", "w") as f:
                for a in anchors:
                    f.write(json.dumps(
                        {"question_id": a["question_id"], "weight": a["weight"]},
                        ensure_ascii=False) + "\n")

            manifest["subsets"][size] = {
                "file": f"{size}.jsonl",
                "n": data["n_anchors"],
                "ratio": round(1 - data["n_anchors"] / n_total, 4),
                "anchor_mae": parse_mae(Path(data["anchor_pkl"]).name),
                "weight_sum": weight_sum,
            }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        if dataset == "osworld_verified":
            write_osworld_domains(osworld_test_meta, src_dir, out_dir)
        sizes = ", ".join(
            f"{s}={v['n']}" for s, v in manifest["subsets"].items())
        print(f"{dataset}: total={manifest['n_total_questions']} {sizes}")


def select_datasets(src_dir, requested):
    """Datasets to convert: the explicit list, or those whose sources exist."""
    if requested:
        for dataset in requested:
            src_name = DATASETS[dataset].get("src_name", dataset)
            for size in ("small", "tiny"):
                src = src_dir / f"{src_name}_{size}_anchor_subset.json"
                if not src.is_file():
                    raise SystemExit(f"error: missing source file {src}")
        return list(requested)
    present, skipped = [], []
    for dataset, meta in DATASETS.items():
        src_name = meta.get("src_name", dataset)
        if all((src_dir / f"{src_name}_{size}_anchor_subset.json").is_file()
               for size in ("small", "tiny")):
            present.append(dataset)
        else:
            skipped.append(dataset)
    if not present:
        raise SystemExit(
            f"error: no <dataset>_<size>_anchor_subset.json sources found in "
            f"{src_dir}")
    for dataset in skipped:
        print(f"warning: skipping {dataset} (no source files in {src_dir})")
    return present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path,
                        help="directory containing <dataset>_<size>_anchor_subset.json files")
    parser.add_argument("--dst", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "zipbench_agent" / "subsets_data")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                        help="convert only these datasets (default: every "
                             "dataset whose source files exist in --src)")
    parser.add_argument("--osworld-test-meta", type=Path,
                        help="OSWorld meta file ({domain: [uuid]}, e.g. "
                             "evaluation_examples/test_nogdrive.json) used to "
                             "build osworld_verified/task_domains.json")
    args = parser.parse_args()
    datasets = select_datasets(args.src, args.datasets)
    convert(args.src, args.dst, datasets, args.osworld_test_meta)


if __name__ == "__main__":
    main()
