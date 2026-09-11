import argparse
import glob
import json
import os
import os.path as osp
from typing import Dict, Iterable, List, Tuple

from evalplus.evaluate import evaluate

from opencompass.datasets.mbpp import MBPPEvaluator


def _sorted_task_ids(task_ids: Iterable[str]) -> List[str]:
    def sort_key(task_id: str) -> Tuple[str, int]:
        if "/" in task_id:
            prefix, suffix = task_id.rsplit("/", 1)
            if suffix.isdigit():
                return (prefix, int(suffix))
        return (task_id, -1)

    return sorted(task_ids, key=sort_key)


def _load_predictions(pred_dir: str,
                      pattern: str) -> Iterable[Dict[str, str]]:
    pred_files = sorted(glob.glob(osp.join(pred_dir, pattern)))
    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files matched: {osp.join(pred_dir, pattern)}")

    for path in pred_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        keys = list(data.keys())
        try:
            keys.sort(key=lambda x: int(x))
        except ValueError:
            keys.sort()
        for key in keys:
            yield data[key]


def build_samples_jsonl(pred_dir: str,
                        output_jsonl: str,
                        pattern: str) -> None:
    evaluator = MBPPEvaluator(metric="MBPPPlus")
    os.makedirs(osp.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in _load_predictions(pred_dir, pattern):
            task_id = item.get("gold") or item.get("task_id")
            if not task_id:
                raise KeyError(
                    "Missing task_id/gold in prediction entry; "
                    "cannot build evalplus samples."
                )
            solution = evaluator._process_answer(item["prediction"])
            record = {"task_id": task_id, "solution": solution}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_evalplus(samples_path: str) -> str:
    flags = dict(
        dataset="mbpp",
        samples=samples_path,
        base_only=None,
        parallel=None,
        i_just_wanna_run=None,
        test_details=True,
        min_time_limit=0.2,
        gt_time_limit_factor=4.0,
        mini=None,
    )
    evaluate(flags)
    return samples_path.replace(".jsonl", "_eval_results.json")


def export_per_item(eval_results_path: str, output_jsonl: str) -> None:
    with open(eval_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    items = []
    for task_id in _sorted_task_ids(results["eval"].keys()):
        task_res = results["eval"][task_id]
        base_list = task_res.get("base", [])
        plus_list = task_res.get("plus", [])

        def _is_success(idx: int) -> bool:
            if idx >= len(base_list):
                return False
            base_ok = base_list[idx][0] == "success"
            if not plus_list:
                return base_ok
            if idx >= len(plus_list):
                return False
            return base_ok and (plus_list[idx][0] == "success")

        any_correct = any(_is_success(i) for i in range(task_res["nfiles"]))
        first_base = base_list[0] if base_list else None
        first_plus = plus_list[0] if plus_list else None

        items.append({
            "task_id": task_id,
            "any_correct": any_correct,
            "base_status_first": first_base[0] if first_base else None,
            "plus_status_first": first_plus[0] if first_plus else None,
            "base_details_first": first_base[1] if first_base else None,
            "plus_details_first": first_plus[1] if first_plus else None,
        })

    os.makedirs(osp.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-item MBPPPlus correctness with evalplus.")
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Prediction directory, e.g. outputs/.../predictions/<model>")
    parser.add_argument(
        "--pattern",
        default="mbpp_plus_*.json",
        help="Prediction file glob pattern.")
    parser.add_argument(
        "--out-dir",
        required=False,
        default=None,
        help="Directory to save evalplus samples and results. "
             "If not provided, will be auto-generated from pred-dir.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-generate out_dir from pred_dir if not provided
    if args.out_dir is None:
        # Get parent's parent directory of pred_dir, then append mbpp_plus_details
        pred_parent_parent = osp.dirname(osp.dirname(args.pred_dir))
        args.out_dir = osp.join(pred_parent_parent, "mbpp_plus_details")

    samples_path = osp.join(args.out_dir, "mbpp_plus_samples.jsonl")
    eval_results_path = samples_path.replace(".jsonl", "_eval_results.json")
    per_item_path = osp.join(args.out_dir, "mbpp_plus_per_item.jsonl")

    build_samples_jsonl(args.pred_dir, samples_path, args.pattern)
    eval_results_path = run_evalplus(samples_path)
    export_per_item(eval_results_path, per_item_path)

    print(f"evalplus results: {eval_results_path}")
    print(f"per-item correctness: {per_item_path}")


if __name__ == "__main__":
    main()
