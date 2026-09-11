import argparse
import json
import os
import os.path as osp
from typing import Dict, Iterable, List, Optional, Tuple


def _load_any_correct_map(per_item_path: str) -> Dict[str, bool]:
    mapping: Dict[str, bool] = {}
    with open(per_item_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            if not task_id:
                raise KeyError("Missing task_id in per-item record.")
            mapping[task_id] = bool(record.get("any_correct"))
    if not mapping:
        raise ValueError(f"No records loaded from {per_item_path}")
    return mapping


def _normalize_reference(reference) -> Optional[str]:
    if isinstance(reference, str):
        return reference
    if isinstance(reference, list) and len(reference) == 1:
        if isinstance(reference[0], str):
            return reference[0]
    return None


def _iter_detail_entries(details: Dict) -> Iterable[Tuple[str, Dict]]:
    for key, value in details.items():
        if key == "type":
            continue
        if isinstance(value, dict):
            yield key, value


def _update_details(
    details: Dict,
    any_correct_map: Dict[str, bool],
) -> Tuple[int, List[Tuple[str, Optional[str]]]]:
    updated = 0
    missing: List[Tuple[str, Optional[str]]] = []
    for idx, entry in _iter_detail_entries(details):
        reference = _normalize_reference(entry.get("references"))
        if not reference:
            missing.append((idx, None))
            continue
        if reference not in any_correct_map:
            missing.append((idx, reference))
            continue
        entry["any_correct"] = any_correct_map[reference]
        updated += 1
    return updated, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill mbpp_plus.json with per-item any_correct flags."
    )
    parser.add_argument(
        "result_dir",
        help="Result directory containing mbpp_plus.json and mbpp_plus_details.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: overwrite mbpp_plus.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report stats, do not write output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = osp.abspath(args.result_dir)
    result_path = osp.join(result_dir, "mbpp_plus.json")
    per_item_path = osp.join(
        result_dir, "mbpp_plus_details", "mbpp_plus_per_item.jsonl"
    )

    if not osp.exists(result_path):
        raise FileNotFoundError(f"Missing result file: {result_path}")
    if not osp.exists(per_item_path):
        raise FileNotFoundError(f"Missing per-item file: {per_item_path}")

    any_correct_map = _load_any_correct_map(per_item_path)

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("details")
    if not isinstance(details, dict):
        raise KeyError("Missing or invalid 'details' in mbpp_plus.json")

    updated, missing = _update_details(details, any_correct_map)

    total = sum(1 for _ in _iter_detail_entries(details))
    if missing:
        print(f"Warning: {len(missing)} entries missing any_correct match.")
        for idx, ref in missing[:10]:
            print(f"  - details[{idx}] references={ref}")
        if len(missing) > 10:
            print("  - ...")

    print(
        f"Updated {updated}/{total} entries with any_correct from per-item file."
    )

    if args.dry_run:
        print("Dry-run mode: no file written.")
        return

    output_path = args.output or result_path
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print(f"Wrote updated results to: {output_path}")


if __name__ == "__main__":
    main()
