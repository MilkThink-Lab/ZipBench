#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic ensemble models from continuous scores (continuous counterpart of batch_vote_uniform_bins_results_json_multiqa.py).

Differences from the binary version:
- the per-item ensemble score is the mean of `details[qid].is_correct` (float, [0,1]) over the k models,
  instead of a majority vote (voting is undefined for continuous values);
- no tie-break, hence fully deterministic with no random seed;
- model accuracy = mean of the per-item scores.

Binning and selection follow the binary version:
- split the [min, max] of real model accuracies into --num-bins equal bins;
- for each bin pick the candidate whose accuracy is closest to the bin center;
- enumerate all k=2 combinations first; if bins remain empty, move on to k=3, ... until all bins are filled or k is exhausted.

The output directory layout matches the binary version:
  <output-dir>/generated/<name>/<dataset>.json
  <output-dir>/selected_list.txt
  <output-dir>/selected_summary.jsonl
"""

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def load_model_details(json_path: Path) -> Tuple[Dict[str, dict], float]:
    """Read a model's details (qid -> record) and overall accuracy (mean of per-item is_correct)."""
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    details = payload.get("details")
    if not isinstance(details, dict) or not details:
        raise ValueError(f"{json_path} has no valid details dict")
    scores = {}
    for qid, rec in details.items():
        if not isinstance(rec, dict) or "is_correct" not in rec:
            raise ValueError(f"record {qid} in {json_path} is missing is_correct")
        scores[str(qid)] = float(rec["is_correct"])
    acc = sum(scores.values()) / len(scores)
    return details, acc


def build_bins(m: int, acc_min: float, acc_max: float) -> List[Tuple[float, float]]:
    width = (acc_max - acc_min) / m
    return [(acc_min + i * width, acc_min + (i + 1) * width) for i in range(m)]


def find_bin_index(acc: float, bins: Sequence[Tuple[float, float]]) -> Optional[int]:
    for i, (low, high) in enumerate(bins):
        # The last bin is closed; the others are half-open [low, high) (same as the binary driver)
        if (low <= acc < high) or (i == len(bins) - 1 and low <= acc <= high):
            return i
    return None


def build_ensemble(
    member_names: Sequence[str],
    model_details: Dict[str, Dict[str, dict]],
    common_ids: Sequence[str],
) -> Tuple[Dict[str, dict], float]:
    """Average the members' continuous scores per item; returns (details, accuracy)."""
    out_details = {}
    total = 0.0
    for qid in common_ids:
        member_scores = [float(model_details[m][qid]["is_correct"]) for m in member_names]
        score = sum(member_scores) / len(member_scores)
        base = copy.deepcopy(model_details[member_names[0]][qid])
        base["is_correct"] = score
        base["any_correct"] = max(member_scores)
        base["ensemble_member_scores"] = {m: s for m, s in zip(member_names, member_scores)}
        base.pop("attempts", None)
        out_details[qid] = base
        total += score
    return out_details, total / len(common_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory of real models (each sub-directory holds <dataset>.json with continuous scores)")
    parser.add_argument("--dataset-name", default="osworld_verified")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-bins", type=int, default=6)
    parser.add_argument("--max-k", type=int, default=None, help="Maximum combination size; defaults to the number of models")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and (d / f"{args.dataset_name}.json").is_file())
    if len(model_dirs) < 2:
        raise SystemExit(f"Fewer than 2 valid models found under {input_dir}")

    model_details: Dict[str, Dict[str, dict]] = {}
    model_accs: Dict[str, float] = {}
    for d in model_dirs:
        details, acc = load_model_details(d / f"{args.dataset_name}.json")
        model_details[d.name] = {str(k): v for k, v in details.items()}
        model_accs[d.name] = acc
    names = sorted(model_accs, key=model_accs.get)
    print(f"[INFO] Loaded {len(names)} model JSON files.")
    for n in names:
        print(f"  - {n}: accuracy={model_accs[n]*100:.4f}")

    common_ids = set(model_details[names[0]])
    for n in names[1:]:
        common_ids &= set(model_details[n])
    common_ids = sorted(common_ids)
    print(f"[INFO] Common task ids: {len(common_ids)}")

    acc_min, acc_max = min(model_accs.values()), max(model_accs.values())
    bins = build_bins(args.num_bins, acc_min, acc_max)
    print(f"[INFO] Bins over [{acc_min:.6f}, {acc_max:.6f}]:")
    for i, (lo, hi) in enumerate(bins):
        print(f"  bin {i}: [{lo:.6f}, {hi:.6f})")

    max_k = args.max_k or len(names)
    # bin_idx -> (distance_to_center, candidate_info)
    best_per_bin: Dict[int, Tuple[float, dict]] = {}
    comb_counter = 0
    for k in range(2, max_k + 1):
        for combo in itertools.combinations(names, k):
            details, acc = build_ensemble(combo, model_details, common_ids)
            bin_idx = find_bin_index(acc, bins)
            comb_counter += 1
            if bin_idx is None:
                continue
            center = 0.5 * (bins[bin_idx][0] + bins[bin_idx][1])
            dist = abs(acc - center)
            if bin_idx not in best_per_bin or dist < best_per_bin[bin_idx][0]:
                best_per_bin[bin_idx] = (dist, {
                    "name": f"acc_{acc:.6f}_k_{k}_comb_{comb_counter}",
                    "accuracy": acc,
                    "k": k,
                    "members": list(combo),
                    "details": details,
                })
        if len(best_per_bin) == args.num_bins:
            break
    if len(best_per_bin) < args.num_bins:
        missing = [i for i in range(args.num_bins) if i not in best_per_bin]
        print(f"[WARN] Empty bins (no candidate landed): {missing}; emitting the {len(best_per_bin)} filled bins anyway")

    list_lines, summary_lines = [], []
    for bin_idx in sorted(best_per_bin):
        _, cand = best_per_bin[bin_idx]
        lo, hi = bins[bin_idx]
        model_dir = generated_dir / cand["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": args.dataset_name,
            "source": "generated_continuous_mean",
            "ensemble_members": cand["members"],
            "is_correct": cand["accuracy"],
            "accuracy": cand["accuracy"] * 100.0,
            "total": len(common_ids),
            "evaluated": len(common_ids),
            "missing": 0,
            "details": cand["details"],
        }
        out_json = model_dir / f"{args.dataset_name}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"[INFO] bin {bin_idx} [{lo:.6f},{hi:.6f}) -> {cand['name']} (acc={cand['accuracy']:.6f}, members={cand['members']})")
        list_lines.append(f"{bin_idx}\t{lo:.6f}\t{hi:.6f}\t{cand['accuracy']:.6f}\tgenerated/{cand['name']}/{args.dataset_name}.json")
        summary_lines.append(json.dumps({
            "bin_index": bin_idx,
            "bin_low": lo,
            "bin_high": hi,
            "accuracy": cand["accuracy"],
            "json_path": str(out_json),
            "source": "generated_continuous_mean",
            "members": cand["members"],
            "metric": "is_correct",
            "dataset": args.dataset_name,
        }, ensure_ascii=False))

    (output_dir / "selected_list.txt").write_text("\n".join(list_lines) + "\n", encoding="utf-8")
    (output_dir / "selected_summary.jsonl").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[INFO] Done. {len(best_per_bin)}/{args.num_bins} bins filled, outputs under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
