"""Compress a benchmark to a fixed ratio using a trained IRT model.

Given a trained IRT model and the train pkl, this script selects anchor items
via per-subscenario K-means clustering on the IRT item parameters and writes
the anchor (compressed subset) pickle. Optionally, when a test pkl is
provided, it evaluates the compressed subset against the full benchmark
(MAE / RMSE / Spearman rank correlation / per-model errors) and writes an
evaluation JSON next to each anchor file.

Recommended compression ratios: 0.5 and 0.75 (ratio = fraction of items
removed; e.g. ratio 0.75 keeps ~25% of the items as anchors).

Example:
    python compress_dataset.py \
        --scenario my_bench --subscenarios my_bench \
        --train-data path/to/my_bench_train.pkl \
        --model-path path/to/irt_model_run1/ \
        --test-data path/to/my_bench_test.pkl \
        --ratios 0.5 0.75 \
        --output-dir ./compressed/my_bench
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from evaluation_utils import evaluate_split
from sample_clustering import run_clustering

DEFAULT_RATIOS = [0.5, 0.75]
DEFAULT_SEED = 42


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _evaluate_anchor(test_data, scenarios, anchor_entry):
    indices = np.asarray(anchor_entry["indices"], dtype=int)
    weights = np.asarray(anchor_entry["weights"], dtype=float)
    result = evaluate_split(
        data=test_data,
        scenarios=scenarios,
        anchor_indices=indices,
        weights=weights,
    )
    full = result["full_accuracies"]
    subset = result["subset_accuracies"]
    errors = {m: subset[m] - full[m] for m in result["model_names"]}
    abs_errors = np.array([abs(e) for e in errors.values()])
    result["per_model_errors"] = errors
    result["mae"] = float(abs_errors.mean())
    result["rmse"] = float(np.sqrt(np.mean(abs_errors**2)))
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select anchor items at fixed compression ratios from a trained "
            "IRT model and optionally evaluate the compressed subset."
        )
    )
    parser.add_argument("--scenario", required=True,
                        help="Scenario key present in the pkl data dict.")
    parser.add_argument("--subscenarios", nargs="+", default=None,
                        help="Data keys under the scenario; defaults to the scenario itself.")
    parser.add_argument("--train-data", required=True,
                        help="Path to the train pkl (defines item positions for clustering).")
    parser.add_argument("--model-path", required=True,
                        help="Directory of the trained IRT model (e.g. .../irt_model_run1/).")
    parser.add_argument("--test-data", default=None,
                        help="Optional test pkl; when given, the subset is evaluated "
                             "against the full benchmark and an eval JSON is written.")
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                        help="Compression ratios (fraction of items removed). "
                             f"Recommended: 0.5 and 0.75. Default: {DEFAULT_RATIOS}.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random state for K-means clustering. Default: {DEFAULT_SEED}.")
    parser.add_argument("--output-dir", default="./compressed",
                        help="Directory for anchor pkl / eval JSON outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    scenario = args.scenario
    subscenarios = args.subscenarios or [scenario]
    scenarios = {scenario: tuple(subscenarios)}
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_data = _load_pickle(args.test_data) if args.test_data else None

    for ratio in args.ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1); got {ratio}")
        tag = f"{ratio:g}".replace(".", "p")
        anchor_path = out_dir / f"{scenario}_anchor_ratio_{tag}.pkl"

        print(f"\n=== ratio {ratio} (seed={args.seed}) ===")
        summary = run_clustering(
            scenario=scenario,
            subscenarios=subscenarios,
            data_path=args.train_data,
            model_path=args.model_path,
            anchor_path=str(anchor_path),
            ratio=ratio,
            random_state=args.seed,
        )

        if test_data is None:
            continue

        anchor_entry = _load_pickle(anchor_path)[scenario]
        result = _evaluate_anchor(test_data, scenarios, anchor_entry)
        result["ratio"] = ratio
        result["seed"] = args.seed
        result["anchor_path"] = str(anchor_path)
        result["n_items_total"] = summary["n_items_total"]
        result["n_anchor_items"] = summary["n_anchor_items"]

        eval_path = out_dir / f"{scenario}_eval_ratio_{tag}.json"
        with eval_path.open("w") as handle:
            json.dump(result, handle, indent=2)

        print(
            f"subset {result['num_items_subset']}/{result['num_items_full']} items, "
            f"{result['num_models_evaluated']} models | "
            f"MAE={result['mae']:.5f} RMSE={result['rmse']:.5f} "
            f"Spearman={result['spearman']['coeff']:.4f} "
            f"(p={result['spearman']['pvalue']:.2e})"
        )
        print(f"Evaluation written to {eval_path}")


if __name__ == "__main__":
    main()
