import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Compatibility for numpy 2.x-created pickles loaded in numpy 1.x envs.
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

from evaluation_utils import evaluate_split


DEFAULT_OUTPUT_JSON = Path("./evaluation_results.json")


def _parse_test_splits(raw_tests: Iterable[str]) -> Dict[str, Path]:
    """Parse repeated --test-data arguments of the form name=/abs/path."""

    parsed: Dict[str, Path] = {}
    for entry in raw_tests:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --test-data value '{entry}'. Use the form split_name=/absolute/path/to/file.pkl."
            )
        name, raw_path = entry.split("=", maxsplit=1)
        name = name.strip()
        if not name:
            raise ValueError(f"Split name missing in --test-data argument '{entry}'.")
        if name in parsed:
            raise ValueError(f"Duplicate split name '{name}' detected in --test-data arguments.")
        path = Path(raw_path).expanduser()
        parsed[name] = path
    return parsed


def _load_pickle(path: Path) -> Dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _evaluate_split(
    data: Dict,
    scenarios: Dict[str, Tuple[str, ...]],
    anchor_indices: np.ndarray,
    model_filter: Optional[Iterable[str]] = None,
    weights: Optional[np.ndarray] = None,
    aggregation_mode: str = "item",
    prediction_method: str = "weighted",
    irt_params: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    train_responses: Optional[np.ndarray] = None,
    subscenario_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    return evaluate_split(
        data,
        scenarios,
        anchor_indices,
        model_filter=model_filter,
        weights=weights,
        aggregation_mode=aggregation_mode,
        prediction_method=prediction_method,
        irt_params=irt_params,
        train_responses=train_responses,
        subscenario_weights=subscenario_weights,
    )


def _load_model_splits(
    split_path: Path, available_models: Iterable[str]
) -> Optional[List[str]]:
    with split_path.open("r") as f:
        payload = json.load(f)

    model_list = list(available_models)

    if "train_model_ids" in payload and payload["train_model_ids"]:
        return [str(name) for name in payload["train_model_ids"]]
    if "train_idx" in payload and payload["train_idx"]:
        return [str(model_list[int(idx)]) for idx in payload["train_idx"]]

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Spearman correlation between full and anchor-based model rankings "
            "for training and test splits."
        )
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario key in the anchor file and dataset.",
    )
    parser.add_argument(
        "--subscenarios",
        nargs="+",
        default=None,
        help=(
            "Subscenario keys to use for the scenario. Defaults to the order found in the training dataset."
        ),
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to the training dataset pickle file.",
    )
    parser.add_argument(
        "--test-data",
        action="append",
        default=[],
        help=(
            "Absolute path(s) to test dataset pickle files in the form name=/abs/path. "
            "Repeat the flag for multiple test splits."
        ),
    )
    parser.add_argument(
        "--anchor-file",
        required=True,
        help="Path to the anchor pickle file.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help=(
            "Optional path to a JSON file containing 'train_model_ids' / 'train_idx' for automatic subset selection."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--output-pkl",
        default=None,
        help="Optional path to also persist results as a pickle file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_data_path = Path(args.train_data).expanduser()
    test_data_paths = _parse_test_splits(args.test_data)
    anchor_path = Path(args.anchor_file).expanduser()
    split_file_path: Optional[Path] = None
    train_model_filter: Optional[List[str]] = None

    data_train = _load_pickle(train_data_path)

    if args.subscenarios is None:
        if "data" in data_train:
            subscenarios = tuple(data_train["data"].keys())  # Preserve original order
        else:
            subscenarios = (args.scenario,)
    else:
        subscenarios = tuple(args.subscenarios)

    scenarios = {args.scenario: subscenarios}

    anchor_points = _load_pickle(anchor_path)
    if args.scenario not in anchor_points:
        raise KeyError(
            f"Scenario '{args.scenario}' not found in anchor file '{anchor_path}'."
        )
    _raw = anchor_points[args.scenario]
    anchor_indices = np.array(
        _raw["indices"] if isinstance(_raw, dict) else _raw, dtype=int
    )

    if args.split_file:
        candidate_split_path = Path(args.split_file).expanduser()
        if candidate_split_path.exists():
            train_model_filter = _load_model_splits(
                candidate_split_path, data_train["models"]
            )
            split_file_path = candidate_split_path
            print(
                f"Loaded training model list from {candidate_split_path}:"
                f" train={len(train_model_filter or [])}"
            )
        else:
            print(
                f"Warning: split file {candidate_split_path} not found. "
                "Proceeding without automatic train/test model filtering."
            )

    results = {
        "scenario": args.scenario,
        "subscenarios": list(subscenarios),
        "anchor_file": str(anchor_path),
        "split_file": str(split_file_path) if split_file_path else None,
        "model_filters": {
            "train": train_model_filter,
        },
        "splits": {},
    }

    print(f"Evaluating Spearman correlation for training split using {train_data_path}...")
    split_result = _evaluate_split(
        data_train, scenarios, anchor_indices, model_filter=train_model_filter
    )
    results["splits"]["train"] = {
        "data_file": str(train_data_path),
        **split_result,
    }
    train_spearman = split_result["spearman"]
    print(
        f"[train] Spearman Rank Correlation: {train_spearman['coeff']:.4f} "
        f"(p-value: {train_spearman['pvalue']:.4f})"
    )

    for split_name, split_path in test_data_paths.items():
        print(
            f"Evaluating Spearman correlation for test split '{split_name}' using {split_path}..."
        )
        data_test = _load_pickle(split_path)
        split_result = _evaluate_split(
            data_test,
            scenarios,
            anchor_indices,
        )
        results["splits"][split_name] = {
            "data_file": str(split_path),
            **split_result,
        }
        sp = split_result["spearman"]
        print(
            f"[{split_name}] Spearman Rank Correlation: {sp['coeff']:.4f} "
            f"(p-value: {sp['pvalue']:.4f})"
        )

    output_json_path = Path(args.output_json).expanduser()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w") as f:
        json.dump(results, f, indent=4)

    if args.output_pkl:
        output_pkl_path = Path(args.output_pkl).expanduser()
        output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with output_pkl_path.open("wb") as f:
            pickle.dump(results, f)

    print(f"\nAll results saved to {output_json_path}")
    if args.output_pkl:
        print(f"Results pickle saved to {output_pkl_path}")


if __name__ == "__main__":
    main()
