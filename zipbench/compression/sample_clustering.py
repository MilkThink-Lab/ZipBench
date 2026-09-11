import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

from utils import create_responses, prepare_data
from irt import load_irt_parameters


DEFAULT_RATIO = 0.95


def _normalize_model_path(model_path: str) -> str:
    """Ensure the provided model directory ends with a path separator."""
    expanded = Path(model_path).expanduser()
    if expanded.is_dir():
        return str(expanded) + os.sep
    return str(expanded)


def run_clustering(
    scenario: str,
    subscenarios: Sequence[str],
    data_path: str,
    model_path: str,
    anchor_path: str,
    ratio: float = DEFAULT_RATIO,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run K-means clustering on IRT parameters to extract anchor points."""
    data_file = Path(data_path).expanduser()
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    with data_file.open("rb") as handle:
        data = pickle.load(handle)

    scenario_key = scenario
    scenario_map = {scenario_key: list(subscenarios)}

    scenarios_position, subscenarios_position = prepare_data(scenario_map, data)
    Y = create_responses(scenario_map, data)

    if scenario_key not in scenarios_position:
        raise KeyError(f"Scenario '{scenario_key}' not present in dataset.")

    model_dir = _normalize_model_path(model_path)
    A, B, _ = load_irt_parameters(model_dir)
    X = np.vstack((A.squeeze(), B.squeeze().reshape((1, -1)))).T
    X = X[scenarios_position[scenario_key]]

    # --- Allocate cluster counts per sub-scenario ---
    sub_lengths = {
        sub: len(subscenarios_position[scenario_key][sub])
        for sub in scenario_map[scenario_key]
    }
    total_items = sum(sub_lengths.values())
    if total_items < 1:
        raise ValueError(f"No items found for scenario '{scenario_key}'.")
    N = total_items

    target_total = int(round((1.0 - ratio) * total_items))
    target_total = max(1, min(total_items, target_total))

    allocations = {}
    fractions = {}
    for sub, length in sub_lengths.items():
        raw_k = (1.0 - ratio) * length
        base_k = int(np.floor(raw_k))
        base_k = max(1, min(length, base_k))
        allocations[sub] = base_k
        fractions[sub] = raw_k - np.floor(raw_k)

    current_total = sum(allocations.values())
    delta = target_total - current_total

    # Adjust the allocation toward the global target total (largest fractional parts first)
    if delta > 0:
        for sub in sorted(fractions, key=fractions.get, reverse=True):
            if delta == 0:
                break
            if allocations[sub] < sub_lengths[sub]:
                allocations[sub] += 1
                delta -= 1
    elif delta < 0:
        for sub in sorted(fractions, key=fractions.get):
            if delta == 0:
                break
            if allocations[sub] > 1:
                allocations[sub] -= 1
                delta += 1

    if verbose:
        print(
            f"Target anchors (approx): {target_total}/{N} "
            f"(ratio={ratio}, subscenarios={len(sub_lengths)})"
        )
        for sub in scenario_map[scenario_key]:
            print(
                f"  - {sub}: items={sub_lengths[sub]}, k={allocations[sub]}"
            )

    # --- Cluster each sub-scenario independently ---
    anchor_points_by_sub = {}
    anchor_weights_by_sub = {}
    aggregate_indices = []
    aggregate_weights = []

    for sub in scenario_map[scenario_key]:
        indices = np.array(subscenarios_position[scenario_key][sub], dtype=int)
        if indices.size == 0:
            continue

        k_sub = allocations[sub]
        k_sub = max(1, min(indices.size, k_sub))

        sub_X = X[indices]
        sub_weights = np.ones(indices.shape[0], dtype=float)
        sub_weights /= sub_weights.sum()

        kmeans = KMeans(
            n_clusters=k_sub,
            n_init="auto",
            random_state=random_state,
        )
        kmeans.fit(sub_X, sample_weight=sub_weights)

        nearest = pairwise_distances(
            kmeans.cluster_centers_, sub_X, metric="euclidean"
        ).argmin(axis=1)
        anchor_idx = indices[nearest].astype(int)

        anchor_points_by_sub[sub] = anchor_idx
        cluster_weights = np.array(
            [
                np.sum(sub_weights[kmeans.labels_ == c])
                for c in range(k_sub)
            ]
        )
        anchor_weights_by_sub[sub] = cluster_weights

        aggregate_indices.append(anchor_idx)
        aggregate_weights.append(cluster_weights)

    if not aggregate_indices:
        raise ValueError(f"No anchors produced for scenario '{scenario_key}'.")

    anchor_all = np.concatenate(aggregate_indices).astype(int)
    weight_all = np.concatenate(aggregate_weights)
    number_item = anchor_all.size

    # Structured format: indices + weights + per-sub-scenario info
    anchor_points = {
        scenario_key: {
            "indices": anchor_all,
            "weights": weight_all,
            "by_sub": {
                sub: {
                    "indices": anchor_points_by_sub[sub],
                    "weights": anchor_weights_by_sub[sub],
                }
                for sub in anchor_points_by_sub
            },
        }
    }

    anchor_output = Path(anchor_path).expanduser()
    anchor_output.parent.mkdir(parents=True, exist_ok=True)
    with anchor_output.open("wb") as f:
        pickle.dump(anchor_points, f)

    if verbose:
        print(
            f"K-means clustering finished for scenario '{scenario_key}' "
            f"({number_item}/{N} items)."
        )
        print(f"Anchor points saved to {anchor_output}")

    return {
        "scenario": scenario_key,
        "n_items_total": int(N),
        "n_anchor_items": int(number_item),
        "anchor_path": str(anchor_output),
        "by_sub": {scenario_key: {k: int(v.size) for k, v in anchor_points_by_sub.items()}},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster IRT parameters to select anchor items per scenario."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario key present in the dataset.",
    )
    parser.add_argument(
        "--subscenarios",
        nargs="+",
        default=None,
        help="Ordered list of subscenario keys under the scenario; defaults to the scenario itself.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the aggregated data pickle.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Directory containing the trained IRT model.",
    )
    parser.add_argument(
        "--anchor-path",
        required=True,
        help="Where to write the resulting anchor pickle.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="Fraction of items to keep unselected (1 - ratio determines anchors).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional random state passed to KMeans.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_clustering(
        scenario=args.scenario,
        subscenarios=args.subscenarios or [args.scenario],
        data_path=args.data_path,
        model_path=args.model_path,
        anchor_path=args.anchor_path,
        ratio=args.ratio,
        random_state=args.random_state,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()