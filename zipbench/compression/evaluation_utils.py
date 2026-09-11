from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from irt import estimate_ability_parameters
from utils import create_responses, item_curve, prepare_data


VALID_AGGREGATION_MODES = ("item", "task_balanced", "weighted_task")
VALID_PREDICTION_METHODS = ("weighted", "pirt", "gpirt")


def _ranked_accuracies(model_names: np.ndarray, accuracies: np.ndarray) -> Tuple[Dict[str, float], List[Tuple[str, float]], Dict[str, int]]:
    accuracy_dict = {
        str(model_names[i]): float(accuracies[i]) for i in range(len(model_names))
    }
    ranked = sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True)
    ranks = {model: rank for rank, (model, _) in enumerate(ranked, 1)}
    return accuracy_dict, ranked, ranks


def _item_level_accuracies(
    Y: np.ndarray,
    anchor_indices: np.ndarray,
    weights: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    full_accuracies = np.mean(Y, axis=1)
    Y_subset = Y[:, anchor_indices]
    if weights is not None:
        subset_accuracies = np.average(Y_subset, axis=1, weights=weights)
    else:
        subset_accuracies = np.mean(Y_subset, axis=1)
    return full_accuracies, subset_accuracies


def _task_balanced_accuracies(
    Y: np.ndarray,
    subscenario_positions: Dict[str, Dict[str, List[int]]],
    scenario: str,
    subscenarios: Tuple[str, ...],
    anchor_indices: np.ndarray,
    weights: Optional[np.ndarray],
    subscenario_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    if subscenario_weights is not None:
        missing = [sub for sub in subscenarios if sub not in subscenario_weights]
        if missing:
            raise ValueError(f"subscenario_weights missing entries for: {missing}")
    if weights is not None and len(weights) != len(anchor_indices):
        raise ValueError(
            f"weights length ({len(weights)}) must match anchor_indices length ({len(anchor_indices)})"
        )

    anchor_indices = np.asarray(anchor_indices, dtype=int)
    anchor_position = {int(index): pos for pos, index in enumerate(anchor_indices.tolist())}
    full_parts = []
    subset_parts = []
    task_counts = {}

    for sub in subscenarios:
        full_indices = np.asarray(subscenario_positions[scenario][sub], dtype=int)
        if full_indices.size == 0:
            raise ValueError(f"Subscenario {sub!r} has no full items.")

        mask_positions = [
            anchor_position[int(index)]
            for index in full_indices.tolist()
            if int(index) in anchor_position
        ]
        if not mask_positions:
            raise ValueError(
                f"Subscenario {sub!r} has no selected anchor items; cannot use task_balanced aggregation."
            )

        sub_anchor_indices = anchor_indices[mask_positions]
        full_parts.append(np.mean(Y[:, full_indices], axis=1))

        Y_subset_sub = Y[:, sub_anchor_indices]
        if weights is not None:
            subset_parts.append(
                np.average(Y_subset_sub, axis=1, weights=np.asarray(weights)[mask_positions])
            )
        else:
            subset_parts.append(np.mean(Y_subset_sub, axis=1))

        task_counts[sub] = {
            "num_items_full": int(full_indices.size),
            "num_items_subset": int(len(mask_positions)),
        }
        if subscenario_weights is not None:
            task_counts[sub]["weight"] = float(subscenario_weights[sub])

    if subscenario_weights is not None:
        sub_weight_array = np.asarray([subscenario_weights[sub] for sub in subscenarios], dtype=float)
        return (
            np.average(np.vstack(full_parts), axis=0, weights=sub_weight_array),
            np.average(np.vstack(subset_parts), axis=0, weights=sub_weight_array),
            task_counts,
        )
    return (
        np.mean(np.vstack(full_parts), axis=0),
        np.mean(np.vstack(subset_parts), axis=0),
        task_counts,
    )


def _fit_thetas(Y: np.ndarray, seen: np.ndarray, A: np.ndarray, B: np.ndarray) -> List[np.ndarray]:
    """Fit each model's ability vector theta from its responses on the seen (anchor) items."""
    A_seen = A[:, :, seen]
    B_seen = B[:, :, seen]
    return [
        estimate_ability_parameters(Y[j, seen], A_seen, B_seen)
        for j in range(Y.shape[0])
    ]


def _pirt_accuracies(
    Y: np.ndarray,
    anchor_indices: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """tinyBenchmarks p-IRT: subset score = lambd*mean(seen) + (1-lambd)*mean(IRT-predicted unseen).

    lambd = n_seen / N. The data part is an unweighted mean over anchors (faithful to
    tinyBenchmarks; anchor cluster weights are intentionally not used here).
    """
    n_items = Y.shape[1]
    seen = np.asarray(anchor_indices, dtype=int)
    unseen = np.setdiff1d(np.arange(n_items), seen)
    full_accuracies = np.mean(Y, axis=1)

    data_part = np.mean(Y[:, seen], axis=1)
    if unseen.size == 0:
        return full_accuracies, data_part

    lambd = seen.size / n_items
    thetas = _fit_thetas(Y, seen, A, B)
    irt_part = np.array(
        [item_curve(thetas[j], A, B)[0, unseen].mean() for j in range(Y.shape[0])]
    )
    subset_accuracies = lambd * data_part + (1 - lambd) * irt_part
    return full_accuracies, subset_accuracies


def _gpirt_accuracies(
    Y: np.ndarray,
    anchor_indices: np.ndarray,
    weights: Optional[np.ndarray],
    A: np.ndarray,
    B: np.ndarray,
    Y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """tinyBenchmarks gp-IRT: lam*anchor_estimate + (1-lam)*pirt_estimate.

    lam = b^2 / (v/(4n) + b^2), where b is the mean absolute error of the pure IRT
    extrapolation on the training models (theta refitted from anchors only) and v is
    the mean per-model item variance in the training responses.
    """
    n_items = Y.shape[1]
    seen = np.asarray(anchor_indices, dtype=int)
    unseen = np.setdiff1d(np.arange(n_items), seen)

    full_accuracies, pirt_estimates = _pirt_accuracies(Y, anchor_indices, A, B)

    Y_seen = Y[:, seen]
    if weights is not None:
        anchor_estimates = np.average(Y_seen, axis=1, weights=weights)
    else:
        anchor_estimates = np.mean(Y_seen, axis=1)

    if unseen.size == 0:
        b = 0.0
    else:
        train_thetas = _fit_thetas(Y_train, seen, A, B)
        train_irt_preds = np.array(
            [item_curve(train_thetas[t], A, B)[0, unseen].mean() for t in range(Y_train.shape[0])]
        )
        train_truth = np.mean(Y_train[:, unseen], axis=1)
        b = float(np.mean(np.abs(train_irt_preds - train_truth)))

    v = float(np.var(Y_train, axis=1).mean())
    anchor_var = v / (4 * seen.size)
    lam = (b ** 2) / (anchor_var + b ** 2) if (anchor_var + b ** 2) > 0 else 0.0

    subset_accuracies = lam * anchor_estimates + (1 - lam) * pirt_estimates
    diagnostics = {
        "b": b,
        "v": v,
        "gpirt_lambda": float(lam),
        "pirt_lambda": float(seen.size / n_items),
        "num_train_models": int(Y_train.shape[0]),
    }
    return full_accuracies, subset_accuracies, diagnostics


def evaluate_split(
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
    if aggregation_mode not in VALID_AGGREGATION_MODES:
        raise ValueError(
            f"aggregation_mode must be one of {VALID_AGGREGATION_MODES}; got {aggregation_mode!r}"
        )
    if prediction_method not in VALID_PREDICTION_METHODS:
        raise ValueError(
            f"prediction_method must be one of {VALID_PREDICTION_METHODS}; got {prediction_method!r}"
        )
    if prediction_method != "weighted":
        if aggregation_mode != "item":
            raise ValueError(
                f"prediction_method {prediction_method!r} only supports aggregation_mode 'item'."
            )
        if irt_params is None:
            raise ValueError(
                f"prediction_method {prediction_method!r} requires irt_params=(A, B) from the trained IRT model."
            )
    if aggregation_mode == "weighted_task" and subscenario_weights is None:
        raise ValueError(
            "aggregation_mode 'weighted_task' requires subscenario_weights (mapping subscenario -> weight)."
        )
    if prediction_method == "gpirt" and train_responses is None:
        raise ValueError(
            "prediction_method 'gpirt' requires train_responses (training models' response matrix)."
        )
    if len(scenarios) != 1:
        raise ValueError("evaluate_split currently expects exactly one aggregate scenario.")

    scenarios_position, subscenarios_position = prepare_data(scenarios, data)
    Y = create_responses(scenarios, data)

    all_model_names = np.asarray(data["models"])
    num_models_total = Y.shape[0]

    filter_list: Optional[List[str]] = None
    missing_models: List[str] = []
    if model_filter is not None:
        filter_list = list(model_filter)
        lookup = {name: idx for idx, name in enumerate(all_model_names)}
        selected_indices = []
        for name in filter_list:
            if name not in lookup:
                missing_models.append(name)
                continue
            selected_indices.append(lookup[name])

        if not selected_indices:
            raise ValueError("None of the requested model filters were found in the dataset.")

        Y = Y[selected_indices]
        model_names = all_model_names[selected_indices]
    else:
        model_names = all_model_names

    scenario = next(iter(scenarios))
    subscenarios = tuple(scenarios[scenario])
    anchor_indices = np.asarray(anchor_indices, dtype=int)
    weights_array = None if weights is None else np.asarray(weights, dtype=float)

    task_counts = None
    gpirt_diagnostics = None
    if prediction_method in ("pirt", "gpirt"):
        # Align IRT item parameters with Y's columns (A/B's third dim spans all dataset items).
        positions = np.asarray(scenarios_position[scenario], dtype=int)
        A, B = irt_params
        A_scenario = A[:, :, positions]
        B_scenario = B[:, :, positions]
        if prediction_method == "pirt":
            full_accuracies, subset_accuracies = _pirt_accuracies(
                Y, anchor_indices, A_scenario, B_scenario
            )
        else:
            Y_train = np.asarray(train_responses, dtype=float)
            if Y_train.shape[1] != Y.shape[1]:
                raise ValueError(
                    f"train_responses has {Y_train.shape[1]} items but the evaluated split has {Y.shape[1]}."
                )
            full_accuracies, subset_accuracies, gpirt_diagnostics = _gpirt_accuracies(
                Y, anchor_indices, weights_array, A_scenario, B_scenario, Y_train
            )
    elif aggregation_mode in ("task_balanced", "weighted_task"):
        full_accuracies, subset_accuracies, task_counts = _task_balanced_accuracies(
            Y,
            subscenarios_position,
            scenario,
            subscenarios,
            anchor_indices,
            weights_array,
            subscenario_weights=subscenario_weights if aggregation_mode == "weighted_task" else None,
        )
    else:
        full_accuracies, subset_accuracies = _item_level_accuracies(
            Y, anchor_indices, weights_array
        )

    full_accuracies_dict, full_ranked_models, full_ranks = _ranked_accuracies(
        model_names, full_accuracies
    )
    subset_accuracies_dict, subset_ranked_models, subset_ranks = _ranked_accuracies(
        model_names, subset_accuracies
    )

    full_rank_list = np.array([full_ranks[str(model)] for model in model_names])
    subset_rank_list = np.array([subset_ranks[str(model)] for model in model_names])
    sp_coeff, sp_pvalue = spearmanr(full_rank_list, subset_rank_list)

    result = {
        "aggregation_mode": aggregation_mode,
        "prediction_method": prediction_method,
        "num_models_total": int(num_models_total),
        "num_models_evaluated": int(Y.shape[0]),
        "num_items_full": int(len(scenarios_position[scenario])),
        "num_items_subset": int(anchor_indices.size),
        "full_accuracies": full_accuracies_dict,
        "full_ranking": full_ranked_models,
        "subset_accuracies": subset_accuracies_dict,
        "subset_ranking": subset_ranked_models,
        "spearman": {"coeff": float(sp_coeff), "pvalue": float(sp_pvalue)},
        "model_names": [str(name) for name in model_names.tolist()],
        "model_filter": filter_list,
        "missing_models": missing_models,
    }
    if task_counts is not None:
        result["task_counts"] = task_counts
    if gpirt_diagnostics is not None:
        result["gpirt"] = gpirt_diagnostics
    return result
