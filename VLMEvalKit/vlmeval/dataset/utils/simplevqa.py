mapper = {"正确": "is_correct", "错误": "is_incorrect", "未尝试": "is_not_attempted"}


def normalize_judge_conclusion(value):
    """Normalise one judge verdict exactly the way ``SimpleVQAEval`` does.

    The dict branch deliberately has no len()>3 fallback and the result may be
    unmappable (e.g. the judge's parse-failure marker) -- such items count into the denominator
    but into none of the three buckets, matching the official scorer.
    """
    if isinstance(value, dict):
        return value["conclusion"].replace("**", "")
    if value not in ["正确", "错误", "未尝试"]:
        value = value.replace("**", "")
        if len(value) > 3:
            value = "未尝试"
    return value


def divide_dict_values(d, divisor):
    for key, value in d.items():
        if isinstance(value, dict):
            divide_dict_values(value, divisor)
        else:
            d[key] = value / divisor
    return d


def SimpleVQAEval(data):
    amount = len(data)
    result_metrics = {}
    model_keys = ["model_response"]

    for idx, line in enumerate(data):
        judge_res = line["judge_res"].copy()
        for key, value in judge_res.items():
            judge_res[key] = normalize_judge_conclusion(value)

            if key not in result_metrics:
                result_metrics[key] = {
                    "is_correct": 0,
                    "is_incorrect": 0,
                    "is_not_attempted": 0
                }

            try:
                result_metrics[key][mapper[judge_res[key]]] += 1
            except Exception:
                print(f"Error in Mapper Key: {judge_res[key]}")

            if key not in model_keys:
                model_keys.append(key)

    print("ALL data count: ", amount)
    res_json = []
    # Aggregate metrics
    for model_key in model_keys:
        aggregate_metrics = {
            "LVLM_name": model_key,
            "is_correct": round(result_metrics[model_key]["is_correct"] / amount, 4),
            "is_incorrect": round(result_metrics[model_key]["is_incorrect"] / amount, 4),
            "is_not_attempted": round(result_metrics[model_key]["is_not_attempted"] / amount, 4),
        }
        aggregate_metrics["is_given_attempted"] = round(
            aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"], 4)
        # Calculate accuracy_given_attempted
        aggregate_metrics["accuracy_given_attempted"] = (
            aggregate_metrics["is_correct"]
            / aggregate_metrics["is_given_attempted"]
            if aggregate_metrics["is_given_attempted"] > 0
            else 0
        )
        aggregate_metrics["f1"] = (
            2 * aggregate_metrics["accuracy_given_attempted"] * aggregate_metrics["is_correct"]
            / (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"])
            if (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"]) > 0
            else 0
        )
        print(model_key + ": AGGREGATE METRICS")
        print(aggregate_metrics)
        print("##################")

        res_json.append(aggregate_metrics)

        output_d = {
            "accuracy_given_attempted": aggregate_metrics["accuracy_given_attempted"],
            "f1": aggregate_metrics["f1"]
        }

        print(f"Accuracy Given Attempted: {output_d['accuracy_given_attempted']:.3f}")
        print(f"F1 Score: {output_d['f1']:.3f}")

    return aggregate_metrics
