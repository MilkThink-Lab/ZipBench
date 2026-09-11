import argparse
import numpy as np
import pickle
from irt import *
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--out-subdir", type=str, default="",
                        help="Optional subdirectory under dataset/{scenario}/ for outputs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override full output directory for IRT dataset/model files")
    parser.add_argument("--train-pkl", type=str, required=True,
                        help="Path to the train pkl")
    parser.add_argument("--scenario", type=str, required=True,
                        help="Scenario key and output directory name")
    parser.add_argument("--subscenarios", nargs="+", default=None,
                        help="Data keys to concatenate under the aggregate scenario; defaults to scenario")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device passed to py-irt, e.g. cuda or cpu")
    args = parser.parse_args()

    scenario = args.scenario
    scenarios = {scenario: args.subscenarios or [scenario]}

    train_pkl = args.train_pkl
    with open(train_pkl, "rb") as f:
        data = pickle.load(f)

    names = np.array(data["models"])
    scenarios_position, subscenarios_position = prepare_data(scenarios, data)
    Y = create_responses(scenarios, data)

    D = 5
    device = args.device
    epochs = 2000
    lr = 0.1

    out_dir = args.output_dir or f"./output/{scenario}"
    if args.out_subdir and args.output_dir is None:
        out_dir = f"{out_dir}/{args.out_subdir}"
    import os
    os.makedirs(out_dir, exist_ok=True)
    model_dir = f"{out_dir}/irt_model_run{args.run_id}"
    dataset_path = f"{out_dir}/irt_dataset_run{args.run_id}.jsonlines"

    create_irt_dataset(Y, dataset_path)
    train_irt_model(
        dataset_name=dataset_path,
        model_name=model_dir,
        D=D, lr=lr, epochs=epochs, device=device, seed=args.seed,
    )
    A, B, Theta = load_irt_parameters(f"{model_dir}/")
    print(f"Run {args.run_id} (seed={args.seed}): A={A.shape}, B={B.shape}, Theta={Theta.shape}")

if __name__ == "__main__":
    main()
