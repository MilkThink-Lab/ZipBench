#!/usr/bin/env python
"""Standalone MMMU-Pro evaluator.

Subcommands:
- run:  inference on MMMU_Pro_10c / MMMU_Pro_V and save predictions
        (--backend vllm for local vLLM batch inference, --backend api for
        API models configured in vlmeval/config.py).
- eval: convert predictions to official JSONL, call official evaluate.py, summarize.
- run_and_eval: run inference then evaluate in one shot.
"""

from __future__ import annotations

import argparse
import os

from mmmu_pro import backend_api, backend_vllm, common, official


def _infer_model_name(model_path: str) -> str:
    """Infer a short model name from the model path."""
    return os.path.basename(model_path.rstrip("/"))


def _prepare_run(args: argparse.Namespace):
    """Shared run prologue: validate names, load input, ensure official repo,
    resolve prompts. Returns (df, prompts)."""
    # Model name resolution
    if getattr(args, "model", None) and not args.model_name:
        args.model_name = _infer_model_name(args.model)
    if not args.model_name:
        raise ValueError("Please provide --model-name (or --model to auto-infer it)")
    if args.backend == "vllm" and not getattr(args, "model", None):
        raise ValueError("--model is required for --backend vllm")

    # Resolve input
    input_path = args.input
    if not input_path:
        input_path = os.path.join(args.work_dir, "LMUData", f"{args.dataset}.tsv")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = common.read_table(input_path)
    print(f"Loaded {len(df)} samples from {input_path}")

    df = common.apply_zip_subset(df, args.dataset, getattr(args, "subset", "full"))

    # Ensure official repo is available (for prompts.yaml)
    if args.work_dir:
        official.ensure_official_mmmu_pro(
            args.work_dir,
            getattr(args, "official_repo", None),
            getattr(args, "official_ref", None),
        )

    prompts = common.get_prompts(
        args.work_dir,
        prompt_version=args.prompt_version,
        prompt_file=args.prompt_file,
    )
    print(f"Using prompts (version={args.prompt_version}):")
    print(f"  standard={prompts['standard']!r:.60s}...")
    print(f"  vision  ={prompts['vision']!r:.60s}...")
    return df, prompts


def run_inference(args: argparse.Namespace) -> str:
    df, prompts = _prepare_run(args)
    if args.backend == "vllm":
        return backend_vllm.run_inference(args, df, prompts)
    return backend_api.run_inference(args, df, prompts)


def run_and_eval(args: argparse.Namespace) -> None:
    """Run inference then evaluate in a single invocation."""
    pred_path = run_inference(args)
    args.input = pred_path
    official.evaluate_mmmu_pro(args)


def evaluate(args: argparse.Namespace) -> None:
    official.evaluate_mmmu_pro(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_shared_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend", required=True, choices=["vllm", "api"],
        help="Inference backend: local vLLM or API models from supported_VLM",
    )
    p.add_argument(
        "--model", default=None,
        help="[vllm] Model path or HF id (required for --backend vllm)",
    )
    p.add_argument(
        "--model-name", default=None,
        help="Model alias for output paths. For --backend api this must be a "
             "key in supported_VLM; for vllm it defaults to basename of --model",
    )
    p.add_argument(
        "--dataset", required=True,
        choices=["MMMU_Pro_10c", "MMMU_Pro_V"],
        help="Dataset variant",
    )
    p.add_argument(
        "--subset", default="full", choices=["full", "small", "tiny"],
        help="ZipBench subset: 'full' runs the whole set; small/tiny run the "
             "weighted compressed subsets from vlmeval/zipbench/subsets/",
    )
    p.add_argument(
        "--run-tag", default="",
        help="Optional tag appended to the run dir name (e.g. r1) so repeated "
             "runs of the same config never clobber each other",
    )
    p.add_argument("--input", default=None,
                   help="Input TSV (default: {work_dir}/LMUData/{dataset}.tsv)")
    p.add_argument("--output", default=None, help="Output predictions TSV")
    p.add_argument("--work-dir", default=os.getcwd(),
                   help="Working directory (default: current directory)")
    p.add_argument("--prompt-version", default="direct",
                   help="Prompt version key in prompts.yaml (e.g. direct, cot)")
    p.add_argument("--prompt-file", default=None,
                   help="Custom YAML file path for prompts (overrides official prompts.yaml)")
    p.add_argument("--official-repo", default=None,
                   help=f"Official repo URL (default: {official.DEFAULT_OFFICIAL_REPO})")
    p.add_argument("--official-ref", default=None, help="Git ref to pin")
    # vllm-specific
    p.add_argument("--batch-size", type=int, default=16, help="[vllm] Batch size")
    p.add_argument("--tensor-parallel-size", type=int, default=0,
                   help="[vllm] TP size (0 = auto-detect GPU count)")
    p.add_argument("--max-model-len", type=int, default=32768, help="[vllm] Max model length")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="[vllm] Max new tokens (for api the model config value is used)")
    p.add_argument("--temperature", type=float, default=0.6, help="[vllm] Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.95, help="[vllm] Top-p")
    p.add_argument("--top-k", type=int, default=20, help="[vllm] Top-k")
    p.add_argument("--seed", type=int, default=0,
                   help="[vllm] Engine seed (default 0 = prior behaviour)")
    p.add_argument("--max-num-seqs", type=int, default=8,
                   help="[vllm] Max concurrent sequences in the engine "
                        "(default 8 = prior behaviour)")
    p.add_argument("--save-every", type=int, default=100,
                   help="[vllm] Checkpoint predictions every N samples")
    p.add_argument("--resume", action="store_true",
                   help="[vllm] Resume from existing prediction TSV")
    p.add_argument("--trust-remote-code", action="store_true", help="[vllm]")
    # api-specific
    p.add_argument("--nproc", type=int, default=4,
                   help="[api] Number of concurrent threads")


def _add_eval_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-score", default=None, help="Score CSV path")
    p.add_argument("--output-detail", default=None, help="Detail TSV path")
    p.add_argument(
        "--official-method",
        choices=official.OFFICIAL_METHOD_CHOICES,
        default=None,
        help="Method tag in the official JSONL filename. Defaults to 'cot' when "
             "--prompt-version contains 'cot', otherwise 'direct'.",
    )
    p.add_argument(
        "--keep-official-output",
        action="store_true",
        help="Keep the official eval output directory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone MMMU-Pro evaluator (vLLM / API backends)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- run ----
    run_p = sub.add_parser("run", help="Run inference")
    _add_shared_run_args(run_p)
    run_p.set_defaults(func=run_inference)

    # ---- eval ----
    eval_p = sub.add_parser("eval", help="Evaluate predictions via official script")
    eval_p.add_argument("--model", default=None,
                        help="Model path (used to infer --model-name if not set)")
    eval_p.add_argument("--model-name", default=None,
                        help="Model alias used during inference (for run dir / JSONL naming)")
    eval_p.add_argument(
        "--dataset", required=True,
        choices=["MMMU_Pro_10c", "MMMU_Pro_V"],
        help="Dataset variant",
    )
    eval_p.add_argument(
        "--subset", default="full", choices=["full", "small", "tiny"],
        help="ZipBench subset used during inference (locates the _ZIP_<subset> "
             "run dir and enables weighted aggregation)",
    )
    eval_p.add_argument("--input", default=None,
                        help="Prediction TSV with 'prediction' column "
                             "(default: auto-resolved from run dir)")
    eval_p.add_argument("--work-dir", default=os.getcwd(),
                        help="Working directory (default: current directory)")
    eval_p.add_argument("--prompt-version", default="direct",
                        help="Prompt version used during inference (for locating run dir)")
    eval_p.add_argument("--max-tokens", default=2048,
                        help="Max tokens used during inference (for locating run dir)")
    eval_p.add_argument(
        "--run-tag", default="",
        help="Run tag used during inference (for locating the tagged run dir)",
    )
    eval_p.add_argument("--official-repo", default=None,
                        help=f"Official repo URL (default: {official.DEFAULT_OFFICIAL_REPO})")
    eval_p.add_argument("--official-ref", default=None, help="Git ref to pin")
    _add_eval_output_args(eval_p)
    eval_p.set_defaults(func=evaluate)

    # ---- run_and_eval ----
    re_p = sub.add_parser("run_and_eval", help="Run inference then evaluate in one shot")
    _add_shared_run_args(re_p)
    _add_eval_output_args(re_p)
    re_p.set_defaults(func=run_and_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # eval also supports inferring model_name from --model
    if getattr(args, "model", None) and not args.model_name:
        args.model_name = _infer_model_name(args.model)
    if not getattr(args, "model_name", None):
        parser.error("Please provide --model-name (or --model to auto-infer it)")

    args.func(args)


if __name__ == "__main__":
    main()
