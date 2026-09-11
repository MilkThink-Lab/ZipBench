#!/usr/bin/env python
"""Standalone OmniDocBench v1.5 evaluator (vLLM / API backends).

Subcommands:
- run:          inference from official v1.5 GT JSON + image root
                (--backend vllm for local vLLM batch inference, --backend api
                for API models configured in vlmeval/config.py).
- eval:         run the official OmniDocBench v1.5 end2end evaluator.
- run_and_eval: run inference and official evaluation in one invocation.

The script intentionally does not modify vlmeval/dataset/OmniDocBench so it can
coexist with the existing v1 support in this checkout.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional, Tuple

from omnidocbench_v15 import backend_api, backend_vllm, common, official, zip_subset


def _infer_model_name(model_path: str) -> str:
    """Infer a short model name from the model path."""
    return os.path.basename(model_path.rstrip("/"))


def _prepare_run(args: argparse.Namespace) -> Optional[Tuple[list, str]]:
    """Shared run prologue: resolve model name, load GT, resolve images and
    prompt. Returns (records, prompt), or None on --dry-run."""
    args.work_dir = common._abspath(args.work_dir)

    if getattr(args, "model", None) and not args.model_name:
        args.model_name = _infer_model_name(args.model)
    if args.backend == "vllm" and not getattr(args, "model", None):
        raise ValueError("--model is required for --backend vllm")
    if not args.model_name:
        raise ValueError(
            "Please provide --model-name (a supported_VLM key for --backend api; "
            "auto-inferred from --model for --backend vllm)"
        )

    gt_json = common._abspath(args.gt_json)
    image_root = common._abspath(args.image_root)
    out_dir = common._output_dir(args)
    common._pred_md_dir(args)

    samples = common.load_gt_samples(gt_json, args.limit)
    if getattr(args, "subset", "full") != "full":
        pages = zip_subset.union_pages(zip_subset.load_specs(args.subset))
        samples = zip_subset.filter_gt_samples(samples, pages)
        print(f"ZipBench subset {args.subset}: union of 6 sub-metric specs = "
              f"{len(pages)} pages, {len(samples)} GT samples kept")
    records = common.resolve_sample_images(samples, image_root)
    print(f"Loaded {len(samples)} GT samples from {gt_json}")
    print(f"Resolved {len(records)} images under {image_root}")
    print(f"Output directory: {out_dir}")

    prompt = common.load_prompt(args)

    if args.dry_run:
        if args.backend == "api":
            print("Dry run: not calling model API")
        else:
            print("Dry run: not running vLLM inference")
        if records:
            print(f"First sample: {records[0]['image_basename']} -> {records[0]['image_path']}")
            print(f"Prompt preview: {prompt[:200].replace(os.linesep, ' ')}...")
        return None
    return records, prompt


def run_inference(args: argparse.Namespace) -> Optional[str]:
    prep = _prepare_run(args)
    if prep is None:
        return None
    records, prompt = prep
    if args.backend == "vllm":
        return backend_vllm.run_inference(args, records, prompt)
    return backend_api.run_inference(args, records, prompt)


def run_and_eval(args: argparse.Namespace) -> None:
    run_inference(args)
    args.pred_md_dir = common._pred_md_dir(args)
    official.evaluate_omnidocbench(args)


def evaluate(args: argparse.Namespace) -> None:
    official.evaluate_omnidocbench(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--work-dir",
        default=common.DEFAULT_WORK_DIR,
        help=f"Working directory (default: {common.DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--official-dir",
        default=None,
        help=(
            "Path to official OmniDocBench repo checkout "
            f"(default: <work-dir>/third_party/{official.DEFAULT_OFFICIAL_DIRNAME})"
        ),
    )
    parser.add_argument(
        "--official-repo",
        default=official.DEFAULT_OFFICIAL_REPO,
        help=f"Official repo URL (default: {official.DEFAULT_OFFICIAL_REPO})",
    )
    parser.add_argument(
        "--official-ref",
        default=official.DEFAULT_OFFICIAL_REF,
        help=f"Official git ref (default: {official.DEFAULT_OFFICIAL_REF})",
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", required=True, choices=["vllm", "api"],
        help="Inference backend: local vLLM or API models from supported_VLM",
    )
    parser.add_argument(
        "--model", default=None,
        help="[vllm] Model path or HF id (required for --backend vllm)",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Model alias for output paths. For --backend api this must be a "
             "key in supported_VLM; for vllm it defaults to basename of --model",
    )
    parser.add_argument("--gt-json", required=True, help="Official OmniDocBench v1.5 GT JSON")
    parser.add_argument("--image-root", required=True, help="Directory containing OmniDocBench page images")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/OmniDocBench_v1_5/<model>)")
    parser.add_argument("--output", default=None, help="Prediction TSV path (default: <output-dir>/pred.tsv)")
    parser.add_argument("--checkpoint", default=None, help="[api] Checkpoint pkl path (default: <output-dir>/supp.pkl)")
    parser.add_argument("--pred-md-dir", default=None, help="Prediction markdown directory (default: <output-dir>/pred_md)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N GT samples")
    parser.add_argument(
        "--subset", choices=zip_subset.SUBSET_CHOICES, default="full",
        help="ZipBench subset: restrict inference to the union of the 6 sub-metric "
             "specs and report weighted scores (default: full = unchanged behaviour)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print actions without running inference")
    parser.add_argument("--prompt-file", default=None, help="Custom prompt text file (overrides built-in OmniDocBench v1.5 prompt)")
    # vllm-specific
    parser.add_argument("--batch-size", type=int, default=16, help="[vllm] Batch size")
    parser.add_argument("--tensor-parallel-size", type=int, default=0,
                        help="[vllm] TP size (0 = auto-detect GPU count)")
    parser.add_argument("--max-model-len", type=int, default=32768, help="[vllm] Max model length")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="[vllm] Max new tokens; full-page Markdown is long "
                             "(for api the model config value is used)")
    parser.add_argument("--temperature", type=float, default=0.6, help="[vllm] Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="[vllm] Top-p")
    parser.add_argument("--top-k", type=int, default=20, help="[vllm] Top-k")
    parser.add_argument("--seed", type=int, default=0, help="[vllm] Engine seed")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                        help="[vllm] Max concurrent sequences in the engine")
    parser.add_argument("--save-every", type=int, default=100,
                        help="[vllm] Checkpoint predictions every N samples")
    parser.add_argument("--resume", action="store_true",
                        help="[vllm] Resume from existing prediction TSV")
    parser.add_argument("--trust-remote-code", action="store_true", help="[vllm]")
    # api-specific
    parser.add_argument("--nproc", type=int, default=4, help="[api] Number of concurrent API calls")
    _add_repo_args(parser)


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gt-json", required=True, help="Official OmniDocBench v1.5 GT JSON")
    parser.add_argument("--pred-md-dir", default=None, help="Prediction markdown directory (default: <output-dir>/pred_md)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/OmniDocBench_v1_5/<model>)")
    parser.add_argument(
        "--model-name",
        default="unknown_model",
        help="Model key/name used only for default output-dir when --output-dir is omitted",
    )
    _add_eval_only_args(parser)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N GT samples")
    parser.add_argument(
        "--subset", choices=zip_subset.SUBSET_CHOICES, default="full",
        help="ZipBench subset: evaluate against the subset GT (union of the 6 "
             "sub-metric specs) and report weighted scores (default: full)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write config and print command without running eval")
    _add_repo_args(parser)


def _add_eval_only_args(parser: argparse.ArgumentParser) -> None:
    """Eval options shared by the `eval` and `run_and_eval` subcommands."""
    parser.add_argument(
        "--formula-metric",
        choices=["CDM", "CDM_plain"],
        default="CDM",
        help="Formula metric in official config (default: CDM)",
    )
    parser.add_argument(
        "--eval-backend",
        choices=["docker", "local"],
        default="docker",
        help="Run official eval in Docker or current Python env (default: docker). "
             "Unrelated to the inference --backend flag.",
    )
    parser.add_argument(
        "--docker-image",
        default=official.DEFAULT_DOCKER_IMAGE,
        help=f"Docker image for official eval (default: {official.DEFAULT_DOCKER_IMAGE})",
    )
    parser.add_argument("--match-method", default="quick_match", help="Official matching method")
    parser.add_argument("--output-score-json", default=None, help="Summary score JSON path")
    parser.add_argument("--output-score-csv", default=None, help="Summary score CSV path")
    parser.add_argument(
        "--keep-official-output",
        action="store_true",
        help="Do not remove an existing official_eval/result directory before evaluation",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniDocBench v1.5 runner (vLLM / API) and official evaluator"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run inference (--backend vllm or api)")
    _add_run_args(run_p)
    run_p.set_defaults(func=run_inference)

    eval_p = sub.add_parser("eval", help="Run official OmniDocBench v1.5 evaluation")
    _add_eval_args(eval_p)
    eval_p.set_defaults(func=evaluate)

    re_p = sub.add_parser("run_and_eval", help="Run inference then official evaluation")
    _add_run_args(re_p)
    # argparse cannot reuse the same option string in one parser, so only add
    # eval-only options that are not already present from _add_run_args.
    _add_eval_only_args(re_p)
    re_p.set_defaults(func=run_and_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    start = time.time()
    args.func(args)
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
