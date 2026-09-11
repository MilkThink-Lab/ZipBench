#!/usr/bin/env python3
"""LiveCodeBench evaluation script aligned with official lcb_runner flow.

DEPRECATED: for codegeneration / codeexecution / testoutputprediction use the
OpenCompass config-flow entry point instead (``examples/eval_livecodebench_api.py``
+ ``opencompass/configs/datasets/livecodebench/livecodebench_official_gen.py``
/ ``livecodebench_official_extra_gen.py``), which works with any OpenCompass
model config (API or local). This standalone vLLM script is kept only as the
reference for the selfrepair flow, pending its migration into the config flow.

Supports 4 sub-tasks:
  - codegeneration
  - selfrepair
  - codeexecution
  - testoutputprediction

Uses official HuggingFace datasets, official prompt content, official
evaluation metrics (vendored lcb_runner/evaluation/), and
tokenizer.apply_chat_template() for prompt formatting.
"""

import argparse
import base64
import json
import os
import pickle
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Vendored official evaluation --  add to sys.path
# ---------------------------------------------------------------------------
_THIRD_PARTY_DIR = str(
    Path(__file__).resolve().parent.parent / "third_party" / "livecodebench"
)
if _THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_DIR)

from lcb_runner.evaluation import (  # noqa: E402
    code_execution_metrics,
    codegen_metrics,
    extract_instance_results,
    test_output_metrics,
)

# Reusable helpers from OpenCompass (already confirmed aligned with official).
# Import directly via importlib to avoid triggering the full opencompass
# package init chain which has heavy dependencies.
import importlib.util as _ilu

_LCB_DATASET_DIR = str(
    Path(__file__).resolve().parent.parent
    / "opencompass"
    / "datasets"
    / "livecodebench"
)
_DEFAULT_HF_CACHE_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "hf_cache_livecodebench_lite"
)
_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "default"
    / "livecodebench"
)


def _import_module_from_file(module_name, file_path):
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prompts_mod = _import_module_from_file(
    "lcb_prompts", os.path.join(_LCB_DATASET_DIR, "prompts.py")
)
_extract_mod = _import_module_from_file(
    "lcb_extract", os.path.join(_LCB_DATASET_DIR, "extract_utils.py")
)

make_code_execution_prompt = _prompts_mod.make_code_execution_prompt
get_generic_question_template_test_completion = (
    _prompts_mod.get_generic_question_template_test_completion
)
extract_code = _extract_mod.extract_code_generation_v2
extract_code_execution = _extract_mod.extract_code_execution
extract_test_output_code = _extract_mod.extract_test_output_code


# ===================================================================
# Prompt constants (aligned with official lcb_runner/prompts/)
# ===================================================================
CODEGEN_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)

SELFREPAIR_SYSTEM_MESSAGE = (
    "You are a helpful programming assistant and an expert Python programmer. "
    "You are helping a user write a program to solve a problem. The user has "
    "written some code, but it has some errors and is not passing the tests. "
    "You will help the user by first giving a concise (at most 2-3 sentences) "
    "textual explanation of what is wrong with the code. After you have "
    "pointed out what is wrong with the code, you will then generate a fixed "
    "version of the program. You must put the entire fixed program within "
    "code delimiters only for once."
)

CODE_EXEC_SYSTEM_MESSAGE = (
    "You are an expert at Python programming, code execution, test case "
    "generation, and fuzzing."
)

TEST_OUTPUT_SYSTEM_MESSAGE = (
    "You are a helpful programming assistant and an expert Python programmer. "
    "You are helping a user to write a test case to help to check the "
    "correctness of the function. The user has written a input for the "
    "testcase. You will calculate the output of the testcase and write the "
    "whole assertion statement in the markdown code block with the correct "
    "output."
)

FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


# ===================================================================
# Prompt formatting
# ===================================================================
def format_prompt(tokenizer, system_message, user_message, enable_thinking=False):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **kwargs,
    )


# ===================================================================
# Benchmark loading
# ===================================================================
def configure_hf_cache(args):
    cache_dir = Path(args.hf_cache_dir).expanduser() if args.hf_cache_dir else None

    if cache_dir is None and _DEFAULT_HF_CACHE_DIR.exists():
        cache_dir = _DEFAULT_HF_CACHE_DIR

    if cache_dir is None:
        return None

    cache_dir = cache_dir.resolve()
    datasets_dir = cache_dir / "datasets"
    hub_dir = cache_dir / "hub"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    hub_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    if not args.hf_online:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    args.hf_cache_dir = str(cache_dir)
    return args.hf_cache_dir


def load_lcb_dataset(dataset_name, split, args, **kwargs):
    from datasets import DownloadConfig, load_dataset

    load_kwargs = dict(split=split, **kwargs)
    if args.hf_cache_dir:
        load_kwargs["cache_dir"] = args.hf_cache_dir
    if not args.hf_online:
        load_kwargs["download_config"] = DownloadConfig(local_files_only=True)
    return load_dataset(dataset_name, **load_kwargs)


def load_codegen_benchmark(args):
    """Load code generation benchmark from official HF dataset."""
    if args.not_fast:
        dataset = load_lcb_dataset(
            "livecodebench/code_generation",
            args=args,
            split="test",
            trust_remote_code=True,
        )
    else:
        dataset = load_lcb_dataset(
            "livecodebench/code_generation_lite",
            args=args,
            split="test",
            version_tag=args.release_version,
            trust_remote_code=True,
        )

    benchmark = []
    for item in dataset:
        entry = dict(item)

        # Deserialize test cases
        public_test_cases = json.loads(entry["public_test_cases"])
        try:
            private_test_cases = json.loads(entry["private_test_cases"])
        except Exception:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(
                            entry["private_test_cases"].encode("utf-8")
                        )
                    )
                )
            )

        metadata = json.loads(entry["metadata"])
        entry["evaluation_sample"] = json.dumps(
            {
                "inputs": [
                    t["input"] for t in public_test_cases + private_test_cases
                ],
                "outputs": [
                    t["output"] for t in public_test_cases + private_test_cases
                ],
                "fn_name": metadata.get("func_name", None),
            }
        )
        benchmark.append(entry)

    # Date filtering
    if args.start_date:
        p_start = datetime.strptime(args.start_date, "%Y-%m-%d")
        benchmark = [
            b
            for b in benchmark
            if p_start <= datetime.fromisoformat(b["contest_date"])
        ]
    if args.end_date:
        p_end = datetime.strptime(args.end_date, "%Y-%m-%d")
        benchmark = [
            b
            for b in benchmark
            if datetime.fromisoformat(b["contest_date"]) <= p_end
        ]

    benchmark.sort(key=lambda x: x["question_id"])
    return benchmark


def load_code_execution_benchmark(args):
    """Load code execution benchmark from official HF dataset."""
    dataset = load_lcb_dataset("livecodebench/execution-v2", args=args, split="test")
    benchmark = [dict(item) for item in dataset]
    benchmark.sort(key=lambda x: int(x["id"].split("_")[1]))
    return benchmark


def load_test_output_benchmark(args):
    """Load test output prediction benchmark from official HF dataset."""
    dataset = load_lcb_dataset(
        "livecodebench/test_generation",
        args=args,
        split="test",
        trust_remote_code=True,
    )
    benchmark = [dict(item) for item in dataset]
    benchmark.sort(key=lambda x: (x["question_id"], x["test_id"]))
    return benchmark


# ===================================================================
# Prompt builders for each scenario
# ===================================================================
def build_codegen_prompts(benchmark, tokenizer, args):
    prompts = []
    for item in benchmark:
        starter_code = item.get("starter_code", "")
        if starter_code:
            formatting_msg = FORMATTING_WITH_STARTER
            code_block = f"```python\n{starter_code}\n```"
        else:
            formatting_msg = FORMATTING_WITHOUT_STARTER
            code_block = "```python\n# YOUR CODE HERE\n```"

        user_message = (
            f"### Question:\n{item['question_content']}\n\n"
            f"### Format: {formatting_msg}\n{code_block}\n\n"
            f"### Answer: (use the provided format with backticks)\n\n"
        )
        prompts.append(
            format_prompt(
                tokenizer, CODEGEN_SYSTEM_MESSAGE, user_message,
                enable_thinking=args.enable_thinking,
            )
        )
    return prompts


def build_selfrepair_prompts(benchmark, eval_all_data, tokenizer, args):
    """Build self-repair prompts from codegen eval_all.json.

    Returns:
        prompts: list of prompts (one per (question, candidate) pair)
        prompt_indices: list of (question_idx, candidate_idx) for each prompt
        passed_mask: list of (question_idx, candidate_idx, original_output)
            for candidates that already passed
    """
    # Index eval_all by question_id
    eval_all_by_qid = {}
    for entry in eval_all_data:
        eval_all_by_qid[entry["question_id"]] = entry

    prompts = []
    prompt_indices = []
    passed_mask = []

    for q_idx, item in enumerate(benchmark):
        qid = item["question_id"]
        if qid not in eval_all_by_qid:
            print(f"WARNING: question_id {qid} not found in eval_all, skipping")
            continue

        ea = eval_all_by_qid[qid]
        code_list = ea["code_list"]
        graded_list = ea["graded_list"]
        metadata_list = ea.get("metadata", [])
        output_list = ea.get("output_list", [])

        for c_idx in range(len(code_list)):
            if graded_list[c_idx]:
                # Already passed -- reuse original output
                orig_output = output_list[c_idx] if c_idx < len(output_list) else ""
                passed_mask.append((q_idx, c_idx, orig_output))
            else:
                # Build repair prompt
                buggy_code = code_list[c_idx]
                metadata_str = (
                    metadata_list[c_idx]
                    if c_idx < len(metadata_list)
                    else "{}"
                )
                message_from_metadata = _get_check_prompt(metadata_str)

                user_message = (
                    f"### Question:\n{item['question_content']}\n\n"
                    f"### Answer:\n```python\n{buggy_code}\n```\n\n"
                    f"{message_from_metadata}\n"
                    f"### Format: {FORMATTING_WITHOUT_STARTER}\n"
                    f"```python\n# YOUR CODE HERE\n```\n\n"
                    f"### Answer: (use the provided format with backticks)\n\n"
                )
                prompts.append(
                    format_prompt(
                        tokenizer, SELFREPAIR_SYSTEM_MESSAGE, user_message,
                        enable_thinking=args.enable_thinking,
                    )
                )
                prompt_indices.append((q_idx, c_idx))

    return prompts, prompt_indices, passed_mask


def _get_check_prompt(metadata_str):
    """Map error_code to error message, aligned with official."""
    try:
        metadata = json.loads(metadata_str)
    except (json.JSONDecodeError, TypeError):
        return ""
    if "error_code" not in metadata:
        return ""
    ec = metadata["error_code"]
    if ec == -1:
        return (
            f"The above code is incorrect and got the following compilation "
            f"error.\n{metadata.get('error', '')}"
        )
    elif ec == -2:
        return (
            f"The above code is incorrect and got a wrong answer.\n"
            f"Input: {metadata.get('inputs', '')}\n"
            f"Generated Output: {metadata.get('output', '')}\n"
            f"Expected: {metadata.get('expected', '')}"
        )
    elif ec == -3:
        return (
            f"The above code is incorrect and got time limit exceeded.\n"
            f"{metadata.get('error', '')}\n"
            f"Input: {metadata.get('inputs', '')}\n"
            f"Expected: {metadata.get('expected', '')}"
        )
    elif ec == -4:
        return (
            f"The above code is incorrect and got a runtime error.\n"
            f"Input: {metadata.get('inputs', '')}\n"
            f"Expected: {metadata.get('expected', '')}\n"
            f"{metadata.get('error', '')}"
        )
    return ""


def build_code_execution_prompts(benchmark, tokenizer, args):
    prompts = []
    for item in benchmark:
        user_message = make_code_execution_prompt(
            item["code"], item["input"], cot=args.cot_code_execution
        )
        prompts.append(
            format_prompt(
                tokenizer, CODE_EXEC_SYSTEM_MESSAGE, user_message,
                enable_thinking=args.enable_thinking,
            )
        )
    return prompts


def build_test_output_prompts(benchmark, tokenizer, args):
    prompts = []
    for item in benchmark:
        test_data = json.loads(item["test"])
        testcase_input = test_data[0]["input"]
        user_message = get_generic_question_template_test_completion(
            question_content=item["question_content"],
            starter_code=item["starter_code"],
            testcase_input=testcase_input,
        )
        prompts.append(
            format_prompt(
                tokenizer, TEST_OUTPUT_SYSTEM_MESSAGE, user_message,
                enable_thinking=args.enable_thinking,
            )
        )
    return prompts


# ===================================================================
# vLLM inference
# ===================================================================
def run_inference(llm, prompts, sampling_params):
    """Run vLLM batch inference and return list of list of output strings."""
    outputs = llm.generate(prompts, sampling_params)
    return [[o.text for o in item.outputs] for item in outputs]


# ===================================================================
# Output extraction helpers
# ===================================================================
def extract_codegen_outputs(raw_outputs):
    """Extract code from raw outputs using 'last code block' strategy."""
    return [[extract_code(o) for o in outputs] for outputs in raw_outputs]


def extract_code_exec_outputs(raw_outputs, cot=False):
    return [
        [extract_code_execution(o, cot=cot) for o in outputs]
        for outputs in raw_outputs
    ]


def extract_test_output_outputs(raw_outputs):
    return [
        [extract_test_output_code(o) for o in outputs]
        for outputs in raw_outputs
    ]


# ===================================================================
# File naming helpers
# ===================================================================
def get_output_prefix(scenario, n, temperature):
    return f"Scenario.{scenario}_{n}_{temperature}"


def get_output_dir(args):
    model_name = args.model_name or Path(args.model_path).name
    return Path(args.output_dir) / model_name


# ===================================================================
# Save results
# ===================================================================
def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {path}")


def save_codegen_results(
    benchmark,
    raw_outputs,
    code_lists,
    metrics,
    eval_results,
    final_metadata,
    scenario_name,
    n,
    temperature,
    args,
):
    out_dir = get_output_dir(args)
    prefix = get_output_prefix(scenario_name, n, temperature)

    # Raw outputs
    raw_data = []
    for i, item in enumerate(benchmark):
        raw_data.append(
            {
                "question_id": item["question_id"],
                "output_list": raw_outputs[i],
                "code_list": code_lists[i],
            }
        )
    save_json(raw_data, out_dir / f"{prefix}.json")

    # Eval summary
    save_json(metrics, out_dir / f"{prefix}_eval.json")

    # Eval all (critical for self-repair)
    instance_results = extract_instance_results(eval_results)
    eval_all = []
    for i, item in enumerate(benchmark):
        graded = instance_results[i] if i < len(instance_results) else []
        md = final_metadata[i] if i < len(final_metadata) else []
        eval_all.append(
            {
                "question_id": item["question_id"],
                "question_content": item.get("question_content", ""),
                "contest_id": item.get("contest_id", ""),
                "contest_date": item.get("contest_date", ""),
                "starter_code": item.get("starter_code", ""),
                "output_list": raw_outputs[i],
                "code_list": code_lists[i],
                "graded_list": graded,
                "pass@1": float(
                    np.mean([all(g > 0 for g in gen) for gen in eval_results.get(i, [])])
                    * 100
                )
                if i in eval_results
                else 0.0,
                "metadata": md,
            }
        )
    save_json(eval_all, out_dir / f"{prefix}_eval_all.json")
    return eval_all


def save_code_execution_results(
    benchmark,
    raw_outputs,
    pred_lists,
    metrics,
    eval_results,
    scenario_name,
    n,
    temperature,
    args,
):
    out_dir = get_output_dir(args)
    prefix = get_output_prefix(scenario_name, n, temperature)

    raw_data = []
    for i, item in enumerate(benchmark):
        raw_data.append(
            {
                "id": item["id"],
                "output_list": raw_outputs[i],
                "pred_list": pred_lists[i],
            }
        )
    save_json(raw_data, out_dir / f"{prefix}.json")
    save_json(metrics, out_dir / f"{prefix}_eval.json")

    # eval_all
    eval_all = []
    for i, item in enumerate(benchmark):
        graded = []
        if i in eval_results:
            for gen_result in eval_results[i]:
                graded.append(all(g is True for g in gen_result))
        eval_all.append(
            {
                "id": item["id"],
                "code": item.get("code", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "output_list": raw_outputs[i],
                "pred_list": pred_lists[i],
                "graded_list": graded,
                "pass@1": float(np.mean(graded) * 100) if graded else 0.0,
            }
        )
    save_json(eval_all, out_dir / f"{prefix}_eval_all.json")


def save_test_output_results(
    benchmark,
    raw_outputs,
    pred_lists,
    metrics,
    eval_results,
    scenario_name,
    n,
    temperature,
    args,
):
    out_dir = get_output_dir(args)
    prefix = get_output_prefix(scenario_name, n, temperature)

    raw_data = []
    for i, item in enumerate(benchmark):
        raw_data.append(
            {
                "question_id": item["question_id"],
                "test_id": item["test_id"],
                "output_list": raw_outputs[i],
                "pred_list": pred_lists[i],
            }
        )
    save_json(raw_data, out_dir / f"{prefix}.json")
    save_json(metrics, out_dir / f"{prefix}_eval.json")

    eval_all = []
    for i, item in enumerate(benchmark):
        graded = []
        if i in eval_results:
            for gen_result in eval_results[i]:
                graded.append(all(g is True for g in gen_result))
        test_data = json.loads(item["test"])
        eval_all.append(
            {
                "question_id": item["question_id"],
                "test_id": item["test_id"],
                "question_content": item.get("question_content", ""),
                "starter_code": item.get("starter_code", ""),
                "output_list": raw_outputs[i],
                "pred_list": pred_lists[i],
                "graded_list": graded,
                "pass@1": float(np.mean(graded) * 100) if graded else 0.0,
            }
        )
    save_json(eval_all, out_dir / f"{prefix}_eval_all.json")


# ===================================================================
# Scenario runners
# ===================================================================
def run_codegeneration(llm, tokenizer, args, sampling_params):
    print("\n" + "=" * 60)
    print("Running: Code Generation")
    print("=" * 60)

    benchmark = load_codegen_benchmark(args)
    print(f"Loaded {len(benchmark)} problems")

    prompts = build_codegen_prompts(benchmark, tokenizer, args)

    # Debug: print first prompt
    if prompts:
        print(f"\n--- First prompt (truncated) ---\n{prompts[0][:500]}...\n")

    raw_outputs = run_inference(llm, prompts, sampling_params)
    code_lists = extract_codegen_outputs(raw_outputs)

    # Evaluation
    samples_list = [
        {"input_output": item["evaluation_sample"]} for item in benchmark
    ]
    k_list = [k for k in [1, 5, 10] if k <= args.n]
    metrics, eval_results, final_metadata = codegen_metrics(
        samples_list,
        code_lists,
        k_list=k_list,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )

    print(f"\nCode Generation metrics: {metrics}")

    eval_all = save_codegen_results(
        benchmark,
        raw_outputs,
        code_lists,
        metrics,
        eval_results,
        final_metadata,
        "codegeneration",
        args.n,
        args.temperature,
        args,
    )
    return eval_all


def run_selfrepair(llm, tokenizer, args, sampling_params):
    print("\n" + "=" * 60)
    print("Running: Self Repair")
    print("=" * 60)

    # Load same benchmark as codegen
    benchmark = load_codegen_benchmark(args)
    print(f"Loaded {len(benchmark)} problems (same as codegen)")

    # Find codegen eval_all
    if args.codegen_eval_all_path:
        eval_all_path = args.codegen_eval_all_path
    else:
        out_dir = get_output_dir(args)
        prefix = get_output_prefix(
            "codegeneration", args.codegen_n, args.temperature
        )
        eval_all_path = str(out_dir / f"{prefix}_eval_all.json")

    print(f"Loading codegen eval_all from: {eval_all_path}")
    with open(eval_all_path, "r") as f:
        eval_all_data = json.load(f)

    prompts, prompt_indices, passed_mask = build_selfrepair_prompts(
        benchmark, eval_all_data, tokenizer, args
    )

    print(
        f"Need to repair: {len(prompts)} candidates, "
        f"already passed: {len(passed_mask)} candidates"
    )

    # Run inference only on failed candidates
    if prompts:
        raw_repair_outputs = run_inference(llm, prompts, sampling_params)
    else:
        raw_repair_outputs = []

    # Reconstruct per-question outputs
    # Index eval_all by question_id
    eval_all_by_qid = {e["question_id"]: e for e in eval_all_data}

    repair_output_iter = iter(raw_repair_outputs)

    # Build repair_map: (q_idx, c_idx) -> repair output or original
    repair_map = {}
    for pi, (q_idx, c_idx) in enumerate(prompt_indices):
        repair_map[(q_idx, c_idx)] = next(repair_output_iter)

    for q_idx, c_idx, orig_output in passed_mask:
        repair_map[(q_idx, c_idx)] = [orig_output]

    # Reconstruct per-question lists
    all_raw_outputs = []
    all_code_lists = []
    for q_idx, item in enumerate(benchmark):
        qid = item["question_id"]
        ea = eval_all_by_qid.get(qid)
        if ea is None:
            continue
        n_candidates = len(ea["code_list"])
        q_raw = []
        q_codes = []
        for c_idx in range(n_candidates):
            key = (q_idx, c_idx)
            if key in repair_map:
                out = repair_map[key]
                q_raw.append(out[0] if out else "")
                # For passed candidates, reuse original code
                if ea["graded_list"][c_idx]:
                    q_codes.append(ea["code_list"][c_idx])
                else:
                    q_codes.append(extract_code(out[0]) if out else "")
            else:
                q_raw.append("")
                q_codes.append("")
        all_raw_outputs.append(q_raw)
        all_code_lists.append(q_codes)

    # Filter benchmark to only those with eval_all entries
    filtered_benchmark = [
        item for item in benchmark if item["question_id"] in eval_all_by_qid
    ]

    # Evaluation
    samples_list = [
        {"input_output": item["evaluation_sample"]} for item in filtered_benchmark
    ]
    k_list = [1]
    metrics, eval_results, final_metadata = codegen_metrics(
        samples_list,
        all_code_lists,
        k_list=k_list,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )

    print(f"\nSelf Repair metrics: {metrics}")

    save_codegen_results(
        filtered_benchmark,
        all_raw_outputs,
        all_code_lists,
        metrics,
        eval_results,
        final_metadata,
        "selfrepair",
        1,
        args.temperature,
        args,
    )


def run_codeexecution(llm, tokenizer, args, sampling_params):
    print("\n" + "=" * 60)
    print("Running: Code Execution")
    print("=" * 60)

    benchmark = load_code_execution_benchmark(args)
    print(f"Loaded {len(benchmark)} problems")

    prompts = build_code_execution_prompts(benchmark, tokenizer, args)

    if prompts:
        print(f"\n--- First prompt (truncated) ---\n{prompts[0][:500]}...\n")

    raw_outputs = run_inference(llm, prompts, sampling_params)
    pred_lists = extract_code_exec_outputs(
        raw_outputs, cot=args.cot_code_execution
    )

    # Evaluation
    samples = [
        {"code": item["code"], "input": item["input"], "output": item["output"]}
        for item in benchmark
    ]
    metrics, eval_results = code_execution_metrics(samples, pred_lists)

    print(f"\nCode Execution metrics: {metrics}")

    save_code_execution_results(
        benchmark,
        raw_outputs,
        pred_lists,
        metrics,
        eval_results,
        "codeexecution",
        args.n,
        args.temperature,
        args,
    )


def run_testoutputprediction(llm, tokenizer, args, sampling_params):
    print("\n" + "=" * 60)
    print("Running: Test Output Prediction")
    print("=" * 60)

    benchmark = load_test_output_benchmark(args)
    print(f"Loaded {len(benchmark)} problems")

    prompts = build_test_output_prompts(benchmark, tokenizer, args)

    if prompts:
        print(f"\n--- First prompt (truncated) ---\n{prompts[0][:500]}...\n")

    raw_outputs = run_inference(llm, prompts, sampling_params)
    pred_lists = extract_test_output_outputs(raw_outputs)

    # Evaluation
    samples = []
    for item in benchmark:
        test_data = json.loads(item["test"])
        samples.append(
            {
                "input": item["question_content"],
                "output": test_data[0]["output"],
            }
        )
    k_list = [k for k in [1, 5] if k <= args.n]
    metrics, eval_results = test_output_metrics(
        samples, pred_lists, k_list=k_list
    )

    print(f"\nTest Output Prediction metrics: {metrics}")

    save_test_output_results(
        benchmark,
        raw_outputs,
        pred_lists,
        metrics,
        eval_results,
        "testoutputprediction",
        args.n,
        args.temperature,
        args,
    )


# ===================================================================
# Main
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="LiveCodeBench evaluation (official-aligned)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--scenario",
        type=str,
        default="codegeneration",
        choices=[
            "codegeneration",
            "selfrepair",
            "codeexecution",
            "testoutputprediction",
            "all",
        ],
    )

    parser.add_argument("--release_version", type=str, default="release_latest")
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--not_fast", action="store_true")

    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--codegen_n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--stop", type=str, default="")

    parser.add_argument("--tensor_parallel_size", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--cot_code_execution", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument("--num_process_evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--codegen_eval_all_path", type=str, default=None)
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help=(
            "Hugging Face cache root used for datasets and hub downloads. "
            "Defaults to opencompass/data/hf_cache_livecodebench_lite if it "
            "already exists."
        ),
    )
    parser.add_argument(
        "--hf_online",
        action="store_true",
        help=(
            "Allow Hugging Face network access. By default the script uses the "
            "local LiveCodeBench cache in offline mode when available."
        ),
    )

    return parser.parse_args()


def validate_args(args):
    """Validate argument constraints."""
    if args.scenario == "selfrepair":
        if args.n != 1:
            print("WARNING: selfrepair requires n=1, forcing n=1")
            args.n = 1

    if args.scenario in ("codeexecution", "testoutputprediction"):
        if args.not_fast:
            print(
                f"WARNING: --not_fast only applies to codegeneration, "
                f"ignoring for {args.scenario}"
            )
        if args.start_date or args.end_date:
            print(
                f"WARNING: --start_date/--end_date only apply to "
                f"codegeneration/selfrepair, ignoring for {args.scenario}"
            )


def main():
    args = parse_args()
    validate_args(args)
    cache_dir = configure_hf_cache(args)

    model_name = args.model_name or Path(args.model_path).name
    args.model_name = model_name

    print(f"Model: {args.model_path}")
    print(f"Model name: {model_name}")
    print(f"Scenario: {args.scenario}")
    print(f"Output dir: {get_output_dir(args)}")
    if cache_dir:
        print(f"HF cache dir: {cache_dir}")

    # ---- Tokenizer ----
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # ---- vLLM ----
    from vllm import LLM, SamplingParams
    import torch

    tp_size = (
        torch.cuda.device_count()
        if args.tensor_parallel_size == "auto"
        else int(args.tensor_parallel_size)
    )

    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=tp_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    stop_list = [s for s in args.stop.split(",") if s]

    def make_sampling_params(n):
        sp_kwargs = dict(
            n=n,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_list,
        )
        if args.top_k > 0:
            sp_kwargs["top_k"] = args.top_k
        return SamplingParams(**sp_kwargs)

    # ---- Run scenarios ----
    if args.scenario == "all":
        # 1. Code Generation
        sp = make_sampling_params(args.n)
        run_codegeneration(llm, tokenizer, args, sp)

        # 2. Self Repair (n=1, consumes codegen eval_all)
        sp_repair = make_sampling_params(1)
        orig_n = args.n
        args.n = 1
        args.codegen_n = orig_n
        run_selfrepair(llm, tokenizer, args, sp_repair)
        args.n = orig_n

        # 3. Code Execution
        sp = make_sampling_params(args.n)
        run_codeexecution(llm, tokenizer, args, sp)

        # 4. Test Output Prediction
        run_testoutputprediction(llm, tokenizer, args, sp)

    elif args.scenario == "codegeneration":
        sp = make_sampling_params(args.n)
        run_codegeneration(llm, tokenizer, args, sp)

    elif args.scenario == "selfrepair":
        sp = make_sampling_params(1)
        run_selfrepair(llm, tokenizer, args, sp)

    elif args.scenario == "codeexecution":
        sp = make_sampling_params(args.n)
        run_codeexecution(llm, tokenizer, args, sp)

    elif args.scenario == "testoutputprediction":
        sp = make_sampling_params(args.n)
        run_testoutputprediction(llm, tokenizer, args, sp)

    print("\nDone!")


if __name__ == "__main__":
    main()
