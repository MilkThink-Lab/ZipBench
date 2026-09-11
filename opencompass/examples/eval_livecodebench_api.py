# LiveCodeBench official evaluation — unified config-flow entry point.
# Swap the model config (API or local vLLM) to switch models; the dataset /
# eval wiring stays the same.
#
# Dataset + eval logic come from upstream LiveCodeBench (cloned under
# third_party/livecodebench/, pinned at 28fef95), exposed via
# LCBOfficialCodeGenerationDataset / LCBOfficialCodeGenerationEvaluator.
# System message and prompt body match upstream
# `lcb_runner/prompts/code_generation.py` for the OpenAIChat style.
#
# Scenarios: codegeneration always runs; set RUN_EXTRA_SCENARIOS = True to add
# codeexecution + testoutputprediction (full-set, n=1; see
# opencompass/configs/datasets/livecodebench/livecodebench_official_extra_gen.py).
#
# Run:
#     python run.py examples/eval_livecodebench_api.py
#
# Switch model: the default model is defined inline below via the generic
# OpenAISDK class (any OpenAI-compatible endpoint). Set OC_API_BASE /
# OC_API_KEY / OC_MODEL_NAME env vars, or edit the dict. For a local vLLM
# model, use the commented block further down instead.
#
# Sampling protocol: the official LiveCodeBench standard is n=10 samples per
# problem, reporting pass@1 and pass@5. With N_SAMPLES = 10 (default),
# OpenCompass repeats the test set 10x at inference (10x API spend) and the
# evaluator groups the samples per question, reporting the official
# "pass@1" / "pass@5" (and "pass@10") directly. Set N_SAMPLES = 1 for a
# cheaper single-sample smoke run (pass@1 only); 1 < n < 5 falls back to
# pass@1 + pass@n.

import os
from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.livecodebench.livecodebench_official_extra_gen import \
        LCB_official_extra_datasets
    from opencompass.configs.datasets.livecodebench.livecodebench_official_gen import \
        LCBOfficialCodeGeneration_dataset

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')])

# Local vLLM alternative: uncomment (and comment out inline_api_model below /
# point `all_models` at this) to evaluate a local model through the same
# dataset wiring. temperature / max_out_len belong to the model config.
# from opencompass.models import VLLMwithChatTemplate
# inline_vllm_model = [
#     dict(
#         type=VLLMwithChatTemplate,
#         abbr='qwen3-4b-instruct-vllm',
#         path=os.path.join(os.environ.get('OC_MODEL_ROOT', 'Qwen'),
#                           'Qwen3-4B-Instruct-2507'),
#         model_kwargs=dict(tensor_parallel_size=1,
#                           gpu_memory_utilization=0.9,
#                           max_model_len=262144),
#         max_out_len=32768,
#         batch_size=16,
#         # Official LCB generation temperature (lcb_runner default: 0.2).
#         generation_kwargs=dict(temperature=0.2),
#         run_cfg=dict(num_gpus=1),
#     ),
# ]

inline_api_model = [
    dict(
        abbr=os.environ.get('OC_MODEL_NAME', 'gpt-4o-2024-05-13'),
        type=OpenAISDK,
        path=os.environ.get('OC_MODEL_NAME', 'gpt-4o-2024-05-13'),
        key=os.environ.get('OC_API_KEY', 'ENV'),  # 'ENV' -> $OPENAI_API_KEY
        openai_api_base=os.environ.get(
            'OC_API_BASE', 'https://api.openai.com/v1'),
        meta_template=api_meta_template,
        query_per_second=1,
        max_seq_len=131072,
        # Official LCB generation temperature (lcb_runner default: 0.2).
        temperature=0.2,
    ),
]

N_SAMPLES = 10  # official protocol; set 1 for a cheap smoke run

codegen_dataset = deepcopy(LCBOfficialCodeGeneration_dataset)
codegen_dataset['n'] = N_SAMPLES
# k_list is derived from n inside the evaluator (pass@1/5/10 filtered to
# k <= n, with a pass@n fallback when 1 < n < 5), matching
# examples/eval_livecodebench.py — no dataset-level `k` needed.
# Only affects the Scenario.codegeneration_<n>_<temp>*.json dump filenames.
codegen_dataset['eval_cfg']['evaluator']['lcb_runner_temperature'] = 0.2

# Optional overrides for release_version / date range (uncomment as needed):
# RELEASE = 'release_v6'
# START = '2024-08-01'
# END = '2025-04-30'
# codegen_dataset['release_version'] = RELEASE
# codegen_dataset['start_date'] = START
# codegen_dataset['end_date'] = END
# codegen_dataset['eval_cfg']['evaluator']['release_version'] = RELEASE
# codegen_dataset['eval_cfg']['evaluator']['start_date'] = START
# codegen_dataset['eval_cfg']['evaluator']['end_date'] = END

datasets = [codegen_dataset]

# Extra scenarios (codeexecution + testoutputprediction), full-set n=1.
RUN_EXTRA_SCENARIOS = False
if RUN_EXTRA_SCENARIOS:
    datasets += LCB_official_extra_datasets

all_models = inline_api_model
for model in all_models:
    model['max_out_len'] = 32768
    model['retry'] = 2
    model['batch_size'] = 16

models = all_models

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=1,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/default/livecodebench_codegen_api'

# run.py re-dumps this config to a .py file and re-parses it; module-level
# imports / temp vars are not valid when serialised. Keep only the keys
# OpenCompass consumes so the dumped file stays valid Python.
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
