# ./examples/eval_gsm8k.py
#
# GSM8K evaluation (4-shot CoT gen, official test split, all 1319 questions).
#
# - Gold answers are taken after '#### ' with commas removed, matching the
#   official extraction.
# - Prediction extraction takes the last number in the output (after
#   truncating at 'Question:'), which is more lenient than the official
#   "The answer is x" extraction: outputs without the canonical closing
#   sentence still yield a number instead of being scored invalid.
# - Scoring is numeric equality with 1e-6 tolerance (slightly more lenient
#   than the official exact string match, e.g. "140.0" counts as correct);
#   the metric is accuracy.
# - The default gsm8k_gen (= gsm8k_gen_1d7fe4) prompt is 4-shot CoT ending
#   in "The answer is N"; it differs from the common 8-shot Wei et al.
#   prompt (lm-eval-harness / Llama papers) in shot count, exemplars, and
#   format, so scores are not strictly comparable across the two. For a
#   0-shot \boxed{} setting for reasoning models, import
#   gsm8k_0shot_v2_gen_a58960 (MATHEvaluator v2) instead.

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import \
        gsm8k_datasets
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import \
        models as qwen3_4b_instruct_model
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_3b_instruct import \
        models as qwen2_5_3b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as qwen3_30b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import \
        models as qwen3_30b_think_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_32b import \
        models as deepseek_r1_32b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import \
        models as deepseek_r1_14b_model

datasets = gsm8k_datasets

# The inferencer-level max_out_len=512 in the dataset config overrides the
# model-side setting and is too small for think/reasoning models; relax it
# here.
for d in datasets:
    d['infer_cfg']['inferencer']['max_out_len'] = 2048

models = qwen3_30b_think_model

for model in models:
    model['max_out_len'] = 2048
    model['generation_kwargs']['temperature'] = 0

# -------------Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
work_dir = 'outputs/default/gsm8k'
