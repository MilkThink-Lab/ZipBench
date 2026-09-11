# ./examples/eval_math.py
#
# MATH evaluation (0-shot boxed gen, official test split, all 5000 problems;
# Hendrycks et al. 2021).
#
# - Gold answers come from the last \boxed{} of the official solutions.
# - Prediction extraction (math_postprocess_v2) takes the last \boxed{} like
#   the official code; when no box is present it falls back to a
#   "final answer / answer is" sentence instead of scoring invalid, i.e.
#   slightly more lenient.
# - Scoring: MATHEvaluator v2 — the official math_equivalence kernel plus
#   extra normalization (\text/%/\cdot removal, trailing zeros, Minerva
#   normalize_final_answer fallback); it only rescues equivalent answer
#   forms the official matcher would miss. Metric is accuracy.
# - Prompt: the official paper evaluated fine-tuned models and defines no
#   prompt; the 0-shot "reason step by step ... \boxed{}" chat prompt here
#   is the common convention. For base models, use math_4shot_base_gen_db136b
#   (Minerva 4-shot) instead.

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import \
        math_datasets
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

datasets = math_datasets

# The inferencer-level max_out_len=1024 in the dataset config overrides the
# model-side setting and is too small for think/reasoning models; relax it
# here.
for d in datasets:
    d['infer_cfg']['inferencer']['max_out_len'] = 4096

models = qwen3_30b_think_model

for model in models:
    model['max_out_len'] = 4096
    model['generation_kwargs']['temperature'] = 0

# -------------Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
work_dir = 'outputs/default/math'
