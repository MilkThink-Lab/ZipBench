from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # Official setting: the 541 prompts are fed verbatim, zero-shot (no
    # system prompt / no few-shot); the checker follows
    # google-research/instruction_following_eval.
    # 353ae7 is used instead of 3321a3: the latter hardcodes
    # max_out_len=1025 on the inferencer, while the number_words
    # instructions require up to "at least 1200 words" (~1600+ tokens) and
    # would be truncated.
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
        ifeval_datasets
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import \
        models as qwen3_4b_instruct_model
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_3b_instruct import \
        models as qwen2_5_3b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as qwen3_30b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import \
        models as qwen3_30b_think_model
    from opencompass.configs.models.qwen3.vllm_qwen3_5_27b import \
        models as qwen3_5_27b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_32b import \
        models as deepseek_r1_32b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import \
        models as deepseek_r1_14b_model

datasets = ifeval_datasets
models = qwen3_30b_think_model

# IFEval checks the surface form of the answer itself (comma count, all
# lowercase, word count, quoting, JSON format, ...); if a chain of thought
# is left in the prediction it goes into the checker too and fails almost
# everything. Strip <think>...</think> before scoring; this is a no-op for
# non-thinking models. Attached at the evaluator level, not the dataset
# level, so predictions/*.json keep the full CoT and only the scoring copy
# is stripped.
for ds in datasets:
    ds['eval_cfg']['evaluator']['pred_postprocessor'] = dict(
        type='extract-non-reasoning-content')

for model in models:
    # Must fit CoT + answer body; the longest instruction requires 1200
    # words (~1600+ tokens) and truncation fails the item outright.
    model['max_out_len'] = 8192
    model['generation_kwargs']['temperature'] = 0

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/default/ifeval'
