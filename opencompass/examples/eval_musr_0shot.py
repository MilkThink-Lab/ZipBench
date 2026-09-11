from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # Official setting: cot+ prompt, 0-shot, self_consistency_n=1 (the
    # default ablation in the official repo's eval.py).
    from opencompass.configs.datasets.musr.musr_gen_b47fd3 import \
        musr_datasets
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
    from opencompass.configs.summarizers.groups.musr_average import \
        summarizer

datasets = musr_datasets
models = qwen3_30b_think_model

# Match the official MuSR eval: take the LAST "ANSWER:" line, not the first
# (MusrEvaluator defaults to the first, but think models often write
# "ANSWER:" mid-reasoning, which grabs an intermediate conclusion).
# Attached via the evaluator's own pred_postprocessor: only the MuSR scoring
# copy is affected; the full CoT saved to details is untouched. Matching
# stays case-sensitive on "ANSWER:".
for ds in datasets:
    ds['eval_cfg']['evaluator']['pred_postprocessor'] = dict(
        type='musr_answer_last_line')

for model in models:
    # cot+ asks for step-by-step reasoning before "ANSWER: n";
    # murder_mysteries chains are long (the official gpt-4 run used
    # max_tokens=2400), so use 2048 to avoid truncation.
    model['max_out_len'] = 2048
    model['generation_kwargs']['temperature'] = 0

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/default/musr_0shot'
