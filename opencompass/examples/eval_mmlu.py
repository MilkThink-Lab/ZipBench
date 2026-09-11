from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # openai simple-evals style 0-shot CoT prompt (the model ends with
    # "ANSWER: $LETTER", extracted via (?i)ANSWER\s*:\s*([A-D])); single
    # pass, generative scoring, 57 subjects as lukaemon_mmlu_* subsets.
    from opencompass.configs.datasets.mmlu.mmlu_openai_simple_evals_gen_b618ea import \
        mmlu_datasets  # noqa: F401
    # 57 subjects -> mmlu / mmlu-stem / mmlu-humanities /
    # mmlu-social-science / mmlu-other
    from opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups  # noqa: F401
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

datasets = mmlu_datasets
models = qwen3_30b_think_model

for model in models:
    # 0-shot CoT: the model reasons before "ANSWER: X"; a short cap
    # truncates the reasoning and breaks extraction. Think models need
    # >= 8k.
    model['max_out_len'] = 8192
    model['generation_kwargs']['temperature'] = 0

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

# Aggregate the 57 lukaemon_mmlu_* subsets into the overall score and the
# four category scores.
summarizer = dict(
    dataset_abbrs=[
        'mmlu',
        'mmlu-stem',
        'mmlu-humanities',
        'mmlu-social-science',
        'mmlu-other',
    ],
    summary_groups=mmlu_summary_groups,
)

work_dir = './outputs/default/mmlu'
