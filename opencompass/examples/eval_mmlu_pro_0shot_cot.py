from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # 0-shot CoT (openai simple-evals style), the setting behind the
    # OpenCompass alias mmlu_pro_gen: the model ends with
    # "ANSWER: $LETTER", extracted via (?i)ANSWER\s*:\s*([A-P]); single
    # pass + AccEvaluator, 14 subjects as mmlu_pro_* subsets.
    # Note: this differs from the official TIGER-AI-Lab 5-shot CoT setting
    # (same-category 5-shot, subject preamble, tiered A-J extraction,
    # random guess on no answer); the OpenCompass 0-shot default is chosen
    # deliberately for consistency with examples/eval_mmlu.py.
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import \
        mmlu_pro_datasets  # noqa: F401
    # 14 subjects -> overall mmlu_pro score (naive_average)
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups  # noqa: F401
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

datasets = mmlu_pro_datasets
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

# Aggregate the 14 mmlu_pro_* subsets into the overall mmlu_pro score
# (naive_average) plus per-subject rows.
summarizer = dict(
    dataset_abbrs=[
        'mmlu_pro',
        'mmlu_pro_biology',
        'mmlu_pro_business',
        'mmlu_pro_chemistry',
        'mmlu_pro_computer_science',
        'mmlu_pro_economics',
        'mmlu_pro_engineering',
        'mmlu_pro_health',
        'mmlu_pro_history',
        'mmlu_pro_law',
        'mmlu_pro_math',
        'mmlu_pro_philosophy',
        'mmlu_pro_physics',
        'mmlu_pro_psychology',
        'mmlu_pro_other',
    ],
    summary_groups=mmlu_pro_summary_groups,
)

work_dir = './outputs/default/mmlu_pro'
