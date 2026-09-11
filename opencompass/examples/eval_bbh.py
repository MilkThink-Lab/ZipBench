from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # 24-task subset aligned with the Open LLM Leaderboard v2 (drops
    # dyck_languages / multistep_arithmetic_two / word_sorting). Official
    # 3-shot CoT prompts in a single HUMAN turn, max_out_len=512.
    from opencompass.configs.datasets.bbh.bbh_leaderboard24_gen import \
        bbh_datasets
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

datasets = bbh_datasets
models = qwen2_5_3b_instruct_model

for model in models:
    model['generation_kwargs']['temperature'] = 0

# Average over the 24 leaderboard subtasks. Unweighted mean of per-task
# accuracies; the leaderboard reports a size-weighted mean of acc_norm, so
# treat this as the same task set, not a bit-exact score reproduction.
bbh_summary_groups = [
    dict(name='bbh', subsets=[d['abbr'] for d in bbh_datasets])
]
summarizer = dict(
    dataset_abbrs=['bbh'],
    summary_groups=bbh_summary_groups,
)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/default/bbh'
