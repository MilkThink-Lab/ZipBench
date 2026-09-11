from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # openai simple-evals style 0-shot CoT prompt (answers extracted from
    # "ANSWER: $LETTER"); single pass, deterministic cyclic option shuffle.
    from opencompass.configs.datasets.gpqa.gpqa_gen import \
        gpqa_datasets  # noqa: F401
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

datasets = gpqa_datasets
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

work_dir = './outputs/default/gpqa'
