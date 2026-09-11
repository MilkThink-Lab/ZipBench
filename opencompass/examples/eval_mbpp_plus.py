# MBPP+ with the OpenCompass 3-shot [BEGIN]/[DONE] prompt
# (mbpp_plus_gen_0b836a), scored by the evalplus base+plus harness.
from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.mbpp_plus.mbpp_plus_gen_0b836a import \
        mbpp_plus_datasets
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import \
        models as qwen3_4b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as qwen3_30b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import \
        models as qwen3_30b_think_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_32b import \
        models as deepseek_r1_32b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import \
        models as deepseek_r1_14b_model

datasets = mbpp_plus_datasets
models = qwen3_4b_instruct_model

# the dataset config ships max_out_len=300, far too small for CoT/thinking
# models; raise it here (inferencer-level value takes precedence)
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['max_out_len'] = 2048

for model in models:
    model['max_out_len'] = 4096
    model['generation_kwargs'] = dict(temperature=0)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/default/mbpp_plus'
