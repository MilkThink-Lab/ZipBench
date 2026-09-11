from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (HumanevalDataset, HumanEvalPlusEvaluator,
                                  humaneval_postprocess_v2)

with read_base():
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as qwen3_30b_instruct_model

max_tokens = 2048
cot_prompt = "Let's think step by step."

humaneval_plus_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='task_id',
    train_split='test',
)

humaneval_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Complete the following python code:\n{prompt}\n' +
                cot_prompt,
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=max_tokens),
)

humaneval_plus_eval_cfg = dict(
    evaluator=dict(type=HumanEvalPlusEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_plus_datasets = [
    dict(
        abbr='humaneval_plus',
        type=HumanevalDataset,
        path='opencompass/humaneval',
        reader_cfg=humaneval_plus_reader_cfg,
        infer_cfg=humaneval_plus_infer_cfg,
        eval_cfg=humaneval_plus_eval_cfg,
    )
]

datasets = humaneval_plus_datasets
models = qwen3_30b_instruct_model

for model in models:
    model['generation_kwargs'] = dict(temperature=0)
    model['model_kwargs']['tensor_parallel_size'] = 2
    model['run_cfg']['num_gpus'] = 2

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/default/humaneval_plus'