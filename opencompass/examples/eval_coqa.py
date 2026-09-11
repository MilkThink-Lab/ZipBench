from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import commonsenseqaDataset_Local
from opencompass.utils.text_postprocessors import first_option_postprocess

with read_base():
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

# Local dataset config.
commonsenseqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D', 'E'],
    output_column='answerKey',
    test_split='validation',
    # test_range='[0:20]'  # evaluate only the first 20 questions
)
cot_prompt = "Let's think step by step. "
direct_prompt = "Don't think step by step, just give the answer directly."
deep_cot_prompt = "Think in layered depth, explicitly stating key assumptions and decomposing the problem into sub-questions before solving. After each major step, pause to challenge your own logic (possible mistakes, hidden assumptions, edge cases, and alternative paths), revising if needed. Only then deliver a concise final answer that is robust, self-consistent, and clear about any remaining uncertainty."
# concise_cot_prompt_2 = "Let's think step by step, but keep reasoning minimal. Focus only on the key logic needed to reach the answer — no extra details or repetition."
concise_cot_prompt = "Let's think step by step, but focus only on the key logic needed to reach the answer — no extra details or repetition."
_ice_template = dict(
    type=PromptTemplate,
    template=dict(
        begin='</E>',
        round=[
            dict(
                role='HUMAN',
                prompt=
                '{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nAnswer: ' + cot_prompt
            ),
            dict(
                role='BOT',
                prompt='{answerKey}',
            ),
        ],
    ),
    ice_token='</E>',
)

commonsenseqa_infer_cfg = dict(
    prompt_template=_ice_template,
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

commonsenseqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCDE'),
)

commonsenseqa_datasets = [
    dict(
        abbr='commonsense_qa',
        type=commonsenseqaDataset_Local,
        path='./data/commonsenseqa',
        reader_cfg=commonsenseqa_reader_cfg,
        infer_cfg=commonsenseqa_infer_cfg,
        eval_cfg=commonsenseqa_eval_cfg,
    )
]

datasets = commonsenseqa_datasets

# Concatenate model lists to evaluate several models in one pass.
all_models = (
    qwen3_30b_think_model
)

# Apply shared model parameters.
for model in all_models:
    model['max_out_len'] = 900
    model['generation_kwargs']['temperature'] = 0

models = all_models

# -------------Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
work_dir = 'outputs/default/coqa'
