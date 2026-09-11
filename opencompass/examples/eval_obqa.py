# ./examples/eval_openbookqa.py


from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import OBQADataset_Local
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
direct_prompt = "Don't think step by step, just give the answer directly."
cot_prompt = "Let's think step by step."
concise_cot_prompt = "Think step by step, but keep focus only on the key logic needed to reach the answer — no extra details or repetition."
deep_cot_prompt = " Think in layered depth, explicitly stating key assumptions and decomposing the problem into sub-questions before solving. After each major step, pause to challenge your own logic (possible mistakes, hidden assumptions, edge cases, and alternative paths), revising if needed. Only then deliver a concise final answer that is robust, self-consistent, and clear about any remaining uncertainty."
# Dataset configured directly from local paths.
_input_columns = [
    ['question_stem', 'A', 'B', 'C', 'D'],
    ['question_stem', 'A', 'B', 'C', 'D', 'fact1'],
]
# _template = [
#     dict(
#         round=[
#             dict(
#                 role='HUMAN',
#                 prompt='Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: \n' 
#             ),
#         ], ),
#     dict(
#         round=[
#             dict(
#                 role='HUMAN',
#                 prompt='Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: \n' 
#             ),
#         ], ),
# ]

_template = [
    dict(
        round=[
            dict(
                role='HUMAN',
                prompt='Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'
            ),
            dict(
                role='BOT',
                prompt='<think>\n'
            )
        ], ),
    dict(
        round=[
            dict(
                role='HUMAN',
                prompt='Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'
            ),
            dict(
                role='BOT',
                prompt='<think>\n'
            )
        ], ),
]

obqa_datasets = [
    dict(
        abbr='openbookqa',
        type=OBQADataset_Local,
        path='./data/openbookqa/Main/test.jsonl',
        name='main',
    ),
    dict(
        abbr='openbookqa_fact',
        type=OBQADataset_Local,
        path='./data/openbookqa/Additional/test_complete.jsonl',
        name='additional',
    ),
]

for _i in range(2):
    obqa_reader_cfg = dict(
        input_columns=_input_columns[_i], output_column='answerKey')
    obqa_infer_cfg = dict(
        prompt_template=dict(type=PromptTemplate, template=_template[_i]),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    obqa_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    )

    obqa_datasets[_i]['reader_cfg'] = obqa_reader_cfg
    obqa_datasets[_i]['infer_cfg'] = obqa_infer_cfg
    obqa_datasets[_i]['eval_cfg'] = obqa_eval_cfg

datasets = obqa_datasets

# Concatenate model lists to evaluate several models in one pass.
all_models = (
    deepseek_r1_14b_model
)

# Apply shared model parameters.
for model in all_models:
    model['max_out_len'] = 2048
    model['generation_kwargs']['temperature'] = 0.001

models = all_models

# -------------Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
work_dir = 'outputs/default/obqa'