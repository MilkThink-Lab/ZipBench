from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import C3Dataset_V2
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
concise_cot_prompt = "Let's think step by step, but focus only on the key logic needed to reach the answer — no extra details or repetition."
cot_prompt = " Let's think step by step."
direct_prompt = "Don't think step by step, just give the answer directly."
deep_cot_prompt = "Let's think in layered depth, explicitly stating key assumptions and decomposing the problem into sub-questions before solving. After each major step, pause to challenge your own logic (possible mistakes, hidden assumptions, edge cases, and alternative paths), revising if needed. Only then deliver a concise final answer that is robust, self-consistent, and clear about any remaining uncertainty."
deep_cot_cn = "让我们进行分层的深度思考：先明确陈述关键假设，并在求解之前把问题拆解成若干子问题。每完成一个主要步骤，就暂停并质疑自己的推理（可能的错误、隐藏的假设、边界情况，以及其他可选路径），如有必要就修正。只有在完成这些检验后，再给出一个简洁的最终答案；这个答案应当稳健、自洽，并清楚说明仍然存在的不确定性。"
# Local dataset config.
C3_reader_cfg = dict(
    input_columns=[
        'question',
        'content',
        'choice0',
        'choice1',
        'choice2',
        'choice3',
        'choices',
    ],
    output_column='label',
)

C3_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                '{content}\n问：{question}\nA. {choice0}\nB. {choice1}\nC. {choice2}\nD. {choice3}\n请从"A"，"B"，"C"，"D"中进行选择。\n答： ' + cot_prompt,
            ),
            # dict(
            #     role='BOT',
            #     prompt = "</think>\n")
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

C3_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

C3_datasets = [
    dict(
        abbr='C3',
        type=C3Dataset_V2,
        path='./data/CLUE/C3/dev_0.json',  # local data path
        reader_cfg=C3_reader_cfg,
        infer_cfg=C3_infer_cfg,
        eval_cfg=C3_eval_cfg,
    )
]

datasets = C3_datasets

# Concatenate model lists to evaluate several models in one pass.
# all_models = (qwen2_5_3b_instruct_model + qwen3_4b_instruct_model + qwen3_30b_instruct_model + qwen3_30b_think_model)
all_models = (qwen3_4b_instruct_model)

# Shared model parameters:
# for model in all_models:
#     model['max_out_len'] =  2048
#     model['generation_kwargs']['temperature'] = 0

models = all_models

# -------------Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
work_dir = 'outputs/default/C3'