# ./examples/eval_arc_challenge.py
#
# ARC-Challenge evaluation (0-shot gen, official test split, all 1172
# questions).
#
# - The official evaluation (Clark et al. 2018 / lm-eval-harness) uses the
#   test split (1172 questions); the stock OpenCompass ARC_c_gen configs
#   default to the dev split (299 questions), so this script uses
#   'opencompass/ai2_arc-test' instead.
# - ARCDatasetAllChoices keeps every question (the stock ARCDataset drops the
#   3-choice and 5-choice items); options are rendered dynamically as
#   options_str and answer extraction covers ABCDE.
# - Numeric labels (1-4) are mapped to letters by position; the metric is
#   accuracy, matching the official setting.

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDatasetAllChoices
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

ARC_c_reader_cfg = dict(
    input_columns=['question', 'options_str'],
    output_column='answerKey')

ARC_c_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Question: {question}\n{options_str}\nAnswer:'
                ),
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ARC_c_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCDE'),
)

arc_c_datasets = [
    dict(
        abbr='ARC-c-test',
        type=ARCDatasetAllChoices,
        path='opencompass/ai2_arc-test',
        name='ARC-Challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,
        eval_cfg=ARC_c_eval_cfg,
    )
]

datasets = arc_c_datasets

models = qwen3_30b_think_model

for model in models:
    model['max_out_len'] = 2048
    model['generation_kwargs']['temperature'] = 0

# -------------Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
work_dir = 'outputs/default/arc_challenge'
