# ./examples/eval_hellaswag_0shot.py
#
# HellaSwag evaluation (0-shot generative, official validation split, all
# 10042 questions). Kernel taken from hellaswag_gen_6faab5: ZeroRetriever +
# GenInferencer + AccEvaluator + first_option_postprocess; the model
# generates text, a letter A/B/C/D is extracted and compared with the label.
#
# Important caveat vs the official setting:
# - Official HellaSwag (Zellers et al. 2019 / lm-eval-harness) is a
#   likelihood-ranking task: score log P(ending|ctx) for the 4 endings, with
#   acc_norm (length-normalized) as the headline metric.
# - This script uses the OpenCompass generative setting instead, so scores
#   are NOT comparable to official acc_norm numbers; for the official
#   setting switch to a PPL/loglikelihood pipeline.
# - The generative setting is chosen because it runs directly on
#   VLLMwithChatTemplate (which implements generate only, no get_ppl).
#
# Data: path='opencompass/hellaswag' (validation set). HellaswagDataset_V2
# strips the activity_label prefix from ctx and maps the gold index to a
# letter label.

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HellaswagDataset_V2
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

# -------------Dataset (0-shot gen, from hellaswag_gen_6faab5) ---------
hellaswag_reader_cfg = dict(
    input_columns=['ctx', 'A', 'B', 'C', 'D'],
    output_column='label',
)

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=('{ctx}\nQuestion: Which ending makes the most sense?\n'
                        'A. {A}\nB. {B}\nC. {C}\nD. {D}\n'
                        "You may choose from 'A', 'B', 'C', 'D'.\n"
                        'Answer:'),
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=HellaswagDataset_V2,
        path='opencompass/hellaswag',
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg,
    )
]

datasets = hellaswag_datasets

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

work_dir = 'outputs/default/hellaswag_0shot'
