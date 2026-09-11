# MATH-Hard (MATH Level-5, Open LLM Leaderboard v2 setting; 1324 items).
#
# NOTE: this is NOT the 5000-item OpenCompass MATH (opencompass/math). It loads
# data/math_hard/math_hard.json -- the 1324 Level-5 problems in the zipbench
# correctness-matrix row order (alphabetical concatenation of the 7 subject
# blocks) -- via MATHDataset + file_name='math_hard.json'. Prompt/judging mirror
# math_0shot_gen_393424 (0-shot boxed + MATHEvaluator v2). Used as the full-set
# base for the zipbench MATH subsets.
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2

math_hard_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_hard_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096),
)

# postprocess v2 (same as the full 5000-item MATH 0-shot config)
math_hard_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
)

math_hard_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_hard',
        path='opencompass/math_hard',
        file_name='math_hard.json',
        reader_cfg=math_hard_reader_cfg,
        infer_cfg=math_hard_infer_cfg,
        eval_cfg=math_hard_eval_cfg,
    )
]
