# MBPP+ with the official EvalPlus chat prompt (0-shot).
# Prompt follows evalplus' instruction/response-prefix chat template and shows
# the official docstring prompt (fixed assertion), so the assertion the model
# sees matches the semantics the EvalPlus harness tests. Scoring is the same
# evalplus base+plus pass@1 as mbpp_plus_gen_0b836a.
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MBPPEvaluator, MBPPPlusOfficialDataset

mbpp_plus_official_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id')

mbpp_plus_official_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=(
                        'Please provide a self-contained Python script that '
                        'solves the following problem in a markdown code '
                        'block:\n```\n{prompt}\n```\n'),
                ),
                dict(
                    role='BOT',
                    prompt=(
                        'Below is a Python script with a self-contained '
                        'function that solves the problem and passes '
                        'corresponding tests:\n```python\n'),
                ),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))

mbpp_plus_official_eval_cfg = dict(
    evaluator=dict(type=MBPPEvaluator, metric='MBPPPlus'), pred_role='BOT')

mbpp_plus_official_datasets = [
    dict(
        type=MBPPPlusOfficialDataset,
        abbr='mbpp_plus_official',
        path='./data/mbpp_plus/MbppPlus-v0.1.0.jsonl',
        reader_cfg=mbpp_plus_official_reader_cfg,
        infer_cfg=mbpp_plus_official_infer_cfg,
        eval_cfg=mbpp_plus_official_eval_cfg)
]
