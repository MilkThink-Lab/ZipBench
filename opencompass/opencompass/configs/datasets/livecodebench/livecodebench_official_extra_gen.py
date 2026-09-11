"""LiveCodeBench extra scenarios (codeexecution / testoutputprediction).

Companion to ``livecodebench_official_gen.py`` (codegeneration): these two
scenarios use OpenCompass's existing LCB components — the prompt/reader/infer
structure mirrors ``livecodebench_gen_b2b0fd.py`` (already aligned with the
upstream lcb_runner prompts) — wired into the config flow so any OpenCompass
``models`` entry (API or local) can run them. Full-set evaluation only, n=1.

``max_out_len`` belongs to the model config, not the dataset, so no
inferencer-level value is set here.
"""

from opencompass.datasets import (LCBCodeExecutionDataset,
                                  LCBCodeExecutionEvaluator,
                                  LCBTestOutputEvaluator,
                                  LCBTestOutputPredictionDataset)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# ---- codeexecution ----
lcb_code_execution_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='evaluation_sample',
)

lcb_code_execution_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM',
                     fallback_role='HUMAN',
                     prompt=('You are an expert at Python programming, code '
                             'execution, test case generation, and fuzzing.')),
            ],
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

lcb_code_execution_eval_cfg = dict(
    evaluator=dict(
        type=LCBCodeExecutionEvaluator,
        # Set cot=True when using the chain-of-thought execution prompt
        # variant (the extractor then reads the '[ANSWER]' span).
        cot=False,
    ),
    pred_role='BOT',
)

LCBCodeExecution_dataset = dict(
    type=LCBCodeExecutionDataset,
    abbr='lcb_code_execution',
    path='opencompass/execution-v2',
    reader_cfg=lcb_code_execution_reader_cfg,
    infer_cfg=lcb_code_execution_infer_cfg,
    eval_cfg=lcb_code_execution_eval_cfg,
)

# ---- testoutputprediction ----
lcb_test_output_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='evaluation_sample',
)

lcb_test_output_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{prompt}'),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

lcb_test_output_eval_cfg = dict(
    evaluator=dict(type=LCBTestOutputEvaluator),
    pred_role='BOT',
)

LCBTestOutput_dataset = dict(
    type=LCBTestOutputPredictionDataset,
    abbr='lcb_test_output',
    path='opencompass/test_generation',
    reader_cfg=lcb_test_output_reader_cfg,
    infer_cfg=lcb_test_output_infer_cfg,
    eval_cfg=lcb_test_output_eval_cfg,
)

LCB_official_extra_datasets = [
    LCBCodeExecution_dataset,
    LCBTestOutput_dataset,
]
