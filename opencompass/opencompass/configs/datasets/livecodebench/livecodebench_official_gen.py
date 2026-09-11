"""LiveCodeBench codegeneration config using upstream lcb_runner.

This config wires the official LCB dataset + official eval (vendored under
`third_party/livecodebench/`) into OpenCompass's standard config flow, so any
model defined as an OpenCompass `models` entry (API or local) can be evaluated
against the official benchmark.

System message + prompt body mirror upstream `lcb_runner/prompts/code_generation.py`
for `LMStyle.OpenAIChat` (the chat-API path used by gpt/gemini/qwen/etc.).
"""

from opencompass.datasets import (LCBOfficialCodeGenerationDataset,
                                  LCBOfficialCodeGenerationEvaluator)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# Mirrors PromptConstants.SYSTEM_MESSAGE_GENERIC in
# third_party/livecodebench/lcb_runner/prompts/code_generation.py:14
SYSTEM_MESSAGE_GENERIC = (
    'You are an expert Python programmer. You will be given a question '
    '(problem specification) and will generate a correct Python program that '
    'matches the specification and passes all tests.')

# Mirrors get_generic_question_template_answer() body (L40-L51 of upstream).
# {format_prompt} is precomputed in LCBOfficialCodeGenerationDataset rows.
prompt_template = ('### Question:\n{question_content}\n\n'
                   '{format_prompt}'
                   '### Answer: (use the provided format with backticks)\n\n')

# Adjust these to evaluate a different release window. Upstream default is
# `release_latest`, but we pin to `release_v6` because as of 2026-05-16 v6 is
# the latest published release and pinning prevents an accidental cache rebuild
# (and silent dataset drift) when upstream eventually ships v7.
RELEASE_VERSION = 'release_v6'
START_DATE = None
END_DATE = None

lcb_code_generation_reader_cfg = dict(
    input_columns=['question_content', 'format_prompt'],
    output_column='question_id',
)

lcb_code_generation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM',
                     fallback_role='HUMAN',
                     prompt=SYSTEM_MESSAGE_GENERIC),
            ],
            round=[
                dict(role='HUMAN', prompt=prompt_template),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

lcb_code_generation_eval_cfg = dict(
    evaluator=dict(
        type=LCBOfficialCodeGenerationEvaluator,
        release_version=RELEASE_VERSION,
        start_date=START_DATE,
        end_date=END_DATE,
        # k_list is derived from the dataset-level `n` inside evaluate()
        # (pass@1/5/10 filtered to k <= n, pass@n fallback when 1 < n < 5);
        # the k_list ctor arg only matters for direct score() calls.
        num_process_evaluate=12,
        timeout=6,
        lm_style='OpenAIChat',
    ),
    pred_role='BOT',
)

LCBOfficialCodeGeneration_dataset = dict(
    type=LCBOfficialCodeGenerationDataset,
    abbr='lcb_code_generation_official',
    path='livecodebench/code_generation_lite',
    release_version=RELEASE_VERSION,
    start_date=START_DATE,
    end_date=END_DATE,
    reader_cfg=lcb_code_generation_reader_cfg,
    infer_cfg=lcb_code_generation_infer_cfg,
    eval_cfg=lcb_code_generation_eval_cfg,
)

LCB_official_datasets = [LCBOfficialCodeGeneration_dataset]
