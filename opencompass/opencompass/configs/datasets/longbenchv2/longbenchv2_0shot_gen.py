from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchv2Dataset, LongBenchv2Evaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

LongBenchv2_0shot_reader_cfg = dict(
    input_columns=['context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'difficulty', 'length'],
    output_column='answer',
)

concise_cot_prompt = "Let's think step by step, but focus only on the key logic needed to reach the answer — no extra details or repetition."
cot_prompt = " Let's think step by step."
direct_prompt = "Don't think step by step, just give the answer directly."
deep_cot_prompt = "Let's think in layered depth, explicitly stating key assumptions and decomposing the problem into sub-questions before solving. After each major step, pause to challenge your own logic (possible mistakes, hidden assumptions, edge cases, and alternative paths), revising if needed. Only then deliver a concise final answer that is robust, self-consistent, and clear about any remaining uncertainty."

LongBenchv2_0shot_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\nChoices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\nFormat your response as follows: "The correct answer is (insert answer here)".'+ deep_cot_prompt,
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBenchv2_0shot_eval_cfg = dict(
    evaluator=dict(type=LongBenchv2Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

LongBenchv2_0shot_datasets = [
    dict(
        type=LongBenchv2Dataset,
        abbr='LongBenchv2_0shot',
        path='opencompass/longbenchv2',
        reader_cfg=LongBenchv2_0shot_reader_cfg,
        infer_cfg=LongBenchv2_0shot_infer_cfg,
        eval_cfg=LongBenchv2_0shot_eval_cfg,
    )
]
