# TruthfulQA -- local variant (zero-shot generation + purely local automatic metrics)
#
# Protocol notes (follows the existing OpenCompass zero-shot protocol; NOT comparable
# with the official numbers):
# - Task: only the official "generation" task, 817 questions from the validation split.
#   The multiple-choice tasks (MC1/MC2/MC3) are not included.
# - Prompt: zero-shot bare question (single HUMAN turn). It does NOT use the official
#   6-shot QA_PRIMER of the `qa` preset, nor `stop='\n\n'`, nor A:..Q: answer
#   truncation. Numbers produced by this config are therefore NOT directly comparable
#   with the TruthfulQA paper / leaderboard.
# - Metrics: local automatic metrics BLEU / ROUGE (no judge model download, no GPU,
#   CPU-only evaluation). Each metric reports three sub-scores:
#     * <m>_max : max similarity against the correct reference answers
#     * <m>_diff: max(correct) - max(incorrect); the official recommended headline score
#     * <m>_acc : fraction of samples with diff > 0 (answers closer to truthful than
#                 to false references); the most readable of the three
#   "I have no comment." is added to the correct-answer set at evaluation time, as in
#   the official implementation.
#
# To get closer to the official "truthfulness" judgement, switch the metrics below to
# the judge-based protocol (still local inference, but it downloads ~7B judge models
# and needs GPU memory):
#   metrics=('truth', 'info')   # allenai/truthfulqa-{truth,info}-judge-llama2-7B
# A single metric also works: metrics='truth' / metrics='info' / metrics='rouge'.
# (Running several metrics at once relies on the metrics normalisation fix in
# opencompass/datasets/truthfulqa.py.)

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TruthfulQADataset, TruthfulQAEvaluator

truthfulqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='reference',
    train_split='validation',
    test_split='validation')

truthfulqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{question}')])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

truthfulqa_eval_cfg = dict(
    evaluator=dict(type=TruthfulQAEvaluator, metrics=('bleu', 'rouge')), )

truthfulqa_datasets = [
    dict(
        abbr='truthful_qa',
        type=TruthfulQADataset,
        path='truthful_qa',
        name='generation',
        reader_cfg=truthfulqa_reader_cfg,
        infer_cfg=truthfulqa_infer_cfg,
        eval_cfg=truthfulqa_eval_cfg)
]
