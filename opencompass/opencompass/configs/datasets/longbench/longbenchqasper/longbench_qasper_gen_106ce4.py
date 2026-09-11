# Aligned with the official THUDM/LongBench evaluation setup.
# The stock OpenCompass config mistakenly used hotpotqa's multi-doc QA
# prompt (same hash 6b3efc) and copied its max_out_len 32 along with it.
# This file restores the original settings from the official
# config/dataset2prompt.json and dataset2maxlen.json.
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchF1Evaluator, LongBenchqasperDataset

LongBench_qasper_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_qasper_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128),
)

LongBench_qasper_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator), pred_role='BOT'
)

LongBench_qasper_datasets = [
    dict(
        type=LongBenchqasperDataset,
        abbr='LongBench_qasper',
        path='opencompass/Longbench',
        name='qasper',
        reader_cfg=LongBench_qasper_reader_cfg,
        infer_cfg=LongBench_qasper_infer_cfg,
        eval_cfg=LongBench_qasper_eval_cfg,
    )
]
