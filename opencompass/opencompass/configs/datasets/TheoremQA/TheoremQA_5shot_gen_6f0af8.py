from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TheoremQADatasetV3, TheoremQA_postprocess_v3, TheoremQAEvaluatorV3

with read_base():
    from .TheoremQA_few_shot_examples import examples

num_shot = 5
rounds = []
for index, (query, response) in enumerate(examples[:num_shot]):
    if index == 0:
        desc = 'You are supposed to provide a solution to a given problem.\n\n'
    else:
        desc = ''
    rounds += [
        dict(role='HUMAN', prompt=f'{desc}Problem:\n{query}\nSolution:'),
        dict(role='BOT', prompt=f'{response}')
    ]
cot_prompt = 'Let\'s think step by step.'
direct_prompt = "Don't think step by step, just give the answer directly."
concise_cot_prompt ="Think step by step, but keep your reasoning minimal. Focus only on the key logic needed to reach the answer — no extra details or repetition."
deep_cot_prompt = "Think in layered depth, explicitly stating key assumptions and decomposing the problem into sub-questions before solving. After each major step, pause to challenge your own logic (possible mistakes, hidden assumptions, edge cases, and alternative paths), revising if needed. Only then deliver a concise final answer that is robust, self-consistent, and clear about any remaining uncertainty."
# rounds += [dict(role='HUMAN', prompt='Problem:\n{Question}\nSolution:')]
rounds += [dict(role='HUMAN', prompt='Problem:\n{Question}\nSolution: '+ direct_prompt)]
# rounds += [dict(role='HUMAN', prompt='Problem:\n{Question}\nSolution: '+ concise_cot_prompt)]

TheoremQA_reader_cfg = dict(input_columns=['Question', 'Answer_type'], output_column='Answer', train_split='test', test_split='test')
## default max_out_len is 1024
TheoremQA_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=dict(round=rounds)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=3000, stopping_criteria=['Problem:', 'Problem']),
)

TheoremQA_eval_cfg = dict(
    evaluator=dict(type=TheoremQAEvaluatorV3),
    pred_postprocessor=dict(type=TheoremQA_postprocess_v3)
)

TheoremQA_datasets = [
    dict(
        abbr='TheoremQA',
        type=TheoremQADatasetV3,
        path='data/TheoremQA/theoremqa_test.json',
        reader_cfg=TheoremQA_reader_cfg,
        infer_cfg=TheoremQA_infer_cfg,
        eval_cfg=TheoremQA_eval_cfg,
    )
]
