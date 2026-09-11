"""
Evaluation-only config for SciBench API model predictions.

Usage:
    cd /path/to/opencompass
    python run.py examples/eval_scibench_api.py --mode eval -r api_model

This config automatically discovers model prediction folders under
outputs/default/scibench/api_model/predictions/ and creates minimal
"dummy" model configs (only `abbr` is needed) so that OpenCompass can
run evaluation without requiring actual model weights or API configs.
"""

from mmengine.config import read_base

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NumericAccEvaluator
from opencompass.datasets import ScibenchDataset_Local, scibench_postprocess
from opencompass.summarizers import ScibenchSummarizer

with read_base():
    from opencompass.configs.summarizers.groups.scibench import scibench_summary_groups

# ---------------------------------------------------------------------------
# 1. API model prediction folders (under outputs/default/scibench/api_model/predictions/)
#    Only `abbr` is needed – it tells the evaluator which prediction subfolder to read.
#    To add a new model, simply add its folder name to this list.
# ---------------------------------------------------------------------------
api_model_abbrs = [
    # 'deepseek-v3.2',
    # 'gemini-3-pro',
    # 'gpt-5.2',
    # 'gpt-5.2-high',
    # 'grok-4.1-fast',
    # 'mimo-v2-flash',
    # 'minimax-m2.1',
    # 'qwen3-235b-a22b-instruct-2507',
    'gemini-3-flash',
]

models = [dict(abbr=name) for name in api_model_abbrs]

# ---------------------------------------------------------------------------
# 2. SciBench dataset config (identical to eval_scibench.py)
# ---------------------------------------------------------------------------
scibench_subsets = [
    'atkins',      # physical chemistry
    'calculus',    # calculus
    'chemmc',      # quantum chemistry
    'class',       # classical mechanics
    'diff',        # differential equations
    'fund',        # fundamentals of physics
    'matter',      # matter science
    'quan',        # quantum mechanics
    'stat',        # statistics
    'thermo',      # thermodynamics
]

concise_cot_prompt = (
    "Let's think step by step, but focus only on the key logic needed "
    "to reach the answer — no extra details or repetition."
)

scibench_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)

scibench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=(
                    "Please provide a clear and step-by-step solution for a "
                    "scientific problem in the categories of Chemistry, Physics, "
                    "or Mathematics. The problem will specify the unit of "
                    "measurement, which should not be included in the answer. "
                    "Express the final answer as a decimal number with three "
                    "digits after the decimal point. Conclude the answer by "
                    "stating 'Therefore, the answer is \\boxed[ANSWER].'\n\n"
                    "Problem: {question}\nAnswer: " + concise_cot_prompt
                ),
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)

scibench_eval_cfg = dict(
    evaluator=dict(
        type=NumericAccEvaluator,
        rtol=1e-2,   # relative tolerance 1%
        atol=1e-3,   # absolute tolerance 0.001
    ),
    pred_postprocessor=dict(type=scibench_postprocess),
)

datasets = []
for _name in scibench_subsets:
    datasets.append(
        dict(
            abbr=f'scibench-{_name}',
            type=ScibenchDataset_Local,
            path='./data/scibench',
            name=_name,
            reader_cfg=scibench_reader_cfg,
            infer_cfg=scibench_infer_cfg,
            eval_cfg=scibench_eval_cfg,
        )
    )

# ---------------------------------------------------------------------------
# 3. Work dir – must match the parent of api_model/predictions/
# ---------------------------------------------------------------------------
work_dir = 'outputs/default/scibench'

# ---------------------------------------------------------------------------
# 4. Summarizer (same as eval_scibench.py)
# ---------------------------------------------------------------------------
summarizer = dict(
    type=ScibenchSummarizer,
    scibench_subsets=scibench_subsets,
    dataset_abbrs=[
        ['scibench-weighted', 'weighted_average'],
        ['scibench', 'naive_average'],
        'scibench-atkins',
        'scibench-calculus',
        'scibench-chemmc',
        'scibench-class',
        'scibench-diff',
        'scibench-fund',
        'scibench-matter',
        'scibench-quan',
        'scibench-stat',
        'scibench-thermo',
    ],
    summary_groups=scibench_summary_groups,
)
