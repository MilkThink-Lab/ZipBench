from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NumericAccEvaluator
from opencompass.datasets import ScibenchDataset_Local, scibench_postprocess
from opencompass.summarizers import ScibenchSummarizer

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
    from opencompass.configs.summarizers.groups.scibench import scibench_summary_groups
direct_prompt = "Don't think step by step, just give the answer directly."
cot_prompt = "Let's think step by step."
concise_cot_prompt = "Let's think step by step, but focus only on the key logic needed to reach the answer — no extra details or repetition."
deep_cot_prompt = "Let's think in layered depth, explicitly stating key assumptions and decomposing the problem into sub-questions before solving. After each major step, pause to challenge your own logic (possible mistakes, hidden assumptions, edge cases, and alternative paths), revising if needed. Only then deliver a concise final answer that is robust, self-consistent, and clear about any remaining uncertainty."
prompt_prefix = None
# SciBench subsets.
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
    'thermo'       # thermodynamics
]

# Dataset reader config.
scibench_reader_cfg = dict(
    input_columns=['question'], 
    output_column='answer',
)

# Inference template (zero-shot).
scibench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt="Please provide a clear and step-by-step solution for a scientific problem in the categories of Chemistry, Physics, or Mathematics. The problem will specify the unit of measurement, which should not be included in the answer. Express the final answer as a decimal number with three digits after the decimal point. Conclude the answer by stating 'Therefore, the answer is \\boxed[ANSWER].'\\n\\nProblem: {question}\\nAnswer: " + cot_prompt
            ),
            dict(
                role='BOT',
                prompt= prompt_prefix
            )
        ])
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096),
)

# Evaluator: numeric match with tolerance.
scibench_eval_cfg = dict(
    evaluator=dict(
        type=NumericAccEvaluator,
        rtol=1e-2,  # relative tolerance 1%
        atol=1e-3   # absolute tolerance 0.001
    ),
    pred_postprocessor=dict(type=scibench_postprocess)
)

# Build the dataset list.
scibench_datasets = []
for _name in scibench_subsets:
    scibench_datasets.append(
        dict(
            abbr=f'scibench-{_name}',
            type=ScibenchDataset_Local,
            path='./data/scibench',  # local data path
            name=_name,
            reader_cfg=scibench_reader_cfg,
            infer_cfg=scibench_infer_cfg,
            eval_cfg=scibench_eval_cfg,
        )
    )

datasets = scibench_datasets

# Concatenate model lists to evaluate several models in one pass.
all_models = (deepseek_r1_14b_model
)

# Apply shared model parameters.
for model in all_models:
    model['max_out_len'] = 4096
    model['generation_kwargs']['temperature'] = 0

models = all_models

# Inference stage.
infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size=10000,  # keep each model's task unsplit
        gen_task_coef=1,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)
    ),
)
work_dir = 'outputs/default/scibench'

# Summarizer configuration - using ScibenchSummarizer for automatic result merging
summarizer = dict(
    type=ScibenchSummarizer,  # Use custom summarizer class directly (not string)
    scibench_subsets=scibench_subsets,  # Pass the subset order
    dataset_abbrs=[
        ['scibench-weighted', 'weighted_average'],  # Weighted accuracy (true acc = correct/total)
        ['scibench', 'naive_average'],              # Simple average (for comparison)
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
