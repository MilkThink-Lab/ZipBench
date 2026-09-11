from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import SciCodeChatInferencer
from opencompass.datasets import SciCodeDataset, SciCodeEvaluator

with read_base():
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import \
        models as qwen3_4b_instruct_model
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_3b_instruct import \
        models as qwen2_5_3b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as qwen3_30b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import \
        models as qwen3_30b_think_model
    from opencompass.configs.models.qwen3.vllm_qwen3_32b import \
        models as qwen3_32b_model
    from opencompass.configs.models.qwen3.vllm_qwen3_5_27b import \
        models as qwen3_5_27b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_32b import \
        models as deepseek_r1_32b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import \
        models as deepseek_r1_14b_model

# ---------------- User-editable knobs ----------------
# Dataset variant: 'wo_bg' or 'with_bg'
SCICODE_VARIANT = 'with_bg'

# Prompt suffix appended after the original SciCode prompt.
# Keep it empty to preserve default behavior.
concise_cot_prompt = "Let's think step by step, but focus only on the key logic needed to reach the answer — no extra details or repetition."
cot_prompt = " Let's think step by step."
direct_prompt = " Don't think step by step, just give the answer directly."
deep_cot_prompt = " Let's think in layered depth, explicitly stating key assumptions and decomposing the problem into sub-questions before solving. After each major step, pause to challenge your own logic (possible mistakes, hidden assumptions, edge cases, and alternative paths), revising if needed. Only then deliver a concise final answer that is robust, self-consistent, and clear about any remaining uncertainty."
PROMPT_SUFFIX = cot_prompt

# Generation preset: 'deterministic' or 'sampling'
GEN_PRESET = 'deterministic'

DATASET_PATH = './data/scicode'
INFER_MAX_OUT_LEN = 32768

OVERRIDE_MODEL_LIMITS = dict(
    max_out_len=4096,
)

# Final override applied after GEN_PRESET, for quick manual tuning.
OVERRIDE_GEN_KWARGS = dict(
    temperature=0,
)

GEN_PRESET_KWARGS = dict(
    deterministic=dict(
        temperature=0,
    ),
    sampling=dict(
        temperature=0.7,
        top_p=0.9,
    ),
)
VARIANT_TO_WITH_BG = dict(
    wo_bg=False,
    with_bg=True,
)

VARIANT_TO_ABBR = dict(
    wo_bg='SciCode',
    with_bg='SciCode_with_background',
)

if SCICODE_VARIANT not in VARIANT_TO_WITH_BG:
    raise ValueError(f'Unsupported SCICODE_VARIANT: {SCICODE_VARIANT}')

if GEN_PRESET not in GEN_PRESET_KWARGS:
    raise ValueError(f'Unsupported GEN_PRESET: {GEN_PRESET}')

with_bg = VARIANT_TO_WITH_BG[SCICODE_VARIANT]
dataset_abbr = VARIANT_TO_ABBR[SCICODE_VARIANT]
prompt_text = '{prompt}' if not PROMPT_SUFFIX else '{prompt}\n' + PROMPT_SUFFIX

SciCode_reader_cfg = dict(input_columns=['prompt'], output_column=None)

SciCode_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt=prompt_text),
            # dict(role='BOT', prompt=cot_prompt)
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=SciCodeChatInferencer,
        infer_mode='every',
        max_out_len=INFER_MAX_OUT_LEN,
    ),
)

SciCode_eval_cfg = dict(
    evaluator=dict(
        type=SciCodeEvaluator,
        dataset_path=DATASET_PATH,
        with_bg=with_bg,
    ))

SciCode_datasets = [
    dict(
        abbr=dataset_abbr,
        type=SciCodeDataset,
        path=DATASET_PATH,
        with_bg=with_bg,
        reader_cfg=SciCode_reader_cfg,
        infer_cfg=SciCode_infer_cfg,
        eval_cfg=SciCode_eval_cfg)
]

datasets = SciCode_datasets

# Concatenate model lists to switch or combine evaluated models.
# all_models = (
#     qwen2_5_3b_instruct_model + qwen3_4b_instruct_model + qwen3_5_27b_model
#     qwen3_30b_instruct_model + qwen3_30b_think_model +
#     deepseek_r1_14b_model + deepseek_r1_32b_model qwen3_32b_model
# )
all_models = deepseek_r1_32b_model
models = all_models

generation_kwargs = GEN_PRESET_KWARGS[GEN_PRESET].copy()
generation_kwargs.update(OVERRIDE_GEN_KWARGS)

for model in models:
    for key, value in OVERRIDE_MODEL_LIMITS.items():
        model[key] = value
    model['generation_kwargs'] = generation_kwargs.copy()

# -------------Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
work_dir = f'outputs/default/scicode_{SCICODE_VARIANT}'
