# Most of the code in this file is copied from https://github.com/openai/simple-evals/blob/main/math_eval.py
#
# Self-judge demo:
# - Generation model: vllm_qwen2_5_3b_instruct
# - Judge model:      vllm_qwen2_5_3b_instruct
#
# Run:
#   python run.py examples/eval_simpleqa_vllm_self_judge.py
#
# Notes:
# - SimpleQA uses an LLM-as-a-judge evaluator, which requires `judge_cfg` and
#   `output_path`. These are injected by `SubjectiveEvalTask` at runtime, so we
#   must use the subjective-eval pipeline (instead of the default OpenICLEvalTask).

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import DefaultSubjectiveSummarizer
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.SimpleQA.simpleqa_gen import \
        simpleqa_datasets
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_3b_instruct import \
        models as vllm_qwen2_5_3b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as vllm_qwen3_30b_instruct_model
    from opencompass.configs.models.openai.gpt_4o_2024_05_13 import \
        models as gpt_4o_2024_05_13_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_32b import \
        models as vllm_deepseek_r1_32b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import \
        models as deepseek_r1_14b_model

# Generation model (infer stage)
models = gpt_4o_2024_05_13_model

# Judge model (eval stage) - same as generation model
judge_models = vllm_qwen3_30b_instruct_model


datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
summarizer = dict(type=DefaultSubjectiveSummarizer)

# ------------- Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask),
    ),
)

# ------------- Evaluation Stage ---------------------------------------
# Use subjective evaluation task to inject `judge_cfg` and `output_path` for
# LLM-as-a-judge evaluators (e.g. SimpleQA).
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=SubjectiveEvalTask),
    ),
)




