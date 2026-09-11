# ./examples/eval_winogrande_5shot.py
#
# Winogrande evaluation (5-shot generative, official dev split, all 1267
# questions).
#
# Important caveat vs the standard setting:
# - This script uses winogrande_5shot_gen_b36770, which recasts the cloze
#   task as an explicit A/B multiple-choice question answered generatively;
#   the metric is accuracy. This runs directly on chat models such as
#   VLLMwithChatTemplate.
# - The standard LM evaluation (GPT-3 / lm-eval-harness) instead uses
#   loglikelihood partial scoring: fill each option into the blank and
#   compare LL(suffix | prefix+option). That corresponds to
#   winogrande_5shot_ll_252f01, which requires get_loglikelihood support
#   (base VLLM class, not VLLMwithChatTemplate).
# - Scores from this script are therefore NOT directly comparable to
#   leaderboard numbers; for that, use the loglikelihood config with a base
#   VLLM model.

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # 5-shot generative A/B config (WinograndeDatasetV3 +
    # FixKRetriever[0,2,4,6,8] + GenInferencer +
    # first_option_postprocess('AB'), abbr='winogrande')
    from opencompass.configs.datasets.winogrande.winogrande_5shot_gen_b36770 import \
        winogrande_datasets
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import \
        models as qwen3_4b_instruct_model
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_3b_instruct import \
        models as qwen2_5_3b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as qwen3_30b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import \
        models as qwen3_30b_think_model
    from opencompass.configs.models.qwen3.vllm_qwen3_5_27b import \
        models as qwen3_5_27b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_32b import \
        models as deepseek_r1_32b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import \
        models as deepseek_r1_14b_model

datasets = winogrande_datasets
models = qwen3_30b_think_model

for model in models:
    # The task only needs an A/B answer, but think models reason first; a
    # short cap truncates the reasoning and breaks letter extraction.
    model['max_out_len'] = 2048
    model['generation_kwargs']['temperature'] = 0

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/default/winogrande_5shot'
