from mmengine.config import read_base
from opencompass.models import HuggingFaceChatGLM3, OpenAISDK
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer

with read_base():
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare import arenahard_datasets
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import \
        models as qwen3_4b_instruct_model
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_3b_instruct import \
        models as qwen2_5_3b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as qwen3_30b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import \
        models as qwen3_30b_think_model
    from opencompass.configs.models.qwen3.vllm_qwen3_235b_instruct import \
        models as qwen3_235b_instruct_model
    from opencompass.configs.models.qwen3.vllm_qwen3_235b_instruct_4 import \
        models as qwen3_235b_instruct_model_4
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_32b import \
        models as deepseek_r1_32b_model
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import \
        models as deepseek_r1_14b_model
    
        
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models
# all_models = (
#     minimax_m2_1_model + 
#     qwen3_235b_a22b_instruct_model +
#     gpt_5_2_model +
#     gemini_3_flash_model +
#     gemini_3_pro_model +
#     grok_4_1_fast_model
# )
all_models = (deepseek_r1_14b_model)
# Apply shared model parameters.
for model in all_models:
    model['max_out_len'] = 2048
    model['generation_kwargs']['temperature'] = 0

models = all_models

datasets = arenahard_datasets

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)
    ),
)

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
# ----------------- Qwen3-30B-A3B Judge Model -----------------
judge_models = qwen3_235b_instruct_model
# Example of an API-based judge model:
# judge_models = [
#     dict(
#         abbr='qwen3-235b-a22b-instruct-2507',
#         type=OpenAISDK,
#         path='qwen3-235b-a22b-instruct-2507',
#         key='YOUR_API_KEY',  # TODO: fill in the API key
#         openai_api_base=['https://dashscope.aliyuncs.com/compatible-mode/v1'],  # TODO: API endpoint, e.g. 'http://x.x.x.x:8000/v1'
#         meta_template=api_meta_template,
#         query_per_second=8,   # adjust to the API rate limit
#         tokenizer_path='gpt-4',
#         max_out_len=4096,     # judge outputs are short (a verdict letter)
#         max_seq_len=65536,    # must fit question + answers + judge template
#         batch_size=16,
#         temperature=0,        # deterministic judging
#         retry=10,
#     )
# ]


## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=256,
        task=dict(type=SubjectiveEvalTask)
    ),
)

summarizer = dict(type=SubjectiveSummarizer, function='subjective')
work_dir = 'outputs/arenahard'
