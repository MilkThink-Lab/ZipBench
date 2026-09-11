# ./examples/eval_truthfulqa.py
#
# TruthfulQA evaluation (zero-shot generation, local automatic metrics).
#
# Caveats vs the official evaluation (scores here are NOT comparable to the
# paper / leaderboards):
# - Task: only the generation half (817 questions, HF `truthful_qa`
#   `generation` config, validation split); the multiple-choice tasks
#   (MC1/MC2/MC3) are not run.
# - Prompt: the official generation setting prepends a 6-shot QA_PRIMER
#   with stop='\n\n', max_tokens=50, greedy, and truncates the answer. This
#   script sends the bare question zero-shot with no primer / stop / answer
#   truncation.
# - Scoring: the official headline metric uses fine-tuned judge models
#   (% true / % info); this script uses the local automatic BLEU/ROUGE
#   max/diff/acc metrics (the official secondary metrics, CPU-only).
#   `*_acc` = fraction of answers closer to a correct reference than to any
#   incorrect one. For a truthfulness-judge setting, switch the metrics in
#   truthfulqa_local_gen.py to ('truth', 'info') (local ~7B judge inference,
#   needs a download and GPU memory).
#
# Requires datasets / evaluate / sacrebleu / rouge_score / nltk; the first
# run downloads the bleu/rouge metric modules and the dataset, so network
# access is needed once.
#
# Config check without inference: python run.py examples/eval_truthfulqa.py --dry-run

from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.truthfulqa.truthfulqa_local_gen import \
        truthfulqa_datasets
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

datasets = truthfulqa_datasets

# Change this line to switch the evaluated model.
models = qwen2_5_3b_instruct_model

for model in models:
    # TruthfulQA answers are 1-2 sentences, so 256 tokens suffice;
    # temperature=0 means greedy decoding, matching the official setting.
    model['max_out_len'] = 256
    model['generation_kwargs']['temperature'] = 0

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

# BLEU/ROUGE scoring runs on CPU; no GPU is needed for the eval stage.
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLEvalTask)),
)

work_dir = './outputs/default/truthfulqa'
