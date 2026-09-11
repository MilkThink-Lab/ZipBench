from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    # All 21 LongBench v1 subtasks, 4750 questions, 0-shot. Prompts and
    # max_out_len follow the official config/dataset2prompt.json and
    # dataset2maxlen.json. Three tasks use aligned config variants because
    # the stock OpenCompass configs diverge from the official setting (see
    # each file's header):
    #   qasper       stock copied hotpotqa's prompt (same hash 6b3efc) and
    #                its max_out_len 32; official is 128
    #   narrativeqa  stock "fixed" the official prompt's typo "asconcisely"
    #   multi_news   stock appends an extra trailing newline
    from opencompass.configs.datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen_74ecb5 import \
        LongBench_narrativeqa_datasets
    from opencompass.configs.datasets.longbench.longbenchqasper.longbench_qasper_gen_106ce4 import \
        LongBench_qasper_datasets
    from opencompass.configs.datasets.longbench.longbenchmulti_news.longbench_multi_news_gen_f6e3fb import \
        LongBench_multi_news_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import \
        LongBench_multifieldqa_en_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import \
        LongBench_multifieldqa_zh_datasets
    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import \
        LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import \
        LongBench_2wikimqa_datasets
    from opencompass.configs.datasets.longbench.longbenchmusique.longbench_musique_gen import \
        LongBench_musique_datasets
    from opencompass.configs.datasets.longbench.longbenchdureader.longbench_dureader_gen import \
        LongBench_dureader_datasets
    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen import \
        LongBench_gov_report_datasets
    from opencompass.configs.datasets.longbench.longbenchqmsum.longbench_qmsum_gen import \
        LongBench_qmsum_datasets
    from opencompass.configs.datasets.longbench.longbenchvcsum.longbench_vcsum_gen import \
        LongBench_vcsum_datasets
    from opencompass.configs.datasets.longbench.longbenchtrec.longbench_trec_gen import \
        LongBench_trec_datasets
    from opencompass.configs.datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import \
        LongBench_triviaqa_datasets
    from opencompass.configs.datasets.longbench.longbenchsamsum.longbench_samsum_gen import \
        LongBench_samsum_datasets
    from opencompass.configs.datasets.longbench.longbenchlsht.longbench_lsht_gen import \
        LongBench_lsht_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import \
        LongBench_passage_count_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import \
        LongBench_passage_retrieval_en_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import \
        LongBench_passage_retrieval_zh_datasets
    from opencompass.configs.datasets.longbench.longbenchlcc.longbench_lcc_gen import \
        LongBench_lcc_datasets
    from opencompass.configs.datasets.longbench.longbenchrepobench.longbench_repobench_gen import \
        LongBench_repobench_datasets

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

    # Reproduces the official leaderboard: one EN and one ZH table, six
    # category columns each, Avg = equal-weight mean over the six categories
    # (category macro average, not a flat task-level mean).
    from opencompass.configs.summarizers.longbench_official import summarizer

datasets = sum([
    v for k, v in locals().items()
    if k.startswith('LongBench_') and k.endswith('_datasets')
], [])

models = qwen3_30b_instruct_model

# ---------------------------------------------------------------------------
# Output length
# ---------------------------------------------------------------------------
# None = keep the official per-task max_out_len (32/64/128/512, set in each
# dataset config, which takes precedence over the model config's
# max_out_len). LongBench does no answer extraction: F1 / ROUGE-L /
# fuzz.ratio compare the whole output against a one-line reference, so the
# precision term penalises every extra word -- raising the caps only lowers
# scores.
#
# Thinking models must override this (a 32-token cap cannot even emit
# ``</think>``, so the CoT stripping below has nothing to work on), but doing
# so departs from the official setting and the results are no longer
# comparable to the official leaderboard / paper numbers. Even with CoT
# stripped, a verbose answer is still penalised by precision -- LongBench v1
# targets direct-answer models; prefer v2 for reasoning models.
THINKING_BUDGET = None  # e.g. 8192 for thinking models

if THINKING_BUDGET is not None:
    for ds in datasets:
        ds['infer_cfg']['inferencer']['max_out_len'] = THINKING_BUDGET

for model in models:
    # The official pred.py uses do_sample=False / num_beams=1, i.e. greedy.
    model['generation_kwargs']['temperature'] = 0

    # CoT stripping is model-level: openicl_eval.py runs the model-level
    # pred_postprocessor first, then the dataset-level official postprocess
    # (first-line truncation for trec/triviaqa/samsum/lsht) -- the right
    # order.
    #
    # strip=False is required: the default strip=True calls .strip()
    # unconditionally, while the lcc / repobench-p reference answers are
    # indented code lines and fuzz.ratio is whitespace-sensitive. With
    # strip=False the postprocessor is a strict no-op when the output has no
    # think markers, so non-thinking models are unaffected.
    model['pred_postprocessor'] = dict(type=extract_non_reasoning_content,
                                       strip=False)

# mode='mid' is left unset: the longest LongBench v1 sample (~44k words) is
# far below the Qwen3 context window, so truncation never triggers. If a
# short-context model needs the official middle-truncation, note before
# enabling model['mode']='mid' that VLLMwithChatTemplate truncates AFTER the
# chat template is applied, and decoding with skip_special_tokens=True drops
# markers like <|im_start|> -- truncated samples silently lose their chat
# template.

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/default/longbench'
