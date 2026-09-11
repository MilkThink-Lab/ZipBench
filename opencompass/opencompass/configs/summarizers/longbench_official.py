# Reproduces the summary layout of the official THUDM/LongBench leaderboard.
#
# The official README reports two independent tables (English and Chinese),
# each with six category columns plus an Avg column, where Avg is the
# equal-weight mean of the six category scores (category macro average, not a
# flat task-level mean). Both tables share the same Code Completion column
# (lcc + repobench-p).
#
# Difference vs opencompass.configs.summarizers.groups.longbench: that one
# mixes EN and ZH tasks in the same category groups (e.g. single-document-qa
# includes multifieldqa_zh), and its longbench_en / longbench_zh are flat
# means over 16 / 7 tasks, so the weighting differs from the official tables.

_EN_CATEGORIES = [
    ('longbench_en_single-doc-qa', [
        'LongBench_narrativeqa', 'LongBench_qasper', 'LongBench_multifieldqa_en'
    ]),
    ('longbench_en_multi-doc-qa', [
        'LongBench_hotpotqa', 'LongBench_2wikimqa', 'LongBench_musique'
    ]),
    ('longbench_en_summarization', [
        'LongBench_gov_report', 'LongBench_qmsum', 'LongBench_multi_news'
    ]),
    ('longbench_en_few-shot', [
        'LongBench_trec', 'LongBench_triviaqa', 'LongBench_samsum'
    ]),
    ('longbench_code-completion', [
        'LongBench_lcc', 'LongBench_repobench-p'
    ]),
    ('longbench_en_synthetic', [
        'LongBench_passage_count', 'LongBench_passage_retrieval_en'
    ]),
]

_ZH_CATEGORIES = [
    ('longbench_zh_single-doc-qa', ['LongBench_multifieldqa_zh']),
    ('longbench_zh_multi-doc-qa', ['LongBench_dureader']),
    ('longbench_zh_summarization', ['LongBench_vcsum']),
    ('longbench_zh_few-shot', ['LongBench_lsht']),
    ('longbench_code-completion', ['LongBench_lcc', 'LongBench_repobench-p']),
    ('longbench_zh_synthetic', ['LongBench_passage_retrieval_zh']),
]

# Category groups must precede the Avg groups: the summarizer evaluates
# groups in order and writes each group's result back into parsed_results, so
# later groups can reference earlier group names. code-completion is shared
# by both tables and defined only once.
longbench_official_summary_groups = [
    dict(name=name, subsets=subsets)
    for name, subsets in _EN_CATEGORIES + _ZH_CATEGORIES
    if name != 'longbench_code-completion'
] + [
    dict(name='longbench_code-completion',
         subsets=['LongBench_lcc', 'LongBench_repobench-p']),
    dict(name='longbench_en_avg', subsets=[n for n, _ in _EN_CATEGORIES]),
    dict(name='longbench_zh_avg', subsets=[n for n, _ in _ZH_CATEGORIES]),
]

summarizer = dict(
    dataset_abbrs=[
        '--------- LongBench English (official leaderboard) ---------',
        'longbench_en_avg',
        'longbench_en_single-doc-qa',
        'longbench_en_multi-doc-qa',
        'longbench_en_summarization',
        'longbench_en_few-shot',
        'longbench_code-completion',
        'longbench_en_synthetic',
        '--------- LongBench Chinese (official leaderboard) ---------',
        'longbench_zh_avg',
        'longbench_zh_single-doc-qa',
        'longbench_zh_multi-doc-qa',
        'longbench_zh_summarization',
        'longbench_zh_few-shot',
        'longbench_code-completion',
        'longbench_zh_synthetic',
        '--------- Per-task scores ---------',
        'LongBench_narrativeqa',
        'LongBench_qasper',
        'LongBench_multifieldqa_en',
        'LongBench_multifieldqa_zh',
        'LongBench_hotpotqa',
        'LongBench_2wikimqa',
        'LongBench_musique',
        'LongBench_dureader',
        'LongBench_gov_report',
        'LongBench_qmsum',
        'LongBench_multi_news',
        'LongBench_vcsum',
        'LongBench_trec',
        'LongBench_triviaqa',
        'LongBench_samsum',
        'LongBench_lsht',
        'LongBench_passage_count',
        'LongBench_passage_retrieval_en',
        'LongBench_passage_retrieval_zh',
        'LongBench_lcc',
        'LongBench_repobench-p',
    ],
    summary_groups=longbench_official_summary_groups,
)
