"""LongBench v1 task constants shared by the offline converter and the runtime
adapter (loader / inferencer / evaluator). Stdlib-only on purpose: the converter
imports this module without opencompass installed.

Task order below is the anchor concatenation order:
the six categories in ``CATEGORY_ORDER``, official task order
inside each category, natural jsonl line order inside each task (4750 rows).
Prompt format strings and per-task max_out_len are extracted verbatim from the
21 dataset configs imported by examples/eval_longbench.py (incl. the 3 aligned
variants), which are byte-verified against the official LongBench
config/dataset2prompt.json / dataset2maxlen.json."""

CATEGORY_ORDER = ['single_doc_qa', 'multi_doc_qa', 'summarization', 'few_shot', 'synthetic', 'code']

# category -> task names (== official grouping)
CATEGORY_MAP = {
    'single_doc_qa': ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh'],
    'multi_doc_qa': ['hotpotqa', '2wikimqa', 'musique', 'dureader'],
    'summarization': ['gov_report', 'qmsum', 'multi_news', 'vcsum'],
    'few_shot': ['trec', 'triviaqa', 'samsum', 'lsht'],
    'synthetic': ['passage_count', 'passage_retrieval_en', 'passage_retrieval_zh'],
    'code': ['lcc', 'repobench-p'],
}

TASK_ORDER = [
    'narrativeqa',
    'qasper',
    'multifieldqa_en',
    'multifieldqa_zh',
    'hotpotqa',
    '2wikimqa',
    'musique',
    'dureader',
    'gov_report',
    'qmsum',
    'multi_news',
    'vcsum',
    'trec',
    'triviaqa',
    'samsum',
    'lsht',
    'passage_count',
    'passage_retrieval_en',
    'passage_retrieval_zh',
    'lcc',
    'repobench-p'
]

ABBR_PREFIX = 'LongBench_'

TASK_INFO = {
    'narrativeqa': dict(
        category='single_doc_qa',
        metric='f1_en',
        max_out_len=128,
        firstline_postprocess=False,
        prompt_format='You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    ),
    'qasper': dict(
        category='single_doc_qa',
        metric='f1_en',
        max_out_len=128,
        firstline_postprocess=False,
        prompt_format='You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    ),
    'multifieldqa_en': dict(
        category='single_doc_qa',
        metric='f1_en',
        max_out_len=64,
        firstline_postprocess=False,
        prompt_format='Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    ),
    'multifieldqa_zh': dict(
        category='single_doc_qa',
        metric='f1_zh',
        max_out_len=64,
        firstline_postprocess=False,
        prompt_format='阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：',
    ),
    'hotpotqa': dict(
        category='multi_doc_qa',
        metric='f1_en',
        max_out_len=32,
        firstline_postprocess=False,
        prompt_format='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    ),
    '2wikimqa': dict(
        category='multi_doc_qa',
        metric='f1_en',
        max_out_len=32,
        firstline_postprocess=False,
        prompt_format='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    ),
    'musique': dict(
        category='multi_doc_qa',
        metric='f1_en',
        max_out_len=32,
        firstline_postprocess=False,
        prompt_format='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    ),
    'dureader': dict(
        category='multi_doc_qa',
        metric='rouge_zh',
        max_out_len=128,
        firstline_postprocess=False,
        prompt_format='请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：',
    ),
    'gov_report': dict(
        category='summarization',
        metric='rouge_en',
        max_out_len=512,
        firstline_postprocess=False,
        prompt_format='You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:',
    ),
    'qmsum': dict(
        category='summarization',
        metric='rouge_en',
        max_out_len=512,
        firstline_postprocess=False,
        prompt_format='You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:',
    ),
    'multi_news': dict(
        category='summarization',
        metric='rouge_en',
        max_out_len=512,
        firstline_postprocess=False,
        prompt_format='You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:',
    ),
    'vcsum': dict(
        category='summarization',
        metric='rouge_zh',
        max_out_len=512,
        firstline_postprocess=False,
        prompt_format='下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：',
    ),
    'trec': dict(
        category='few_shot',
        metric='classification',
        max_out_len=64,
        firstline_postprocess=True,
        prompt_format='Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}',
    ),
    'triviaqa': dict(
        category='few_shot',
        metric='f1_en',
        max_out_len=32,
        firstline_postprocess=True,
        prompt_format='Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}',
    ),
    'samsum': dict(
        category='few_shot',
        metric='rouge_en',
        max_out_len=128,
        firstline_postprocess=True,
        prompt_format='Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}',
    ),
    'lsht': dict(
        category='few_shot',
        metric='classification',
        max_out_len=64,
        firstline_postprocess=True,
        prompt_format='请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}',
    ),
    'passage_count': dict(
        category='synthetic',
        metric='count',
        max_out_len=32,
        firstline_postprocess=False,
        prompt_format='There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ',
    ),
    'passage_retrieval_en': dict(
        category='synthetic',
        metric='retrieval_en',
        max_out_len=32,
        firstline_postprocess=False,
        prompt_format='Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    ),
    'passage_retrieval_zh': dict(
        category='synthetic',
        metric='retrieval_zh',
        max_out_len=32,
        firstline_postprocess=False,
        prompt_format='以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    ),
    'lcc': dict(
        category='code',
        metric='code_sim',
        max_out_len=64,
        firstline_postprocess=False,
        prompt_format='Please complete the code given below. \n{context}Next line of code:\n',
    ),
    'repobench-p': dict(
        category='code',
        metric='code_sim',
        max_out_len=64,
        firstline_postprocess=False,
        prompt_format='Please complete the code given below. \n{context}{input}Next line of code:\n',
    ),
}

DATASET_SIZE = 4750

# Official leaderboard tables (copied from
# opencompass/configs/summarizers/longbench_official.py; the two tables share
# the code-completion column).
EN_TABLE = [
    ('single-doc-qa', ['narrativeqa', 'qasper', 'multifieldqa_en']),
    ('multi-doc-qa', ['hotpotqa', '2wikimqa', 'musique']),
    ('summarization', ['gov_report', 'qmsum', 'multi_news']),
    ('few-shot', ['trec', 'triviaqa', 'samsum']),
    ('code-completion', ['lcc', 'repobench-p']),
    ('synthetic', ['passage_count', 'passage_retrieval_en']),
]
ZH_TABLE = [
    ('single-doc-qa', ['multifieldqa_zh']),
    ('multi-doc-qa', ['dureader']),
    ('summarization', ['vcsum']),
    ('few-shot', ['lsht']),
    ('code-completion', ['lcc', 'repobench-p']),
    ('synthetic', ['passage_retrieval_zh']),
]
