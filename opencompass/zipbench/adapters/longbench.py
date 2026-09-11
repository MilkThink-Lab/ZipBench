"""ZipBench adapter for LongBench v1 (21-task merged loader, per-row output cap).

LongBench's anchor index space is the 4750-row correctness matrix formed by
concatenating the 21 tasks grouped by the six categories
(``longbench_tasks.CATEGORY_MAP`` order: single_doc_qa -> multi_doc_qa ->
summarization -> few_shot -> synthetic -> code), official task order inside each
category, natural jsonl line order inside each task. The anchor ``indices`` index this concatenation
directly (``convert_anchor_to_spec._load_longbench_ids`` returns a 2-tuple,
``anchor_order`` stays None). The stable id is LongBench's native ``_id`` field
(a uuid hex per item, dropped by the upstream per-task loaders but present in the
raw jsonl), so ZipSubsetDataset's drift check is ACTIVE.

The upstream config (``examples/eval_longbench.py``) is *21 separate datasets*
with heterogeneous prompt templates, metrics and per-task ``max_out_len``
(32/64/128/512), whereas a ZipBench anchor selects rows of a single flat index
space. Three pieces bridge that:

* :class:`LongbenchAllDataset` -- reads the 21 raw jsonl files in the anchor
  order and pre-renders each task's official prompt format into a single
  ``prompt`` column (MuSR-style), so one ``{prompt}`` template serves every
  task. Each row carries ``task`` / ``category`` / ``max_out_len`` / ``answers``
  / ``all_classes`` for the inferencer and evaluator.

* :class:`LongBenchGenInferencer` -- a ``GenInferencer`` whose per-batch
  generate call groups rows by their ``max_out_len`` column value (at most 4
  distinct values) and calls the model once per group, scattering results back
  into the original row order. Row order never changes, so the '0'..'n-1'
  prediction keys, the tmp-json resume mechanism and partitioner slicing all
  behave exactly like the stock inferencer. ``max_out_len_override`` replaces
  the whole column with one value (the THINKING_BUDGET switch of
  ``examples/eval_longbench.py``; it leaves the official per-task caps, and so
  the official setting, behind).

* :class:`WeightedLongbenchEvaluator` -- replicates the six upstream LongBench
  scorers per row (dispatch on the ``task`` column; the per-item loop bodies are
  copied verbatim from ``opencompass/datasets/longbench/evaluators.py``,
  including the first-line truncation of trec/triviaqa/samsum/lsht that upstream
  applies as ``eval_cfg.pred_postprocessor``), then aggregates the primary
  ``score`` as the six-category equal-weight macro: within each category a
  subset-weight-normalised mean of the per-item scores, then an unweighted mean
  over the six categories, x100. On the full set (no weight column) this
  collapses to the plain six-category macro. The flat item-level micro and the
  official leaderboard EN/ZH tables (task -> category -> table two-level macro,
  same grouping as ``opencompass/configs/summarizers/longbench_official.py``)
  are reported as secondary numbers.
"""
import difflib
import inspect
import os
import os.path as osp
import re
import time
from collections import Counter
from typing import List, Optional

# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.base import BaseDataset
from opencompass.datasets.longbench.evaluators import (normalize_answer,
                                                       normalize_zh_answer)
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.registry import (ICL_EVALUATORS, ICL_INFERENCERS,
                                  LOAD_DATASET)
from opencompass.utils import get_data_path

from zipbench.result import ZipResultShapeMixin, zip_result

# Task constants live in a pure-stdlib sibling so the offline converter can
# enumerate ids without the heavy ``opencompass`` import (see
# longbench_tasks.py).
from zipbench.adapters.longbench_tasks import (  # noqa: E402
    CATEGORY_MAP, CATEGORY_ORDER, DATASET_SIZE, EN_TABLE, TASK_INFO,
    TASK_ORDER, ZH_TABLE)

__all__ = ['LongbenchAllDataset', 'LongBenchGenInferencer',
           'WeightedLongbenchEvaluator', 'TASK_ORDER']


@LOAD_DATASET.register_module()
class LongbenchAllDataset(BaseDataset):
    """All 21 LongBench v1 tasks concatenated in the anchor category order."""

    @staticmethod
    def load(path: str = 'opencompass/Longbench', **kwargs):
        import json as _json

        from datasets import Dataset, DatasetDict

        from opencompass.utils.prompt import safe_format

        base = get_data_path(path)
        rows = []
        for task in TASK_ORDER:
            info = TASK_INFO[task]
            fmt = info['prompt_format']
            with open(osp.join(base, 'data', f'{task}.jsonl'), 'r',
                      encoding='utf-8') as f:
                for line in f:
                    item = _json.loads(line)
                    # Same substitution the upstream per-task pipeline performs:
                    # PromptTemplate.generate_item -> safe_format(template,
                    # **entry) with the row dict ordered ('input', 'context',
                    # ...), i.e. sequential str.replace in that key order.
                    prompt = safe_format(fmt,
                                         input=item.get('input', ''),
                                         context=item.get('context', ''))
                    rows.append({
                        '_id': item['_id'],
                        'task': task,
                        'category': info['category'],
                        'prompt': prompt,
                        'answers': [str(a) for a in item['answers']],
                        'all_classes': [str(c) for c in
                                        (item.get('all_classes') or [])],
                        'max_out_len': info['max_out_len'],
                        'length': int(item.get('length', 0)),
                        'language': str(item.get('language', '')),
                    })
        if len(rows) != DATASET_SIZE:
            raise ValueError(
                f'LongBench merged loader expected {DATASET_SIZE} rows, '
                f'got {len(rows)}')
        full = Dataset.from_list(rows)
        # Both splits are the full 4750-row set (same natural order) so
        # ZipSubsetDataset validates stable ids from 'train' and selects the
        # weighted subset from 'test'.
        return DatasetDict({'train': full, 'test': full})


@ICL_INFERENCERS.register_module()
class LongBenchGenInferencer(GenInferencer):
    """GenInferencer honouring the per-row ``max_out_len`` dataset column.

    ``inference`` is the stock ``GenInferencer.inference`` with three changes
    (the parent is one monolithic method, so it is copied rather than hooked):

    1. the per-row ``max_out_len`` column is fetched next to the gold answers
       and zipped into ``prompt_list`` (so the resume slice keeps them aligned);
    2. every batch entry unpacks to a (prompt, gold, max_out_len) triple;
    3. the single ``generate_from_template`` call becomes an order-preserving
       group-by-value: one model call per distinct ``max_out_len`` in the batch
       (at most 4 for LongBench), results scattered back to the original batch
       positions.

    Row order is never changed, so prediction keys stay '0'..'n-1' aligned with
    the (possibly partitioner-sliced) test split, and the tmp-json resume of the
    parent keeps working: ``index`` always lands on an entry boundary and
    ``prompt_list[index:]`` slices the triples together.

    ``max_out_len_override`` (int) replaces the whole column with one value --
    the THINKING_BUDGET escape hatch for thinking models, which leaves the
    official per-task caps and hence the official setting. Without the column
    and without an override, the scalar ``max_out_len`` ctor argument applies
    (stock behaviour).

    NB: the model config must not set ``max_tokens`` in ``generation_kwargs``.
    That is not the usual dataset-vs-model ``max_out_len`` precedence (which
    this class still wins, ``openicl_infer.py:114``); the vLLM wrapper seeds
    ``max_tokens`` from the per-call ``max_out_len`` and then applies
    ``sampling_kwargs.update(self.generation_kwargs)`` on top
    (``vllm_with_tf_above_v4_33.py:153-159``), so a ``max_tokens`` key there
    silently replaces every per-row cap with one constant.
    """

    def __init__(self,
                 model,
                 max_out_len: int,
                 max_out_len_column: str = 'max_out_len',
                 max_out_len_override: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__(model, max_out_len, **kwargs)
        self.max_out_len_column = max_out_len_column
        self.max_out_len_override = max_out_len_override

    def inference(self,
                  retriever,
                  ice_template=None,
                  prompt_template=None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        import mmengine
        import torch
        from tqdm import tqdm

        from opencompass.openicl.icl_inferencer.icl_base_inferencer import \
            GenInferencerOutputHandler
        from opencompass.openicl.utils.logging import get_logger
        logger = get_logger(__name__)

        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        # 3.1 Fetch gold answers and per-row max_out_len, zip them into
        # prompt_list so the resume slice below keeps everything aligned.
        ds_reader = retriever.dataset_reader
        if ds_reader.output_column:
            golds = ds_reader.dataset['test'][ds_reader.output_column]
        else:
            golds = [None] * len(prompt_list)
        if self.max_out_len_override is not None:
            mols = [int(self.max_out_len_override)] * len(prompt_list)
        elif self.max_out_len_column in ds_reader.dataset['test'].column_names:
            mols = [int(v) for v in
                    ds_reader.dataset['test'][self.max_out_len_column]]
        else:
            mols = [int(self.max_out_len)] * len(prompt_list)
        if len(mols) != len(prompt_list):
            raise ValueError('max_out_len column length mismatch: '
                             f'{len(mols)} != {len(prompt_list)}')
        prompt_list = list(zip(prompt_list, golds, mols))

        # Create tmp json file for saving intermediate results and future
        # resuming
        index = 0
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)
        if osp.exists(tmp_json_filepath):
            try:
                tmp_result_dict = mmengine.load(tmp_json_filepath)
            except Exception:
                pass
            else:
                output_handler.results_dict = tmp_result_dict
                index = len(tmp_result_dict)

        # 4. Wrap prompts with Dataloader
        logger.info('Starting build dataloader')
        dataloader = self.get_dataloader(prompt_list[index:], self.batch_size)

        # 5. Inference for prompts in each batch
        logger.info('Starting inference process...')

        num_return_sequences = getattr(self.model, 'generation_kwargs',
                                       {}).get('num_return_sequences', 1)
        if num_return_sequences != 1:
            raise NotImplementedError(
                'LongBenchGenInferencer supports num_return_sequences=1 only')

        start_time_stamp = time.time()
        num_sample = 0
        for datum in tqdm(dataloader, disable=not self.is_main_process):
            entry, golds, mols = list(zip(*datum))
            # 5-1. Inference with local model
            extra_gen_kwargs = {}
            sig = inspect.signature(self.model.generate)
            if 'stopping_criteria' in sig.parameters:
                extra_gen_kwargs['stopping_criteria'] = self.stopping_criteria
            if 'min_out_len' in sig.parameters:
                extra_gen_kwargs['min_out_len'] = self.min_out_len
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entry, mode='gen')
                # Order-preserving group-by max_out_len (<= 4 groups): one
                # generate call per distinct value, scattered back in place.
                groups = {}
                for i, m in enumerate(mols):
                    groups.setdefault(m, []).append(i)
                generated = [None] * len(entry)
                for mol_value, idxs in groups.items():
                    results = self.model.generate_from_template(
                        [entry[i] for i in idxs],
                        max_out_len=mol_value,
                        **extra_gen_kwargs)
                    for i, res in zip(idxs, results):
                        generated[i] = res

            # 5-3. Save current output (num_return_sequences == 1)
            for prompt, prediction, gold in zip(parsed_entries, generated,
                                                golds):
                output_handler.save_results(prompt,
                                            prediction,
                                            index,
                                            gold=gold)
                index = index + 1

            # 5-4. Save intermediate results
            if (self.save_every is not None and index % self.save_every == 0
                    and self.is_main_process):
                output_handler.write_to_json(output_json_filepath,
                                             'tmp_' + output_json_filename)
            num_sample += len(datum)

        end_time_stamp = time.time()

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        if self.dump_timer and self.is_main_process:
            import json as _json
            timer_filepath = os.path.join(output_json_filepath, 'timer',
                                          'time.jsonl')
            os.makedirs(os.path.dirname(timer_filepath), exist_ok=True)
            time_dict = {
                'dataset_name': output_json_filename.removesuffix('.json'),
                'time': end_time_stamp - start_time_stamp,
                'num_sample': num_sample
            }
            with open(timer_filepath, 'a') as f:
                f.write(_json.dumps(time_dict) + '\n')

        return [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]


# ---------------------------------------------------------------------------
# Per-item scorers. Each returns the [0, 1] score one item contributes in the
# corresponding upstream evaluator (opencompass/datasets/longbench/
# evaluators.py); the loop bodies are copied verbatim, including their quirks
# (cumulative ``prediction`` reassignment across multiple references in the
# rouge/code_sim loops, the always-true ``em_match_list != 0`` comparison and
# the remove-while-iterating in classification), so per-item scores and hence
# every aggregate match the upstream evaluators bit for bit.
# ---------------------------------------------------------------------------

def _f1_score(prediction, reference):
    common = Counter(prediction) & Counter(reference)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(reference)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _score_f1(prediction, reference_list, language):
    import jieba
    task_score = 0.
    for reference in reference_list:
        if language == 'en':
            normalized_prediction = normalize_answer(prediction)
            normalized_reference = normalize_answer(reference)

            prediction_tokens = normalized_prediction.split()
            reference_tokens = normalized_reference.split()

        else:
            prediction_tokens = list(jieba.cut(prediction, cut_all=False))
            reference_tokens = list(jieba.cut(reference, cut_all=False))
            prediction_tokens = [
                normalize_zh_answer(token) for token in prediction_tokens
            ]
            reference_tokens = [
                normalize_zh_answer(token) for token in reference_tokens
            ]
            prediction_tokens = [
                token for token in prediction_tokens if len(token) > 0
            ]
            reference_tokens = [
                token for token in reference_tokens if len(token) > 0
            ]

        task_score = max(task_score,
                         _f1_score(prediction_tokens, reference_tokens))
    return task_score


def _score_rouge(prediction, reference_list, language):
    import jieba
    from rouge import Rouge
    task_score = 0.
    for reference in reference_list:
        if language == 'zh':
            prediction = ' '.join(list(jieba.cut(prediction, cut_all=False)))
            reference = ' '.join(list(jieba.cut(reference, cut_all=False)))

        rouge = Rouge()
        try:
            cur_score = rouge.get_scores([prediction], [reference],
                                         avg=True)['rouge-l']['f']
        except Exception:
            cur_score = 0.
        task_score = max(task_score, cur_score)
    return task_score


def _score_count(prediction, reference_list):
    item_score = 0.
    for reference in reference_list:
        numbers = re.findall(r'\d+', prediction)
        right_num = 0
        for number in numbers:
            if str(number) == str(reference):
                right_num += 1
        item_score += 0.0 if len(numbers) == 0 else float(right_num /
                                                          len(numbers))
    return item_score


def _score_retrieval(prediction, reference_list, language):
    item_score = 0.
    for reference in reference_list:
        if language == 'en':
            pattern = r'Paragraph (\d+)'
        else:
            pattern = r'段落(\d+)'

        matches = re.findall(pattern, reference)
        reference_id = matches[0]
        numbers = re.findall(r'\d+', prediction)
        right_num = 0
        for number in numbers:
            if str(number) == str(reference_id):
                right_num += 1

        item_score += 0.0 if len(numbers) == 0 else float(right_num /
                                                          len(numbers))
    return item_score


def _score_code_sim(prediction, reference_list):
    from fuzzywuzzy import fuzz
    task_score = 0.
    for reference in reference_list:
        all_lines = prediction.lstrip('\n').split('\n')
        prediction = ''
        for line in all_lines:
            if ('`' not in line) and ('#' not in line) and ('//' not in line):
                prediction = line
                break
        task_score = max(task_score, (fuzz.ratio(prediction, reference) / 100))
    return task_score


def _score_classification(prediction, reference_list, all_classes):
    item_score = 0.
    for reference in reference_list:
        em_match_list = []
        for class_name in all_classes:
            if class_name in prediction:
                em_match_list.append(class_name)
        for match_term in em_match_list:
            if match_term in reference and match_term != reference:
                em_match_list.remove(match_term)
        if em_match_list != 0:
            if reference in em_match_list:
                item_score += (1.0 / len(em_match_list))
        else:
            best_match = None
            highest_similarity = 0
            for names in all_classes:
                similarity = difflib.SequenceMatcher(None, names,
                                                     prediction).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = names
            item_score += float(best_match == reference)
    return item_score


def _score_item(metric, prediction, reference_list, all_classes):
    if metric == 'f1_en':
        return _score_f1(prediction, reference_list, 'en')
    if metric == 'f1_zh':
        return _score_f1(prediction, reference_list, 'zh')
    if metric == 'rouge_en':
        return _score_rouge(prediction, reference_list, 'en')
    if metric == 'rouge_zh':
        return _score_rouge(prediction, reference_list, 'zh')
    if metric == 'count':
        return _score_count(prediction, reference_list)
    if metric == 'retrieval_en':
        return _score_retrieval(prediction, reference_list, 'en')
    if metric == 'retrieval_zh':
        return _score_retrieval(prediction, reference_list, 'zh')
    if metric == 'code_sim':
        return _score_code_sim(prediction, reference_list)
    if metric == 'classification':
        return _score_classification(prediction, reference_list, all_classes)
    raise KeyError(f'unknown LongBench metric: {metric}')


@ICL_EVALUATORS.register_module()
class WeightedLongbenchEvaluator(ZipResultShapeMixin, BaseEvaluator):
    """LongBench per-task official scoring, aggregated by six-category macro.

    ``score`` needs ``test_set`` (OpenCompass passes it when the signature asks)
    to read each row's ``task`` / ``category`` / ``weight`` / ``all_classes``.
    References are the ``answers`` lists.

    Primary ``score`` = per-category subset-weight-normalised mean of the
    per-item official scores, then an equal-weight mean over the six categories,
    x100. Secondary: the flat item-level micro, the official leaderboard EN/ZH
    table averages (task -> category -> table two-level macro), ``per_task`` and
    ``per_category`` breakdowns, each with unweighted counterparts.

    The first-line truncation of trec/triviaqa/samsum/lsht (upstream a
    dataset-level ``pred_postprocessor``) runs here, after the model-level
    postprocessor (CoT strip) and before scoring; ``details`` keeps the
    untruncated prediction.
    """

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')
        if len(predictions) != len(references):
            raise ValueError('predictions and references length mismatch')
        if len(predictions) != len(test_set):
            raise ValueError('predictions and test_set length mismatch')

        per_task = {}
        details = []
        for pred, refs, sample in zip(predictions, references, test_set):
            if isinstance(pred, list):
                pred = pred[-1]
            task = sample['task']
            info = TASK_INFO[task]
            weight = float(sample.get('weight', 1.0))
            scored_pred = pred
            if info['firstline_postprocess']:
                # trec/triviaqa/samsum/lsht official truncation
                # (longbench_trec.trec_postprocess et al.).
                scored_pred = scored_pred.lstrip('\n').split('\n')[0]
            item_score = float(
                _score_item(info['metric'], scored_pred, list(refs),
                            list(sample.get('all_classes') or [])))

            acc = per_task.setdefault(
                task, {'w_score': 0.0, 'w_total': 0.0,
                       'n_score': 0.0, 'n_total': 0})
            acc['w_score'] += weight * item_score
            acc['w_total'] += weight
            acc['n_score'] += item_score
            acc['n_total'] += 1
            details.append({
                'task': task,
                'category': info['category'],
                'metric': info['metric'],
                'answer': list(refs),
                'weight': weight,
                'score': item_score,
                'pred': pred,  # untruncated model output
            })

        missing = [t for t in TASK_ORDER if t not in per_task]
        if missing:
            raise ValueError(
                'LongBench evaluator saw no rows for tasks '
                f'{missing}; refusing to aggregate a partial run')

        per_task_out = {}
        for task in TASK_ORDER:
            acc = per_task[task]
            per_task_out[task] = {
                'weighted_score': acc['w_score'] / acc['w_total'] * 100,
                'score_unweighted': acc['n_score'] / acc['n_total'] * 100,
                'n': acc['n_total'],
            }

        # Primary: six-category macro on item-weighted category means.
        per_category_out = {}
        w_cat_scores, u_cat_scores = [], []
        for category in CATEGORY_ORDER:
            w_score = w_total = n_score = 0.0
            n_total = 0
            for task in CATEGORY_MAP[category]:
                acc = per_task[task]
                w_score += acc['w_score']
                w_total += acc['w_total']
                n_score += acc['n_score']
                n_total += acc['n_total']
            w_cat = w_score / w_total * 100
            u_cat = n_score / n_total * 100
            per_category_out[category] = {
                'weighted_score': w_cat,
                'score_unweighted': u_cat,
                'n': n_total,
            }
            w_cat_scores.append(w_cat)
            u_cat_scores.append(u_cat)
        macro = sum(w_cat_scores) / len(w_cat_scores)
        macro_unweighted = sum(u_cat_scores) / len(u_cat_scores)

        # Official leaderboard EN/ZH tables: per-task score -> equal-weight
        # category mean -> equal-weight table mean (two-level macro, same
        # grouping as summarizers/longbench_official.py).
        def _table_avg(table, key):
            cat_means = []
            for _, tasks in table:
                cat_means.append(
                    sum(per_task_out[t][key] for t in tasks) / len(tasks))
            return sum(cat_means) / len(cat_means)

        # Micro over all items.
        w_score_all = sum(a['w_score'] for a in per_task.values())
        w_total_all = sum(a['w_total'] for a in per_task.values())
        n_score_all = sum(a['n_score'] for a in per_task.values())
        n_total_all = sum(a['n_total'] for a in per_task.values())

        return zip_result(
            macro,
            'macro_category_score — six-category equal-weight mean '
            '(single_doc_qa / multi_doc_qa / summarization / few_shot / '
            'synthetic / code) of the item-weighted mean per-item official '
            'LongBench scores, x100',
            {
                'macro_category_score': macro,
                'macro_category_score_unweighted': macro_unweighted,
                'micro_score': w_score_all / w_total_all * 100,
                'micro_score_unweighted': n_score_all / n_total_all * 100,
                'longbench_en_avg': _table_avg(EN_TABLE, 'weighted_score'),
                'longbench_en_avg_unweighted': _table_avg(
                    EN_TABLE, 'score_unweighted'),
                'longbench_zh_avg': _table_avg(ZH_TABLE, 'weighted_score'),
                'longbench_zh_avg_unweighted': _table_avg(
                    ZH_TABLE, 'score_unweighted'),
                'num_tasks': len(per_task_out),
                'num_samples': len(details),
                'per_category': per_category_out,
                'per_task': per_task_out,
            },
            details,
        )
