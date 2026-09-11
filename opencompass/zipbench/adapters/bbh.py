"""ZipBench adapter for BBH (24-task leaderboard set, macro-averaged scoring).

BBH's official metric is the **macro** average: per-task exact-match accuracy,
then an unweighted mean across the 24 tasks (BBH ships data + CoT prompts but no
scoring harness, so this is the paper's convention; the Open-LLM-Leaderboard's
size-weighted micro is a different number we deliberately do not follow).

Two pieces are needed because the upstream OpenCompass config
(``bbh_gen_5b92b0`` / ``bbh_leaderboard24_gen``) is *27/24 separate datasets*,
one per task, whereas a ZipBench anchor selects rows of a single flat index
space:

* :class:`BBHAllDataset` -- concatenates the 24 leaderboard tasks in the fixed
  ``sorted()`` task order (the anchor pkls' 5761-row index space: 21x250 +
  causal_judgement 187 + penguins_in_a_table 146 + snarks 178), each task's
  questions in the natural ``data/BBH/data/<task>.json`` order. Each
  row carries the per-task few-shot ``hint`` and its ``eval_type`` ('mcq' vs
  'freeform') so a single ``infer_cfg`` / evaluator can serve every task.

* :class:`WeightedBBHEvaluator` -- replicates OpenCompass's two BBH scorers
  per row (``BBHEvaluator_mcq`` post-processes *both* prediction and reference;
  ``BBHEvaluator`` free-form post-processes only the prediction), then
  aggregates **macro**: within each task a subset-weight-normalised accuracy,
  then an equal-weight mean over the tasks present. On the full set (no weight
  column, uniform weights) this collapses to the plain per-task mean -> the
  official BBH macro.
"""
import json
import os.path as osp

# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.base import BaseDataset
from opencompass.datasets.bbh import (bbh_freeform_postprocess,
                                      bbh_mcq_postprocess)
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from zipbench.result import ZipResultShapeMixin, zip_result

# Task-name constants live in a pure-stdlib sibling so the offline converter can
# enumerate ids without the ~200s ``opencompass`` import (see bbh_tasks.py).
from zipbench.adapters.bbh_tasks import (  # noqa: E402
    EVAL_TYPE as _EVAL_TYPE, FREEFORM_TASKS, LEADERBOARD_24_TASKS, MCQ_TASKS)

__all__ = ['BBHAllDataset', 'WeightedBBHEvaluator', 'LEADERBOARD_24_TASKS',
           'MCQ_TASKS', 'FREEFORM_TASKS']

# Same wording as bbh_leaderboard24_gen's per-task HUMAN prompt; the eval config
# keeps {hint}/{input} as template placeholders so the rendered prompt is
# byte-identical to the upstream per-task config.
_PROMPT_PREFIX = 'Follow the given examples and answer the question.'


def _hint_dir():
    """Directory of the per-task few-shot exemplar files (``lib_prompt``).

    Located via the installed ``opencompass`` package so it works regardless of
    CWD (the upstream config finds it relative to its own ``__file__``).
    """
    import opencompass
    return osp.join(osp.dirname(opencompass.__file__),
                    'configs', 'datasets', 'bbh', 'lib_prompt')


@LOAD_DATASET.register_module()
class BBHAllDataset(BaseDataset):
    """All 24 leaderboard BBH tasks concatenated in fixed ``sorted()`` order."""

    @staticmethod
    def load(path: str = 'opencompass/bbh'):
        from datasets import Dataset

        base = get_data_path(path)
        hint_dir = _hint_dir()
        rows = []
        for task in LEADERBOARD_24_TASKS:
            with open(osp.join(hint_dir, f'{task}.txt'), 'r',
                      encoding='utf-8') as f:
                hint = f.read()
            with open(osp.join(base, f'{task}.json'), 'r',
                      encoding='utf-8') as f:
                examples = json.load(f)['examples']
            for local_idx, ex in enumerate(examples):
                rows.append({
                    '_id': f'{task}-{local_idx}',
                    'task': task,
                    'eval_type': _EVAL_TYPE[task],
                    'input': ex['input'],
                    'hint': hint,
                    'target': ex['target'],
                })
        return Dataset.from_list(rows)


@ICL_EVALUATORS.register_module()
class WeightedBBHEvaluator(ZipResultShapeMixin, BaseEvaluator):
    """OpenCompass BBH per-row scoring, aggregated MACRO with subset weights.

    ``score`` needs ``test_set`` (OpenCompass passes it when the signature asks)
    to read each row's ``task`` / ``eval_type`` / ``weight``. References arrive
    raw (no ``dataset_postprocessor`` is configured) so the mcq branch
    post-processes them here, matching ``BBHEvaluator_mcq``.
    """

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')
        if len(predictions) != len(references):
            raise ValueError('predictions and references length mismatch')
        if len(predictions) != len(test_set):
            raise ValueError('predictions and test_set length mismatch')

        # Per-task accumulators: subset-weighted and unweighted.
        per_task = {}
        details = []
        for pred, ref, sample in zip(predictions, references, test_set):
            if isinstance(pred, list):
                pred = pred[-1]
            eval_type = sample['eval_type']
            task = sample['task']
            weight = float(sample.get('weight', 1.0))
            if eval_type == 'mcq':
                pred_pp = bbh_mcq_postprocess(pred)
                ref_pp = bbh_mcq_postprocess(ref)
            else:
                pred_pp = bbh_freeform_postprocess(pred)
                ref_pp = ref
            is_correct = bool(pred_pp == ref_pp)
            acc = per_task.setdefault(
                task, {'w_correct': 0.0, 'w_total': 0.0,
                       'n_correct': 0, 'n_total': 0})
            acc['w_correct'] += weight * is_correct
            acc['w_total'] += weight
            acc['n_correct'] += int(is_correct)
            acc['n_total'] += 1
            details.append({
                'task': task,
                'eval_type': eval_type,
                'pred': pred_pp,
                'answer': ref_pp,
                'weight': weight,
                'correct': is_correct,
            })

        # Macro: per-task weighted accuracy, then equal-weight mean over tasks.
        per_task_out = {}
        w_task_accs, u_task_accs = [], []
        for task, acc in per_task.items():
            w_acc = (acc['w_correct'] / acc['w_total'] * 100
                     if acc['w_total'] else 0.0)
            u_acc = (acc['n_correct'] / acc['n_total'] * 100
                     if acc['n_total'] else 0.0)
            per_task_out[task] = {
                'weighted_accuracy': w_acc,
                'accuracy_unweighted': u_acc,
                'n': acc['n_total'],
            }
            w_task_accs.append(w_acc)
            u_task_accs.append(u_acc)

        n_tasks = len(per_task_out)
        macro = sum(w_task_accs) / n_tasks if n_tasks else 0.0
        macro_unweighted = sum(u_task_accs) / n_tasks if n_tasks else 0.0
        return zip_result(
            macro,
            'macro_accuracy — official BBH metric: per-task exact-match '
            'accuracy, then an equal-weight mean over the 24 tasks',
            {
                'macro_accuracy': macro,
                'macro_accuracy_unweighted': macro_unweighted,
                'num_tasks': n_tasks,
                'num_samples': len(details),
                'per_task': per_task_out,
            },
            details,
        )
