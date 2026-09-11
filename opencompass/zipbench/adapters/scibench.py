"""ZipBench adapter for SciBench (weighted numeric-tolerance scoring).

SciBench selection is plain row-level over the *concatenation* of the 10
subject json files (atkins, calculus, chemmc, class, diff, fund, matter,
quan, stat, thermo -- 583 questions in this fixed order). The upstream
config (``examples/eval_scibench.py``) evaluates each subject as a separate
dataset, so :class:`ScibenchZipDataset` below provides the single
concatenated dataset the anchor index space refers to, adding a stable
``_id`` column (``'<subject>-<local_idx>'``; content hashes would collide --
583 questions have only 577 unique question+answer pairs).

Scoring: extract the *last* number from the
post-processed prediction / reference (after reducing LaTeX scientific
notation ``4.7 \\times 10^{14}`` to its coefficient) and compare with
``abs(pred - ref) <= atol + rtol * abs(ref)`` (atol=1e-3, rtol=1e-2 -- the
same tolerances as the reference config's ``NumericAccEvaluator``), falling
back to normalised free-text equality when either side has no number. The
582-item anchor index space excludes the one question with an empty reference
answer (natural index 309, fund[24]), so no anchor can ever select it.
"""
import re

# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from zipbench.result import ZipResultShapeMixin, zip_result

SCIBENCH_SUBJECTS = ['atkins', 'calculus', 'chemmc', 'class', 'diff', 'fund',
                     'matter', 'quan', 'stat', 'thermo']

SCIBENCH_ATOL = 1e-3
SCIBENCH_RTOL = 1e-2

# Signed floats / scientific notation; the *last* match is the answer.
_NUMBER_RE = re.compile(r'[-+]?((?:\d+\.?\d*)|(?:\.\d+))(?:[eE][-+]?\d+)?')

# LaTeX scientific notation: keep only the coefficient (the reference stores
# only the coefficient). Bounded digit counts prevent catastrophic
# backtracking on very long model outputs.
_LATEX_SCI_RE = re.compile(
    r'([-+]?\d{1,30}(?:\.\d{0,30})?)\s*\\times\s*10\s*\^\s*\{?\s*[-+]?\d{1,10}\s*\}?')


@LOAD_DATASET.register_module()
class ScibenchZipDataset(BaseDataset):
    """All 10 SciBench subjects concatenated in fixed natural order."""

    @staticmethod
    def load(path: str):
        import json
        import os.path as osp

        from datasets import Dataset

        rows = []
        for subject in SCIBENCH_SUBJECTS:
            with open(osp.join(path, f'{subject}.json'), 'r',
                      encoding='utf-8') as f:
                raw = json.load(f)
            for local_idx, entry in enumerate(raw):
                rows.append({
                    '_id': f'{subject}-{local_idx}',
                    'subset': subject,
                    'question': entry['problem_text'].strip(),
                    'answer': entry['answer_number'].strip(),
                })
        return Dataset.from_list(rows)


def _extract_last_number(value):
    """Last numeric value in a string, LaTeX sci-notation reduced first."""
    if value is None:
        return None
    s = _LATEX_SCI_RE.sub(r'\1', str(value))
    matches = list(_NUMBER_RE.finditer(s))
    if not matches:
        return None
    try:
        return float(matches[-1].group(0))
    except (TypeError, ValueError):
        return None


def _normalize_free_text(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = re.sub(r'\\boxed\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip(' \t\r\n"\'`.,;:!?')
    return s or None


def scibench_is_correct(pred, ref, atol=SCIBENCH_ATOL, rtol=SCIBENCH_RTOL):
    """Per-item correctness (last-number numeric tolerance).

    Returns True / False, or None when neither a numeric nor a text
    comparison is possible (such items are skipped).
    """
    pred_num = _extract_last_number(pred)
    ref_num = _extract_last_number(ref)
    if pred_num is not None and ref_num is not None:
        return abs(pred_num - ref_num) <= atol + rtol * abs(ref_num)
    pred_text = _normalize_free_text(pred)
    ref_text = _normalize_free_text(ref)
    if pred_text is not None and ref_text is not None:
        return pred_text == ref_text
    return None


@ICL_EVALUATORS.register_module()
class WeightedScibenchEvaluator(ZipResultShapeMixin, BaseEvaluator):
    """Mining-matrix numeric judging with ZipBench subset weights."""

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')
        if len(predictions) != len(references):
            raise ValueError('predictions and references length mismatch')

        details = []
        w_correct = w_total = 0.0
        n_correct = 0
        for pred, ref, sample in zip(predictions, references, test_set):
            if isinstance(pred, list):
                pred = pred[-1]
            is_correct = bool(scibench_is_correct(pred, ref))
            weight = float(sample.get('weight', 1.0))
            w_correct += weight * is_correct
            w_total += weight
            n_correct += int(is_correct)
            details.append({
                'pred': pred,
                'weight': weight,
                'is_correct': is_correct,
            })

        n_total = len(details)
        weighted_accuracy = (w_correct / w_total * 100) if w_total else 0.0
        return zip_result(
            weighted_accuracy,
            'weighted_accuracy — official SciBench metric: the '
            'per-question mean over all 10 textbooks (micro)',
            {
                'weighted_accuracy': weighted_accuracy,
                'accuracy_unweighted':
                (n_correct / n_total * 100) if n_total else 0.0,
                'num_samples': n_total,
            },
            details,
        )
