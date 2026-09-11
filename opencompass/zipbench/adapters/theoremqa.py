"""ZipBench adapter for TheoremQA (weighted numeric-comparison scoring).

TheoremQA selection is plain row-level, so the dataset side reuses the
generic :class:`zipbench.dataset.ZipSubsetDataset` with
``base_loader='opencompass.datasets.TheoremQADatasetV3'`` (the raw json's
unique ``id`` column enables spec drift checking).

Scoring, however, is not exact match: ``TheoremQAEvaluatorV3``
(``opencompass/datasets/TheoremQA/main.py``) builds an answer-type-aware
groundtruth and compares via ``utils.compare_answer_with_groundtruth`` (float
tolerance, list handling, bool/option normalisation). Per-item
correctness is exactly that evaluator's ``is_correct``
flag, so the weighted evaluator below reuses the same judging logic verbatim
and only changes the aggregation to ``sum(w_i * correct_i) / sum(w_i)``.
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.TheoremQA import utils as theoremqa_utils
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

from zipbench.result import ZipResultShapeMixin, zip_result


def _judge(answer, groundtruth, answer_type):
    """Replicates TheoremQAEvaluatorV3's per-item correctness decision."""
    if answer_type in ['float', 'integer', 'bool']:
        groundtruth = [groundtruth, eval(groundtruth)]
    elif answer_type.startswith('list'):
        try:
            groundtruth = [groundtruth, eval(groundtruth)]
        except Exception:
            groundtruth = [groundtruth, None]
    else:
        groundtruth = [groundtruth, None]
    return theoremqa_utils.compare_answer_with_groundtruth(
        answer, *groundtruth)


@ICL_EVALUATORS.register_module()
class WeightedTheoremQAEvaluator(ZipResultShapeMixin, BaseEvaluator):
    """TheoremQAEvaluatorV3 judging with ZipBench subset weights."""

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')
        if len(predictions) != len(references):
            raise ValueError('predictions and references length mismatch')

        details = []
        w_correct = w_total = 0.0
        n_correct = 0
        for pred, ref, sample in zip(predictions, references, test_set):
            is_correct = bool(_judge(pred, ref, sample['Answer_type']))
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
            'weighted_accuracy — subset-weighted TheoremQA accuracy under '
            'the official TheoremQAEvaluatorV3 answer judging '
            '(numeric/list tolerant); official micro',
            {
                'weighted_accuracy': weighted_accuracy,
                'accuracy_unweighted':
                (n_correct / n_total * 100) if n_total else 0.0,
                'num_samples': n_total,
            },
            details,
        )
