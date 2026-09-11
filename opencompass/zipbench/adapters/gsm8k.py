"""ZipBench adapter for GSM8K (weighted numeric-tolerance scoring).

GSM8K selection is plain row-level over the natural ``test.jsonl`` order
(1319 questions); the generic :class:`zipbench.dataset.ZipSubsetDataset` with
``base_loader='opencompass.datasets.GSM8KDataset'`` provides the full dataset
the anchor index space refers to. GSM8K rows carry no stable id, so the spec
id is a content hash ``sha1(question + '\\x1f' + answer)[:16]`` (see
``convert_anchor_to_spec._rec_id`` / ``_load_gsm8k_ids``); selection is by
index and the drift check is skipped at runtime (the loaded rows have no
``_id`` column).

Scoring replicates the upstream ``opencompass.datasets.gsm8k.Gsm8kEvaluator``
correctness definition exactly: after ``gsm8k_postprocess`` (last number in
the output) and ``gsm8k_dataset_postprocess`` (the gold ``#### N`` number),
a prediction is correct iff ``pred == refer`` OR
``abs(float(pred) - int(refer)) < 1e-6``. The numeric branch is what makes
this an adapter rather than the generic string-exact
``WeightedAccuracyEvaluator``: e.g. a model that emits ``"140.0"`` against gold
``"140"`` is correct here (float equality) but would be scored wrong by plain
``==``. The only addition over the upstream evaluator is the ZipBench subset
weighting (``weight`` column added by ``ZipSubsetDataset``).
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

from zipbench.result import ZipResultShapeMixin, zip_result


def gsm8k_is_correct(pred, refer):
    """Per-item correctness, identical to ``Gsm8kEvaluator.is_equal``.

    ``pred`` is the post-processed prediction (last number, as a string),
    ``refer`` the gold ``#### N`` number (an integer string). Returns bool.
    """
    try:
        if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
            return True
    except Exception:
        pass
    return False


@ICL_EVALUATORS.register_module()
class WeightedGsm8kEvaluator(ZipResultShapeMixin, BaseEvaluator):
    """Upstream GSM8K numeric judging with ZipBench subset weights."""

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
            is_correct = gsm8k_is_correct(pred, ref)
            weight = float(sample.get('weight', 1.0))
            w_correct += weight * is_correct
            w_total += weight
            n_correct += int(is_correct)
            details.append({
                'pred': pred,
                'answer': ref,
                'weight': weight,
                'correct': is_correct,
            })

        n_total = len(details)
        weighted_accuracy = (w_correct / w_total * 100) if w_total else 0.0
        return zip_result(
            weighted_accuracy,
            'weighted_accuracy — subset-weighted GSM8K accuracy under the '
            'official numeric-tolerance judging (float equality within '
            '1e-6, not string exact-match); official micro',
            {
                'weighted_accuracy': weighted_accuracy,
                'accuracy_unweighted':
                (n_correct / n_total * 100) if n_total else 0.0,
                'num_samples': n_total,
            },
            details,
        )
