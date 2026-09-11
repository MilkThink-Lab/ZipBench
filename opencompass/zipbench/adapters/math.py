"""ZipBench adapter for MATH-Hard (weighted math-equivalence scoring).

The zipbench "math" scenario is NOT the 5000-item OpenCompass ``math.json``; it
is the Open LLM Leaderboard v2 "MATH-Hard" set (MATH Level-5, 1324 items across
7 subject blocks). Its selection is plain row-level over the correctness-matrix
row order (the alphabetical concatenation of the 7 subject blocks), which the
generic :class:`zipbench.dataset.ZipSubsetDataset` reproduces via
``base_loader='opencompass.datasets.MATHDataset'`` +
``file_name='math_hard.json'`` (``data/math_hard/math_hard.json`` is written in
exactly that order). MATH rows carry no stable id, so the spec id is a content
hash ``sha1(problem + '\\x1f' + solution)[:16]`` (see
``convert_anchor_to_spec._load_math_ids``); selection is by index and the drift
check is skipped at runtime (the loaded rows have no ``_id`` column).

Scoring reuses the upstream ``opencompass.datasets.math.MATHEvaluator``
mathematical-equivalence judging *verbatim* (``self.is_equiv`` -- the Hendrycks
``_strip_string`` normaliser plus the Minerva ``normalize_final_answer``
fallback). This is what makes it an adapter rather than the generic
string-exact ``WeightedAccuracyEvaluator``: e.g. ``\\frac{1}{2}`` vs ``0.5`` or
``\\sqrt{74}`` vs ``\\sqrt {74}`` are equivalent here but not under plain
``==``. The default ``version='v2'`` matches the full-set MATH config
(``math_0shot_gen_393424``) the user chose; the only addition over the upstream
evaluator is the ZipBench subset weighting (``weight`` column added by
``ZipSubsetDataset``).
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.math import MATHEvaluator
from opencompass.registry import ICL_EVALUATORS

from zipbench.result import ZipResultShapeMixin, zip_result


@ICL_EVALUATORS.register_module()
class WeightedMATHEvaluator(ZipResultShapeMixin, MATHEvaluator):
    """Upstream MATH is_equiv judging with ZipBench subset weights.

    Subclasses :class:`opencompass.datasets.math.MATHEvaluator` so the exact
    ``is_equiv`` / ``_strip_string`` normalisation is reused; ``version``
    defaults to ``'v2'`` (the upstream default is ``'v1'``).
    """

    def __init__(self, version='v2', **kwargs):
        super().__init__(version=version, **kwargs)

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
            is_correct = bool(self.is_equiv(pred, ref))
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
            'weighted_accuracy — subset-weighted MATH-Hard accuracy under '
            'the official is_equiv mathematical-equivalence judging; '
            'official micro over the 1324 Level-5 problems',
            {
                'weighted_accuracy': weighted_accuracy,
                'accuracy_unweighted':
                (n_correct / n_total * 100) if n_total else 0.0,
                'num_samples': n_total,
            },
            details,
        )
