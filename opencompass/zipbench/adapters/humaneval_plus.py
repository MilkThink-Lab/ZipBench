"""ZipBench adapter for HumanEval+ (weighted evalplus code-execution scoring).

HumanEval+ selection is plain row-level, so the dataset side reuses the
generic :class:`zipbench.dataset.ZipSubsetDataset` with
``base_loader='opencompass.datasets.HumanevalDataset'`` (the raw jsonl's
unique ``task_id`` column enables spec drift checking).

Scoring runs generated solutions through evalplus (base + plus test suites);
per-item correctness is exactly
``HumanEvalPlusEvaluator``'s ``is_correct`` flag (base AND plus all pass), so
the weighted evaluator below reuses the upstream evaluator's evalplus run and
only adds the ``sum(w_i * correct_i) / sum(w_i)`` aggregation on top of its
per-item details.

One wrinkle: ``evalplus.evaluate.evaluate`` asserts that the samples file
covers *every* problem in the dataset, so a subset cannot be scored directly.
The adapter therefore pads the missing task_ids with a fail-fast dummy
solution (``raise NotImplementedError``), lets evalplus run, and aggregates
over the real subset entries only (the padded items' results -- and the
upstream ``humaneval_plus_*`` pass@k keys they pollute -- are discarded).
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.humaneval import HumanEvalPlusEvaluator
from opencompass.registry import ICL_EVALUATORS

from zipbench.result import ZipResultShapeMixin, zip_result


@ICL_EVALUATORS.register_module()
class WeightedHumanEvalPlusEvaluator(ZipResultShapeMixin,
                                     HumanEvalPlusEvaluator):
    """HumanEvalPlusEvaluator judging with ZipBench subset weights."""

    # Fails on the first test case, so padded problems cost ~nothing.
    DUMMY_SOLUTION = '    raise NotImplementedError\n'

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')
        if len(predictions) != len(references):
            raise ValueError('predictions and references length mismatch')

        from evalplus.data import get_human_eval_plus

        # evalplus refuses to run unless every problem has a sample; pad the
        # unselected task_ids with a dummy solution and ignore their results.
        problems = get_human_eval_plus()
        n_real = len(test_set)
        selected = set(references)
        missing = [tid for tid in problems if tid not in selected]
        predictions = list(predictions) + [self.DUMMY_SOLUTION] * len(missing)
        references = list(references) + missing
        test_set = list(test_set) + [{'prompt': problems[tid]['prompt']}
                                     for tid in missing]

        results = super().score(predictions, references, test_set)
        if 'details' not in results:
            raise ValueError(f'upstream evalplus scoring failed: {results}')
        # The upstream pass@k keys and padded details cover the dummy
        # solutions too -- drop them, keep only the real subset entries.
        details = {str(i): results['details'][str(i)] for i in range(n_real)}

        w_correct = w_total = 0.0
        n_correct = 0
        for index, sample in enumerate(test_set[:n_real]):
            detail = details[str(index)]
            is_correct = bool(detail['is_correct'])
            weight = float(sample.get('weight', 1.0))
            w_correct += weight * is_correct
            w_total += weight
            n_correct += int(is_correct)
            detail['weight'] = weight

        weighted_pass_at_1 = (w_correct / w_total * 100) if w_total else 0.0
        return zip_result(
            weighted_pass_at_1,
            'weighted_pass@1 — subset-weighted HumanEval+ pass@1: an item '
            'counts as correct only when the base AND plus test suites both '
            'pass (the official evalplus definition); official micro',
            {
                'weighted_pass@1': weighted_pass_at_1,
                'pass@1_unweighted':
                (n_correct / n_real * 100) if n_real else 0.0,
                'num_samples': n_real,
            },
            details,
        )
