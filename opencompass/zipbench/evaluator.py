"""Weighted accuracy evaluator for ZipBench subsets (benchmark-agnostic)."""
# Ensure the ``opencompass.datasets`` package initialises before importing from
# ``opencompass.openicl`` -- importing openicl first triggers a circular import
# through the inferencers.
import opencompass.datasets  # noqa: F401
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

from zipbench.result import ZipResultShapeMixin, zip_result

#: Every benchmark wired to this generic evaluator officially scores by plain
#: per-question accuracy (ARC-Challenge, C3, CommonsenseQA, GPQA, HellaSwag,
#: LongBench v2, MMLU, MMLU-Pro, OpenBookQA, Winogrande), so the
#: subset-weighted micro accuracy IS the official metric here. Benchmarks whose
#: official metric is a macro over sub-tasks get their own evaluator instead
#: (see ``adapters/bbh.py``, ``adapters/musr.py``).
DEFAULT_METRIC_DESCRIPTION = (
    'weighted_accuracy — subset-weighted per-question accuracy (micro), '
    "this benchmark's official metric")


@ICL_EVALUATORS.register_module()
class WeightedAccuracyEvaluator(ZipResultShapeMixin, BaseEvaluator):
    """Weighted multiple-choice accuracy for ZipBench subsets.

    ``weighted_accuracy = sum(w_i * correct_i) / sum(w_i) * 100``, returned as
    the headline ``score`` (see :mod:`zipbench.result` for the result shape).

    The plain (unweighted) accuracy is reported as ``secondary``'s
    ``accuracy_unweighted`` so weighted and raw numbers can be compared.
    ``breakdown_fields`` -- a list of metadata field names present on each test
    sample (e.g. ``['difficulty', 'length']``) -- produces per-value weighted +
    unweighted accuracy under ``secondary``, with weights renormalised within
    each group.

    ``metric_description`` overrides the ``metric`` string for a benchmark
    that names its metric differently; the default
    (:data:`DEFAULT_METRIC_DESCRIPTION`) is accurate for every benchmark
    currently wired to this class.
    """

    def __init__(self, breakdown_fields=None, metric_description=None):
        super().__init__()
        if breakdown_fields is None:
            breakdown_fields = []
        elif isinstance(breakdown_fields, str):
            breakdown_fields = [breakdown_fields]
        self.breakdown_fields = list(breakdown_fields)
        self.metric_description = (metric_description
                                   or DEFAULT_METRIC_DESCRIPTION)

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')
        if len(predictions) != len(references):
            raise ValueError('predictions and references length mismatch')

        w_correct = w_total = 0.0
        n_correct = n_total = 0
        # field -> value -> running {wc, wt, nc, nt}
        grp = {f: {} for f in self.breakdown_fields}

        for pred, ref, sample in zip(predictions, references, test_set):
            if isinstance(pred, list):
                pred = pred[-1]
            correct = 1.0 if pred == ref else 0.0
            weight = float(sample.get('weight', 1.0))

            w_correct += weight * correct
            w_total += weight
            n_correct += int(correct)
            n_total += 1

            for field in self.breakdown_fields:
                val = sample.get(field)
                if val is None:
                    continue
                cell = grp[field].setdefault(
                    str(val), {'wc': 0.0, 'wt': 0.0, 'nc': 0, 'nt': 0})
                cell['wc'] += weight * correct
                cell['wt'] += weight
                cell['nc'] += int(correct)
                cell['nt'] += 1

        weighted_accuracy = (w_correct / w_total * 100) if w_total else 0.0
        secondary = {
            'weighted_accuracy': weighted_accuracy,
            'accuracy_unweighted':
            (n_correct / n_total * 100) if n_total else 0.0,
            'num_samples': n_total,
        }

        for field in self.breakdown_fields:
            for val in sorted(grp[field]):
                cell = grp[field][val]
                if cell['wt'] > 0:
                    secondary[f'weighted_accuracy_{val}'] = (
                        cell['wc'] / cell['wt'] * 100)
                    secondary[f'accuracy_{val}_unweighted'] = (
                        cell['nc'] / cell['nt'] * 100)

        return zip_result(weighted_accuracy, self.metric_description,
                          secondary)
