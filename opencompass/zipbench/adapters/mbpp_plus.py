"""ZipBench adapter for MBPP+ (weighted evalplus code-execution scoring).

MBPP+ selection is plain row-level, so the dataset side reuses the generic
:class:`zipbench.dataset.ZipSubsetDataset` with
``base_loader='opencompass.datasets.MBPPPlusDataset'`` (a flat
Dataset, so selection is by index; the spec's ``task_id`` ids are offline
provenance).

Scoring runs generated solutions through evalplus (base + plus test suites);
an item counts as correct only when both suites fully pass (the official
evalplus pass@1 definition). The upstream ``MBPPEvaluator`` MBPPPlus branch
returns only aggregate pass@k with no per-item details, so this adapter
drives the evalplus run itself and reads per-item correctness back from the
``*_eval_results.json`` file evalplus writes.

One wrinkle: ``evalplus.evaluate.evaluate`` asserts that the samples file
covers *every* problem in the dataset, so a subset cannot be scored directly.
The adapter therefore pads the missing task_ids with a fail-fast dummy
solution, lets evalplus run, and aggregates over the real subset entries only
(the padded items' results are discarded). Everything runs in a fresh temp
directory: evalplus silently reuses an existing ``*_eval_results.json``
instead of re-scoring, so a persistent samples path must never be shared
across runs.
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import os.path as osp
import tempfile

import opencompass.datasets  # noqa: F401
from opencompass.datasets.mbpp import (MBPPEvaluator,
                                       _export_mbpp_plus_per_item,
                                       _load_mbpp_plus_any_correct)
from opencompass.registry import ICL_EVALUATORS

from zipbench.result import ZipResultShapeMixin, zip_result


@ICL_EVALUATORS.register_module()
class WeightedMbppPlusEvaluator(ZipResultShapeMixin, MBPPEvaluator):
    """MBPPEvaluator(metric='MBPPPlus') judging with ZipBench subset weights."""

    # MBPP+ solutions are standalone programs (no prompt prepending), so the
    # dummy must be a valid module on its own; the module-level raise fails
    # every test at ~no runtime cost.
    DUMMY_SOLUTION = 'raise NotImplementedError\n'

    def __init__(self, metric='MBPPPlus', breakdown_fields=None,
                 pred_postprocessor=None, **kwargs):
        super().__init__(metric=metric, **kwargs)
        # Upstream MBPPEvaluator.__init__ does not chain to BaseEvaluator, so
        # its attributes must be assigned here, after super().__init__.
        self.breakdown_fields = list(breakdown_fields or [])
        self.pred_postprocessor = pred_postprocessor
        self._dataset_replica_idx = getattr(self, '_dataset_replica_idx', 0)

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')
        if len(predictions) != len(references) or \
                len(predictions) != len(test_set):
            raise ValueError('predictions/references/test_set length mismatch')

        from evalplus.data import get_mbpp_plus, write_jsonl
        from evalplus.evaluate import evaluate
        self.write_jsonl = write_jsonl
        self.eval = evaluate

        processed = []
        for pred in predictions:
            if isinstance(pred, list):
                pred = pred[-1]
            processed.append(self._process_answer(pred))

        # evalplus refuses to run unless every problem has a sample; pad the
        # unselected task_ids with a dummy solution and ignore their results.
        problems = get_mbpp_plus()
        selected = set(references)
        missing = [tid for tid in problems if tid not in selected]
        mbpp_preds = [{'task_id': tid, 'solution': sol}
                      for tid, sol in zip(references, processed)]
        mbpp_preds += [{'task_id': tid, 'solution': self.DUMMY_SOLUTION}
                       for tid in missing]

        with tempfile.TemporaryDirectory() as tmp_dir:
            samples_path = osp.join(tmp_dir, 'mbpp_plus_samples.jsonl')
            self._run_evalplus(samples_path, mbpp_preds, True)
            eval_results_path = samples_path.replace('.jsonl',
                                                     '_eval_results.json')
            if not osp.exists(eval_results_path):
                raise ValueError('upstream evalplus scoring failed: '
                                 f'{eval_results_path} was not written')
            per_item_path = osp.join(tmp_dir, 'mbpp_plus_per_item.jsonl')
            _export_mbpp_plus_per_item(eval_results_path, per_item_path)
            any_correct = _load_mbpp_plus_any_correct(per_item_path)

        w_correct = w_total = 0.0
        n_correct = 0
        details = {}
        for index, (sample, task_id, pred, sol) in enumerate(
                zip(test_set, references, predictions, processed)):
            if task_id not in any_correct:
                raise ValueError(
                    f'task {task_id!r} missing from evalplus results')
            is_correct = bool(any_correct[task_id])
            weight = float(sample.get('weight', 1.0))
            w_correct += weight * is_correct
            w_total += weight
            n_correct += int(is_correct)
            details[str(index)] = {
                'origin': pred,
                'solution': sol,
                'reference': task_id,
                'is_correct': is_correct,
                'weight': weight,
            }

        n_real = len(references)
        weighted_pass_at_1 = (w_correct / w_total * 100) if w_total else 0.0
        return zip_result(
            weighted_pass_at_1,
            'weighted_pass@1 — subset-weighted MBPP+ pass@1: an item counts '
            'as correct only when the base AND plus test suites both pass '
            '(the official evalplus definition); official micro',
            {
                'weighted_pass@1': weighted_pass_at_1,
                'pass@1_unweighted':
                (n_correct / n_real * 100) if n_real else 0.0,
                'num_samples': n_real,
            },
            details,
        )
