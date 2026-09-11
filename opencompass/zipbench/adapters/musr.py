"""ZipBench adapter for MuSR (concatenated 3-scenario loader, macro scoring).

MuSR's anchor index space is the 756-row correctness matrix formed by
concatenating the three scenarios' questions in fixed
``[murder_mysteries, object_placements, team_allocation]`` order (250 + 256 + 250;
object_placements is 64 stories x 4 questions), each scenario in its natural
(story, question) order. The upstream config (``musr_gen_b47fd3`` /
``examples/eval_musr_0shot.py``) is *3 separate datasets*, one ``name=`` per
scenario, whereas a ZipBench anchor selects rows of a single flat index space, so
:class:`MusrAllDataset` provides the one concatenated dataset the anchor indices
refer to. No anchor->natural remap is needed --
the anchor ``indices`` index this concatenation directly
(``convert_anchor_to_spec._load_musr_ids`` returns a 2-tuple, ``anchor_order``
stays None).

Scoring is exact-match on the last ``ANSWER: <int>`` line compared to the gold
choice number (:class:`WeightedMusrEvaluator`), aggregated **macro**: within each
scenario a subset-weight-normalised accuracy, then an equal-weight mean over the
three scenarios. That is the official metric -- MuSR's paper reports per-domain
accuracy and their unweighted mean, which is what OpenCompass's ``musr_average``
summarizer computes -- so ``macro_accuracy`` is the primary metric here (same
convention as :mod:`zipbench.adapters.bbh`). The flat item-level
``micro_accuracy`` is reported alongside as a secondary number.

The per-scenario numbers in ``per_scenario`` are diagnostics only -- a single
scenario holds too few questions for its accuracy to be quoted on its own.

A custom evaluator (rather than ``WeightedAccuracyEvaluator`` + a pred
postprocessor) is used because the reference is an ``int`` gold choice number, so
the generic ``pred == ref`` on a string-extracted answer would never match; the
custom scorer parses the choice integer and compares ``int == int`` while reading
each row's subset ``weight``. Parsing runs inside the evaluator (on a copy) so the
full chain-of-thought is preserved in the saved ``details`` (unlike an
``eval_cfg``-level pred_postprocessor, which would truncate it).

The stable id is scenario-qualified positional -- ``'<scenario>-<local_idx>'``
(e.g. ``murder_mysteries-0``) -- because MuSR questions carry no id column. It
matches the ``_id`` column produced here and the ids enumerated by
``convert_anchor_to_spec._load_musr_ids`` (which imports the same
``MUSR_SCENARIOS`` constant), so spec ids and the live loader's ``_id`` are
generated from a single source of truth and cannot drift apart.

Returning a ``DatasetDict{'train','test'}`` (both the full 756 rows carrying
``_id``) makes ZipSubsetDataset's stable-id drift check ACTIVE at runtime:
``validate_and_select`` reads the ids from the 'train' split and asserts
``raw_ids[index] == spec_id`` for every selected row.
"""
import re

# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from zipbench.result import ZipResultShapeMixin, zip_result

# Scenario-name constants live in a pure-stdlib sibling so the offline converter
# can enumerate ids without the ~200s ``opencompass`` import (see musr_tasks.py).
from zipbench.adapters.musr_tasks import MUSR_SCENARIOS  # noqa: E402

__all__ = ['MusrAllDataset', 'WeightedMusrEvaluator', 'MUSR_SCENARIOS']

# Only these scalar columns survive concatenation. The upstream flatten also
# emits nested dict/list columns (``question``, ``intermediate_trees``,
# ``intermediate_data``) whose inferred PyArrow schemas differ across scenarios,
# which makes ``concatenate_datasets`` raise a feature-mismatch error. The prompt
# is already fully rendered into ``prompt`` and the gold choice number into
# ``gold_answer``, so nothing downstream needs the nested columns.
_KEEP_COLUMNS = ['prompt', 'system_prompt', 'gold_answer']


@LOAD_DATASET.register_module()
class MusrAllDataset(BaseDataset):
    """All 3 MuSR scenarios concatenated in fixed ``MUSR_SCENARIOS`` order."""

    @staticmethod
    def load(path: str = 'opencompass/musr', **kwargs):
        from datasets import DatasetDict, concatenate_datasets

        from opencompass.datasets.musr.musr import MusrDataset

        parts = []
        for scenario in MUSR_SCENARIOS:
            # Reuse the upstream per-scenario flatten verbatim (DRY). Default
            # kwargs (self_consistency_n=1, exclude_contrastive_examples=False,
            # ...) reproduce exactly what ``musr_gen_b47fd3`` produces per
            # scenario -> one row per (story, question) in the anchor order.
            # NB: MusrDataset.load returns a FLAT ``Dataset`` (not a
            # DatasetDict), so there is no ``['test']`` indexing here.
            ds = MusrDataset.load(path=path, name=scenario)
            n = len(ds)
            ds = ds.remove_columns(
                [c for c in ds.column_names if c not in _KEEP_COLUMNS])
            ds = ds.add_column('scenario', [scenario] * n)
            ds = ds.add_column('_id', [f'{scenario}-{i}' for i in range(n)])
            parts.append(ds)
        full = concatenate_datasets(parts)
        # Both splits are the full 756-row set (same natural order) so
        # ZipSubsetDataset validates stable ids from 'train' and selects the
        # weighted subset from 'test'.
        return DatasetDict({'train': full, 'test': full})


def _parse_musr_answer(text):
    """Return the int of the LAST ``ANSWER:`` line, or ``None``.

    Combines ``musr_answer_last_line`` (keep the last line containing
    ``ANSWER:``) with ``MusrEvaluator``'s parse (first integer after
    ``ANSWER:``). Case-sensitive on ``ANSWER:``, identical to the upstream
    generative path that ``examples/eval_musr_0shot.py`` uses.
    """
    lines = [ln for ln in text.split('\n') if 'ANSWER:' in ln]
    if not lines:
        return None
    match = re.search(r'\d+', lines[-1].split('ANSWER:')[-1])
    return int(match.group()) if match else None


@ICL_EVALUATORS.register_module()
class WeightedMusrEvaluator(ZipResultShapeMixin, BaseEvaluator):
    """MuSR last-ANSWER-line exact-match, aggregated MACRO with subset weights.

    ``score`` needs ``test_set`` (OpenCompass passes it when the signature asks)
    to read each row's ``scenario`` / ``weight``. The reference is the gold
    choice number (``gold_answer`` = ``question['answer'] + 1``), an ``int``.

    Primary metric ``macro_accuracy`` is the official MuSR metric: within each
    scenario a subset-weight-normalised accuracy, then an equal-weight mean over
    the three scenarios (== OpenCompass's ``musr_average``). The flat
    ``micro_accuracy`` is
    reported alongside as a secondary number, together with a ``per_scenario``
    breakdown (diagnostics only; a single scenario holds too few questions to
    be quoted on its own). On the full
    set (no weight column, uniform weights) ``macro_accuracy`` collapses to the
    official macro and ``micro_accuracy`` to the plain micro accuracy.

    Key order matters: OpenCompass's summarizer takes the FIRST metric key as the
    dataset's default, so ``macro_accuracy`` is emitted first.
    """

    def score(self, predictions, references, test_set):
        if not test_set:
            raise ValueError('test set is empty')
        if len(predictions) != len(references):
            raise ValueError('predictions and references length mismatch')
        if len(predictions) != len(test_set):
            raise ValueError('predictions and test_set length mismatch')

        w_correct = w_total = 0.0
        n_correct = n_total = 0
        per_scenario = {}
        details = []
        for pred, ref, sample in zip(predictions, references, test_set):
            if isinstance(pred, list):
                pred = pred[-1]
            weight = float(sample.get('weight', 1.0))
            scenario = sample.get('scenario', 'musr')
            parsed = _parse_musr_answer(pred)
            is_correct = bool(parsed is not None and parsed == int(ref))

            w_correct += weight * is_correct
            w_total += weight
            n_correct += int(is_correct)
            n_total += 1
            acc = per_scenario.setdefault(
                scenario, {'w_correct': 0.0, 'w_total': 0.0,
                           'n_correct': 0, 'n_total': 0})
            acc['w_correct'] += weight * is_correct
            acc['w_total'] += weight
            acc['n_correct'] += int(is_correct)
            acc['n_total'] += 1
            details.append({
                'scenario': scenario,
                'pred_answer': parsed,
                'answer': int(ref),
                'weight': weight,
                'correct': is_correct,
                'pred': pred,  # full chain-of-thought preserved
            })

        # Macro = equal-weight mean over scenarios (primary, the official
        # musr_average metric); the flat item-level micro is secondary.
        per_scenario_out = {}
        w_scen_accs, u_scen_accs = [], []
        for scenario, acc in per_scenario.items():
            w_acc = (acc['w_correct'] / acc['w_total'] * 100
                     if acc['w_total'] else 0.0)
            u_acc = (acc['n_correct'] / acc['n_total'] * 100
                     if acc['n_total'] else 0.0)
            per_scenario_out[scenario] = {
                'weighted_accuracy': w_acc,
                'accuracy_unweighted': u_acc,
                'n': acc['n_total'],
            }
            w_scen_accs.append(w_acc)
            u_scen_accs.append(u_acc)

        n_scen = len(per_scenario_out)
        macro = sum(w_scen_accs) / n_scen if n_scen else 0.0
        return zip_result(
            macro,
            'macro_accuracy — official MuSR musr_average: per-domain '
            'accuracy, then an equal-weight mean over the 3 domains',
            {
                'macro_accuracy': macro,
                'macro_accuracy_unweighted': (sum(u_scen_accs) / n_scen
                                              if n_scen else 0.0),
                'micro_accuracy': (w_correct / w_total * 100
                                   if w_total else 0.0),
                'micro_accuracy_unweighted': (n_correct / n_total * 100
                                              if n_total else 0.0),
                'num_scenarios': n_scen,
                'num_samples': len(details),
                'per_scenario': per_scenario_out,
            },
            details,
        )
