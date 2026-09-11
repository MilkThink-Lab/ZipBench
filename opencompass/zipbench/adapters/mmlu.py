"""ZipBench adapter for MMLU (concatenated 57-subject loader, micro scoring).

MMLU's anchor index space is the 14042-row correctness matrix formed by
concatenating the 57 subjects' ``<subject>_test.csv`` files in fixed
``sorted()`` subject order, each subject in its natural CSV row order. The
upstream config (``mmlu_openai_simple_evals_gen_b618ea`` /
``examples/eval_mmlu.py``) is *57 separate datasets*, one ``lukaemon_mmlu_*`` per
subject, whereas a ZipBench anchor selects rows of a single flat index space, so
:class:`MMLUAllDataset` provides the one concatenated dataset the anchor indices
refer to. No
anchor->natural remap is needed -- the anchor ``indices`` index this
concatenation directly (``convert_anchor_to_spec._load_mmlu_ids`` returns a
2-tuple, ``anchor_order`` stays None).

Scoring is plain **micro** exact-match accuracy on the extracted "ANSWER: LETTER"
(``zipbench.evaluator.WeightedAccuracyEvaluator`` with the ``match_answer_pattern``
pred_postprocessor) -- flat item-level
weighted accuracy, no per-subject macro renormalisation; MMLU's subject sizes
are very unbalanced (100 .. 1534), so do not switch to a BBH-style
within-subject macro. No custom evaluator is
needed -- only this concatenating loader.

The stable id is subject-qualified positional -- ``'<subject>_<local_idx>'`` (e.g.
``abstract_algebra-0``) -- because MMLU rows carry no id column. It matches the
``_id`` column produced here and the ids enumerated by
``convert_anchor_to_spec._load_mmlu_ids`` (which imports the same
``MMLU_SUBJECTS`` constant), so spec ids and the live loader's ``_id`` are
generated from a single source of truth and cannot drift apart.

Returning a ``DatasetDict{'train','test'}`` (both the full 14042 rows carrying
``_id``) makes ZipSubsetDataset's stable-id drift check ACTIVE at runtime:
``validate_and_select`` reads the ids from the 'train' split and asserts
``raw_ids[index] == spec_id`` for every selected row.
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.base import BaseDataset
from opencompass.datasets.mmlu import MMLUDataset
from opencompass.registry import LOAD_DATASET

# Subject-name constants live in a pure-stdlib sibling so the offline converter
# can enumerate ids without the ~200s ``opencompass`` import (see mmlu_tasks.py).
from zipbench.adapters.mmlu_tasks import MMLU_SUBJECTS  # noqa: E402

__all__ = ['MMLUAllDataset', 'MMLU_SUBJECTS']


@LOAD_DATASET.register_module()
class MMLUAllDataset(BaseDataset):
    """All 57 MMLU subjects concatenated in fixed ``sorted()`` subject order."""

    @staticmethod
    def load(path: str = 'opencompass/mmlu', **kwargs):
        from datasets import DatasetDict, concatenate_datasets

        parts = []
        for name in MMLU_SUBJECTS:
            # Reuse the upstream per-subject CSV reader verbatim (DRY); its
            # 'test' split is the natural CSV row order the anchor assumes.
            ds = MMLUDataset.load(path=path, name=name)['test']
            n = len(ds)
            ds = ds.add_column('subject', [name] * n)
            ds = ds.add_column('_id', [f'{name}-{i}' for i in range(n)])
            parts.append(ds)
        full = concatenate_datasets(parts)
        # Both splits are the full 14042-row set (same natural order) so
        # ZipSubsetDataset validates stable ids from 'train' and selects the
        # weighted subset from 'test'.
        return DatasetDict({'train': full, 'test': full})
