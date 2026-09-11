"""ZipBench adapter for GPQA (concatenated 3-split loader).

GPQA's anchor index space is the 1192-row correctness matrix formed by
concatenating the three official csvs in a fixed order -- diamond (198),
extended (546), main (448) -- by leaderboard doc_id. The upstream config
(``examples/eval_gpqa.py`` -> ``gpqa_openai_simple_evals_gen``) evaluates only
one csv at a time, so :class:`GPQAZipDataset` provides the single concatenated
dataset the anchor indices refer to. The csv row order equals the
Open-LLM-Leaderboard doc_id order (diamond 0-197, extended 198-743, main
744-1191). So no anchor->natural remap is needed -- the anchor
``indices`` index this concatenation directly (``anchor_order`` is None in
``convert_anchor_to_spec._load_gpqa_ids``).

The 1192-row space has only 546 *unique* questions (diamond subset of main
subset of extended), so content-hash / Record-ID stable ids would collide. The
stable id is therefore subset-qualified positional -- ``'<subset>_<local_idx>'``
(e.g. ``diamond_0``, ``extended_0``, ``main_0``), giving 1192 unique
``gpqa_uid`` values. ``convert_anchor_to_spec._load_gpqa_ids`` imports the same
``GPQA_SPLITS`` constant, so the live loader's ``gpqa_uid`` and the spec ids are
generated from a single source of truth and cannot drift apart.

Selection is plain row-level; scoring is exact-match on the extracted
"ANSWER: $LETTER" letter (``zipbench.evaluator.WeightedAccuracyEvaluator`` with
``GPQA_Simple_Eval_postprocess``), so no custom evaluator is needed -- only this
concatenating loader. Per-csv option shuffling is delegated to the upstream
``GPQADataset.load`` verbatim (DRY; its running ``cnt`` restarts per csv, which
is the behaviour the anchor weights assume); this class only concatenates
and adds the id column.

Returning a ``DatasetDict{'train','test'}`` (both the full 1192 rows carrying
``gpqa_uid``) makes ZipSubsetDataset's stable-id drift check ACTIVE at runtime:
``validate_and_select`` reads the ids from the 'train' split and asserts
``raw_ids[index] == spec_id`` for every selected row.
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.base import BaseDataset
from opencompass.datasets.gpqa import GPQADataset
from opencompass.registry import LOAD_DATASET

# (subset label, csv filename) in the anchor matrix's concatenation order.
# Imported by convert_anchor_to_spec._load_gpqa_ids so spec ids and the live
# loader's gpqa_uid are generated from a single source of truth.
GPQA_SPLITS = [
    ('diamond', 'gpqa_diamond.csv'),
    ('extended', 'gpqa_extended.csv'),
    ('main', 'gpqa_main.csv'),
]


@LOAD_DATASET.register_module()
class GPQAZipDataset(BaseDataset):
    """diamond + extended + main concatenated in fixed anchor order."""

    @staticmethod
    def load(path: str, **kwargs):
        from datasets import DatasetDict, concatenate_datasets

        parts = []
        for subset, csv_name in GPQA_SPLITS:
            # Reuse the upstream per-csv option shuffle verbatim (DRY): the
            # running ``cnt`` restarts per csv, which is what the anchor
            # weights assume.
            ds = GPQADataset.load(path=path, name=csv_name)
            ds = ds.add_column(
                'gpqa_uid', [f'{subset}_{i}' for i in range(len(ds))])
            parts.append(ds)
        full = concatenate_datasets(parts)
        # Both splits are the full 1192-row set (same natural order) so
        # ZipSubsetDataset validates stable ids from 'train' and selects the
        # weighted subset from 'test'.
        return DatasetDict({'train': full, 'test': full})
