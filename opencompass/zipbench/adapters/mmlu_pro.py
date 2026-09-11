"""ZipBench adapter for MMLU-Pro (14 categories -> one flat 12032-item list).

The upstream OpenCompass config (``mmlu_pro_0shot_cot_gen_08c1de`` /
``examples/eval_mmlu_pro_0shot_cot.py``) is *14 separate per-category datasets*
(``MMLUProDataset`` filtered by ``category``, one ``mmlu_pro_<cat>`` abbr each).
A ZipBench anchor instead selects rows of ONE flat index space: the native HF
``test`` split order (business, law, psychology, biology, chemistry, history,
other, health, economics, math, physics, computer science, philosophy,
engineering; each category a contiguous block, ``question_id`` globally
monotonic).

That native order is the anchor order: it equals the Open-LLM-Leaderboard
``leaderboard_mmlu_pro`` details ``doc_id`` order, which is the
``load_dataset(...)['test']`` parquet order. So the anchor ``indices`` index
this native order directly -- no anchor->natural remap
(``convert_anchor_to_spec._load_mmlu_pro_ids`` returns a 2-tuple,
``anchor_order`` stays None).

:class:`MMLUProAllDataset` therefore loads the test split WITHOUT any per-category
filtering or reordering, and reuses the upstream ``_parse`` so ``options_str`` /
``cot_content`` / ``answer_string`` are byte-identical to the full-set config.
Selection is plain row-level; scoring is exact-match on the extracted
"ANSWER: $LETTER" letter (``zipbench.evaluator.WeightedAccuracyEvaluator`` +
``match_answer_pattern``), so no custom evaluator is needed -- only this
combining loader.

Returning a ``DatasetDict{'train','test'}`` (both the full 12032 rows carrying
the native, globally-unique ``question_id``) makes ZipSubsetDataset's stable-id
drift check ACTIVE at runtime: ``validate_and_select`` reads the ids from 'train'
and asserts ``raw_ids[index] == spec_id`` for every selected row.
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.base import BaseDataset
# Reuse the upstream per-item parser verbatim (DRY): builds options_str /
# answer_string and strips the cot_content prefix exactly as the full-set config.
from opencompass.datasets.mmlu_pro import _parse
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

__all__ = ['MMLUProAllDataset']


@LOAD_DATASET.register_module()
class MMLUProAllDataset(BaseDataset):
    """All 14 MMLU-Pro categories; test split kept in native HF/parquet order."""

    @staticmethod
    def load(path: str = 'opencompass/mmlu_pro', **kwargs):
        from datasets import DatasetDict, load_dataset

        mmlu_pro = load_dataset(get_data_path(path))
        # No per-category filter/reorder -> native parquet order (Order A).
        test = mmlu_pro['test'].map(_parse)
        return DatasetDict({'train': test, 'test': test})
