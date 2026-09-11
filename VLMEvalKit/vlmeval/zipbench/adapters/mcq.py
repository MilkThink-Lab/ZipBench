"""ZipBench adapter for the ``ImageMCQDataset`` family (class A).

Upstream ``ImageMCQDataset.evaluate_heuristic`` dumps a per-item file with a
``hit`` column and then aggregates it with ``report_acc`` -- a plain micro mean
(``vlmeval/dataset/utils/multiple_choice.py:81``). The weighted counterpart is
therefore a straight swap of that mean, which is what ``ZipEvalMixin`` already
does; this adapter only has to locate the per-item file.

Covers: MMStar, RealWorldQA, SpatialEval (batch A1) and, via
``adapters/mmmu.py``, the MMMU family.

Circular-evaluation datasets (MMBench / CCBench / *_CIRCULAR) are NOT
supported: ``mcq_circular_eval`` groups four permutations of the same question
under one ``g_index`` and scores the group, so the selection unit is the group,
not the row. Building a subset of one would silently drop permutations.
"""
from .base import ZipEvalMixin

#: Mirrors ``ImageMCQDataset.evaluate_heuristic`` (image_mcq.py:264).
_JUDGE_NAME_MAP = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}


class ZipMCQEvalMixin(ZipEvalMixin):

    ZIP_SCORE_COL = 'hit'
    ZIP_GROUPS = ('l2-category', 'category')
    ZIP_METRIC_NAME = 'micro_accuracy'

    def zip_item_file(self, eval_file, **judge_kwargs):
        from ...smp import get_intermediate_file_path

        if judge_kwargs.get('use_verifier', False):
            # evaluate_verifier() path (image_mcq.py:419)
            return get_intermediate_file_path(eval_file, '_detailed_results')

        judge = judge_kwargs.get('model', 'exact_matching')
        name_str = _JUDGE_NAME_MAP.get(judge, judge)
        return get_intermediate_file_path(eval_file, f'_{name_str}_result')

    def zip_item_frame(self, eval_file, base_result, **judge_kwargs):
        if judge_kwargs.get('use_verifier', False):
            self.ZIP_SCORE_COL = 'verifier_match'
        frame = super().zip_item_frame(eval_file, base_result, **judge_kwargs)
        if self.ZIP_SCORE_COL not in frame:
            raise ValueError(
                f'per-item file has no `{self.ZIP_SCORE_COL}` column '
                f'(columns: {list(frame.columns)[:20]})')
        return frame
