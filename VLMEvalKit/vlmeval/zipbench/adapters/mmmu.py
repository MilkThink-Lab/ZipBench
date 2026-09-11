"""ZipBench adapter for the MMMU family (batch A2: MMMU_DEV_VAL).

``MMMUDataset`` (image_mcq.py:479) only overrides ``build_prompt``; scoring is
inherited from ``ImageMCQDataset.evaluate_heuristic``, so the per-item file is
the same ``_{name_str}_result`` that ``ZipMCQEvalMixin`` already locates.
``MMMU_preproc`` (multiple_choice.py:473-475) rewrites open questions into a
2-way MCQ *inside* ``mcq_vanilla_eval``, downstream of row selection and
upstream of the ``hit`` column -- transparent to the weighted aggregation.

What is MMMU-specific is the ``split`` column (``dev`` 150 / ``validation``
900). The subset weights are normalised over the combined 1050-item space, so
the headline ``score`` is the **combined dev+validation weighted micro
accuracy**. MMMU's official validation-only number is estimated by
renormalising the weights within the validation rows and exposed in
``secondary`` as a diagnostic.
"""
from ..report import weighted_mean
from .base import ZipEvalMixin
from .mcq import ZipMCQEvalMixin

_MMMU_PRO_GUARD_MSG = (
    'ZipBench evaluation for MMMU_Pro_10c / MMMU_Pro_V is served by the '
    'standalone script at the repo root -- `python mmmu_pro.py run/eval '
    '--dataset <DS> --subset small|tiny` -- because the VLMEvalKit in-tree '
    'MMMU_Pro scoring diverges from the official MMMU repo scoring. '
    'The VLMEvalKit main pipeline is intentionally not supported.')


class ZipMMMUProGuardMixin(ZipEvalMixin):
    """Guard adapter: building the subset works (so validate can run its
    build/mae checks), but any attempt to evaluate through the VLMEvalKit
    pipeline fails loudly and points at the standalone script."""

    def zip_item_file(self, eval_file, **judge_kwargs):
        raise NotImplementedError(_MMMU_PRO_GUARD_MSG)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError(_MMMU_PRO_GUARD_MSG)


class ZipMMMUEvalMixin(ZipMCQEvalMixin):

    def zip_aggregate(self, frame):
        weighted, score, secondary = super().zip_aggregate(frame)
        if 'split' in frame:
            for sp in sorted(set(frame['split']), key=str):
                sub = frame[frame['split'] == sp]
                secondary[f'weighted_accuracy_{sp}'] = weighted_mean(sub, self.ZIP_SCORE_COL)
                secondary[f'num_samples_{sp}'] = int(len(sub))
        return weighted, score, secondary
