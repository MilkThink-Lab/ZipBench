"""ZipBench adapter for MMVP (batch C4) -- pair-level selection and scoring.

MMVP's official headline is the *pair-level* ``Overall``: adjacent TSV rows
2p / 2p+1 share one question over two images, and the pair scores only when
BOTH rows hit (``report_acc_MMVP``, ``multiple_choice.py:606-615``; it also
reports the item-level ``Average``). MMVP anchors are mined in the 150-pair
space and the spec keeps whole pairs only.

Because the spec always holds both rows of a selected pair and
``apply_zip_subset`` preserves TSV order, the upstream pairing asserts in
``report_acc_MMVP`` hold on the subset and ``super().evaluate()`` runs
untouched; per-item file location is plain MCQ. Only the aggregation changes:

* ``Overall``  = sum(w_p * pair_hit) / sum(w_p) with w_p = the pair's two row
  weights summed -- the headline;
* ``Average``  = item-level weighted mean -- a diagnostic only.
"""
import numpy as np

from ..report import WEIGHT_COL, plain_mean, weighted_mean
from .mcq import ZipMCQEvalMixin


def _pair_frame(frame, score_col, weight_col=WEIGHT_COL):
    """Fold an item frame into pairs: ``pair_id = (index - 1) // 2``.

    -> ``(pair_hit, pair_weight)`` arrays. Asserts every pair is complete --
    a half pair means the spec (or the prediction file) is broken and any
    pair-level number would be silently wrong.
    """
    idx = np.asarray(frame['index'], dtype=int)
    hits = np.asarray(frame[score_col], dtype=float)
    weights = np.asarray(frame[weight_col], dtype=float)
    pair_ids = (idx - 1) // 2

    pair_hit, pair_w = [], []
    for p in np.unique(pair_ids):
        m = pair_ids == p
        if m.sum() != 2:
            raise ValueError(
                f'pair {p} has {int(m.sum())} row(s) in the scored frame, expected exactly 2 '
                '-- the subset spec must keep whole pairs (index values 2p+1 and 2p+2)')
        pair_hit.append(hits[m].prod())  # both-correct, == report_acc_MMVP's `and`
        pair_w.append(weights[m].sum())  # w_p == the mined pair weight
    return np.asarray(pair_hit), np.asarray(pair_w)


class ZipMMVPEvalMixin(ZipMCQEvalMixin):

    ZIP_METRIC_NAME = 'pair_accuracy'

    def zip_aggregate(self, frame):
        from ...smp import d2df

        pair_hit, pair_w = _pair_frame(frame, self.ZIP_SCORE_COL)
        overall = float((pair_hit * pair_w).sum() / pair_w.sum())
        average = weighted_mean(frame, self.ZIP_SCORE_COL)

        # Same shape as report_acc_MMVP's d2df({'Average', 'Overall'}).
        result = d2df({'Average': average, 'Overall': overall})
        secondary = {
            'weighted_accuracy': overall,
            'accuracy_unweighted': float(pair_hit.mean()),
            'weighted_pair_accuracy': overall,
            'pair_accuracy_unweighted': float(pair_hit.mean()),
            'weighted_item_average': average,
            'item_average_unweighted': plain_mean(frame, self.ZIP_SCORE_COL),
            'n_pairs': int(len(pair_w)),
            'num_samples': int(len(frame)),
        }
        return result, overall, secondary

    # ------------------------------------------------------------ smoke hooks

    @staticmethod
    def zip_smoke_expected(ds, scored, weights):
        """Independent recompute of the pair-level headline for smoke_test."""
        idx = np.asarray(scored['index'], dtype=int)
        hits = np.asarray(scored[ds.ZIP_SCORE_COL], dtype=float)
        w = np.asarray([weights[i] for i in scored['index']], dtype=float)
        pair_ids = (idx - 1) // 2
        num = den = 0.0
        for p in np.unique(pair_ids):
            m = pair_ids == p
            assert m.sum() == 2
            num += hits[m].prod() * w[m].sum()
            den += w[m].sum()
        return num / den
