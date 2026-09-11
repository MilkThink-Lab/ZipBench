"""OCRBench_v2 adapter (batch C1) -- within-group weighting + task macro-average.

The upstream metric is a *task-group macro average*: en is the mean of 8 group
means, cn of 5 (``ocrbrnch_v2_eval.py::ocrbench_v2_aggregate_accuracy``), NOT
a micro accuracy. Correspondingly the subset weights are normalised *within*
each group (each group sums to 1, total 13), and the weighted score is
"within-group weighted mean, then macro" -- both en and cn come out of one
inference run over the merged spec.

Only the two Overall numbers are the weighted headline; the 13 per-group
cells are diagnostics.
"""
import os
import os.path as osp
import warnings

from ..layouts import OCRBenchV2Layout, ocrbench_cat2group
from ..report import WEIGHT_COL, plain_mean, weighted_mean
from .base import ZipEvalMixin


class ZipOCRBenchV2EvalMixin(ZipEvalMixin):

    ZIP_SCORE_COL = 'score'
    ZIP_GROUPS = ('type',)
    ZIP_METRIC_NAME = 'macro_task_average'

    EN_GROUPS = OCRBenchV2Layout.EN_GROUPS
    CN_GROUPS = OCRBenchV2Layout.CN_GROUPS
    #: headline result-dict key -> manifest label (for per-language tolerances).
    ZIP_HEADLINE_LABELS = {'English Overall Score': 'en', 'Chinese Overall Score': 'cn'}

    # ------------------------------------------------------------------ hooks

    def zip_item_file(self, eval_file, **judge_kwargs):
        from ...smp import get_intermediate_file_path

        return get_intermediate_file_path(eval_file, '_per_question', 'json')

    def zip_prepare_item_frame(self, frame):
        """Attach the aggregation group, probed from the upstream if-chain."""
        cat2group = ocrbench_cat2group(frame['type'])
        frame = frame.copy()
        frame['group'] = [cat2group[t] for t in frame['type']]
        return frame

    def zip_item_frame(self, eval_file, base_result, **judge_kwargs):
        import pandas as pd

        from ...smp import load

        path = self.zip_item_file(eval_file, **judge_kwargs)
        if not osp.exists(path):
            raise FileNotFoundError(
                f'{type(self).__name__}: per-item file {path!r} not found after running the '
                'upstream evaluate(); the weighted score cannot be computed.')
        frame = self.zip_prepare_item_frame(pd.DataFrame(load(path)))
        frame = self.zip_attach_weights(frame)
        if 'ignored' in frame and frame['ignored'].astype(bool).any():
            n_ignored = int(frame['ignored'].astype(bool).sum())
            warnings.warn(
                f'OCRBench_v2: {n_ignored} item(s) were ignored by the upstream scorer; their '
                'group weights renormalise implicitly over the remaining items')
            frame = frame[~frame['ignored'].astype(bool)].reset_index(drop=True)
        return frame

    def zip_weight_groups(self, data):
        """Group of every dataset row (for validate's per-group weight check)."""
        cat2group = ocrbench_cat2group(data['category'])
        return [cat2group[c] for c in data['category']]

    def zip_aggregate(self, frame):
        import numpy as np

        groups = (*self.EN_GROUPS, *self.CN_GROUPS)
        present = set(frame['group'])
        missing = [g for g in groups if g not in present]
        if missing:
            # A missing group silently changes the macro-average denominator.
            raise ValueError(
                f'OCRBench_v2: no scored items for group(s) {missing}; refusing to compute a '
                'macro average over a different denominator than upstream')

        per_w, per_u = {}, {}
        for group, sub in frame.groupby('group'):
            per_w[group] = weighted_mean(sub, self.ZIP_SCORE_COL)
            per_u[group] = plain_mean(sub, self.ZIP_SCORE_COL)
        en = float(np.mean([per_w[g] for g in self.EN_GROUPS]))
        cn = float(np.mean([per_w[g] for g in self.CN_GROUPS]))
        en_u = float(np.mean([per_u[g] for g in self.EN_GROUPS]))
        cn_u = float(np.mean([per_u[g] for g in self.CN_GROUPS]))

        # Same shape as the upstream result dict, weighted numbers inside.
        weighted = {g: per_w[g] for g in groups}
        weighted['English Overall Score'] = en
        weighted['Chinese Overall Score'] = cn

        secondary = {
            'weighted_accuracy': en,
            'accuracy_unweighted': en_u,
            'headline': {'English Overall Score': en, 'Chinese Overall Score': cn},
            'headline_unweighted': {'English Overall Score': en_u, 'Chinese Overall Score': cn_u},
            'per_group': {g: {'weighted': per_w[g], 'unweighted': per_u[g],
                              'n': int((frame['group'] == g).sum())} for g in groups},
            'num_samples': int(len(frame)),
            'weight_sum': float(np.asarray(frame[WEIGHT_COL], dtype=float).sum()),
        }
        return weighted, en, secondary

    # -------------------------------------------------------------- evaluate

    def evaluate(self, eval_file, **judge_kwargs):
        from ...smp import dump, get_intermediate_file_path

        weighted = super().evaluate(eval_file, **judge_kwargs)
        # The upstream evaluate() wrote its *unweighted* subset scores to
        # _score.json -- a wrong-metric file whose name looks more official
        # than ours. Move it aside and put the weighted dict in its place.
        score_pth = get_intermediate_file_path(eval_file, '_score', 'json')
        if osp.exists(score_pth):
            os.replace(score_pth,
                       get_intermediate_file_path(eval_file, '_score_unweighted', 'json'))
        dump(weighted, score_pth)
        return weighted

    # ----------------------------------------------------------- smoke hooks

    @staticmethod
    def zip_smoke_fake(ds, data, wrong_every=3):
        """Predict the first gold answer verbatim; blank every Nth row."""
        import ast

        preds = []
        for i, (_, row) in enumerate(data.iterrows()):
            if i % wrong_every == 0:
                preds.append('')
                continue
            try:
                answers = ast.literal_eval(row['answer'])
            except (ValueError, SyntaxError):
                answers = [row['answer']]
            first = answers[0] if isinstance(answers, list) and answers else answers
            preds.append(first if isinstance(first, str) else str(first))
        data['prediction'] = preds
        return data

    @staticmethod
    def zip_smoke_expected(ds, scored, weights):
        """Independent recompute of the en headline (macro over group means)."""
        cat2group = ocrbench_cat2group(scored['type'])
        num, den = {}, {}
        ignored = scored['ignored'] if 'ignored' in scored else [False] * len(scored)
        for idx, typ, s, skip in zip(scored['index'], scored['type'], scored['score'], ignored):
            if skip:
                continue
            group = cat2group[typ]
            w = weights[idx]
            num[group] = num.get(group, 0.0) + w * float(s)
            den[group] = den.get(group, 0.0) + w
        en_groups = ZipOCRBenchV2EvalMixin.EN_GROUPS
        return sum(num[g] / den[g] for g in en_groups) / len(en_groups)
