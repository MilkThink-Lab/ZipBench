"""``ZipEvalMixin`` -- the contract every ZipBench adapter implements.

The mixin is spliced in *front of* the concrete dataset class at build time
(see ``vlmeval.zipbench.apply_zip_subset``), so ``super().evaluate(...)`` runs
the upstream scoring logic completely untouched. All we do is re-aggregate its
per-item output with the subset weights.

An adapter therefore only has to answer one question: **where did upstream put
the per-item scores?** Everything else is inherited.

    class ZipFooEvalMixin(ZipEvalMixin):
        ZIP_SCORE_COL = 'hit'

        def zip_item_file(self, eval_file, **judge_kwargs):
            return get_intermediate_file_path(eval_file, '_foo_result')
"""
import os.path as osp

from ..report import (WEIGHT_COL, group_breakdown, plain_mean, weighted_mean,
                      weighted_report_acc, zip_result)


class ZipEvalMixin:

    #: Column holding the per-item score in the upstream per-item file.
    ZIP_SCORE_COL = 'hit'
    #: Extra columns to break the score down by (diagnostics only).
    ZIP_GROUPS = ('l2-category', 'category')
    #: Short name of the headline metric, used in the ``metric`` string.
    ZIP_METRIC_NAME = 'micro_accuracy'

    # ------------------------------------------------------------------ hooks

    def zip_item_file(self, eval_file, **judge_kwargs):
        """Path of the per-item scored file dumped by the upstream evaluate()."""
        raise NotImplementedError(
            f'{type(self).__name__} must implement zip_item_file()')

    def zip_item_frame(self, eval_file, base_result, **judge_kwargs):
        """Per-item DataFrame carrying ``index``, the score column and weights."""
        from ...smp import load

        path = self.zip_item_file(eval_file, **judge_kwargs)
        if path is None or not osp.exists(path):
            raise FileNotFoundError(
                f'{type(self).__name__}: per-item file {path!r} not found after running the '
                'upstream evaluate(); the weighted score cannot be computed.')
        return self.zip_attach_weights(load(path))

    def zip_aggregate(self, frame):
        """-> ``(value_to_return, headline_score, secondary)``.

        The default reproduces ``report_acc``'s shape with weighted numbers.
        """
        weighted = weighted_report_acc(
            frame, score_col=self.ZIP_SCORE_COL, groups=self.ZIP_GROUPS)
        score = weighted_mean(frame, self.ZIP_SCORE_COL)
        secondary = {
            'weighted_accuracy': score,
            'accuracy_unweighted': plain_mean(frame, self.ZIP_SCORE_COL),
            'num_samples': int(len(frame)),
            'per_group': group_breakdown(
                frame, self.ZIP_SCORE_COL, groups=('split', *self.ZIP_GROUPS)),
        }
        return weighted, score, secondary

    def zip_metric_description(self):
        note = (self.zip_manifest or {}).get('metric_note')
        if note:
            return note
        return (f'{self.ZIP_METRIC_NAME} -- subset-weighted per-item score '
                f'(sum(w*s)/sum(w)) over the {self.zip_subset} subset')

    # --------------------------------------------------------------- helpers

    def zip_attach_weights(self, frame):
        """(Re-)attach the weight column by joining ``self.data`` on ``index``."""
        if 'index' not in frame:
            raise ValueError(
                f'{type(self).__name__}: per-item frame has no `index` column, cannot join weights')
        weights = {i: w for i, w in zip(self.data['index'], self.data[WEIGHT_COL])}
        frame = frame.copy()
        missing = [i for i in frame['index'] if i not in weights]
        if missing:
            raise ValueError(
                f'{type(self).__name__}: {len(missing)} scored items are not in the subset '
                f'(first: {missing[:5]}) -- the prediction file does not match this subset.')
        if len(frame) != len(weights):
            raise ValueError(
                f'{type(self).__name__}: scored {len(frame)} items but the subset has '
                f'{len(weights)} -- a partial evaluation would bias the weighted estimate.')
        frame[WEIGHT_COL] = [weights[i] for i in frame['index']]
        return frame

    def zip_provenance(self):
        manifest = self.zip_manifest or {}
        info = dict(manifest.get('subsets', {}).get(self.zip_subset, {}))
        info.pop('file', None)
        return {
            'subset': self.zip_subset,
            'dataset': self.dataset_name,
            'dataset_size': manifest.get('dataset_size'),
            **info,
        }

    # -------------------------------------------------------------- evaluate

    def evaluate(self, eval_file, **judge_kwargs):
        from ...smp import dump, get_intermediate_file_path

        base = super().evaluate(eval_file, **judge_kwargs)
        frame = self.zip_item_frame(eval_file, base, **judge_kwargs)
        weighted, score, secondary = self.zip_aggregate(frame)

        secondary = dict(secondary)
        secondary['zipbench'] = self.zip_provenance()
        report = zip_result(score, self.zip_metric_description(), secondary)
        dump(report, get_intermediate_file_path(eval_file, '_zip', 'json'))
        return weighted
