"""ZipBench adapter for CountBenchQA (batch B).

Upstream ``CountBenchQA.evaluate`` (image_vqa.py:2865) scores each row with a
plain substring test ``str(answer) in str(prediction)`` and dumps only the
aggregate ``{'accuracy': acc * 100}`` -- there is **no per-item file**. So we
recompute the per-item score with the byte-identical rule, materialise it to a
``_per_question`` file (so validate/smoke can hand-recompute from disk), and
weight it.

The upstream return value is a dict, so the weighted counterpart is the same
dict shape with the weighted number.
"""
from ..report import plain_mean, weighted_mean
from .base import ZipEvalMixin


class ZipCountBenchQAEvalMixin(ZipEvalMixin):

    ZIP_SCORE_COL = 'hit'
    ZIP_GROUPS = ()
    ZIP_METRIC_NAME = 'micro_accuracy'

    def zip_item_file(self, eval_file, **judge_kwargs):
        from ...smp import get_intermediate_file_path

        return get_intermediate_file_path(eval_file, '_per_question')

    def zip_item_frame(self, eval_file, base_result, **judge_kwargs):
        from ...smp import dump, load

        data = load(eval_file)
        # Byte-identical to CountBenchQA.evaluate (image_vqa.py:2872-2874).
        data[self.ZIP_SCORE_COL] = [
            1.0 if str(ans) in str(pred) else 0.0
            for pred, ans in zip(data['prediction'], data['answer'])
        ]
        dump(data, self.zip_item_file(eval_file, **judge_kwargs))
        return self.zip_attach_weights(data)

    def zip_aggregate(self, frame):
        score = weighted_mean(frame, self.ZIP_SCORE_COL)
        weighted = {'accuracy': score * 100}
        secondary = {
            'weighted_accuracy': score,
            'accuracy_unweighted': plain_mean(frame, self.ZIP_SCORE_COL),
            'num_samples': int(len(frame)),
        }
        return weighted, score, secondary

    # ------------------------------------------------------------ smoke hooks

    @staticmethod
    def zip_smoke_fake(ds, data, wrong_every=3):
        """Numeric gold predictions; the miss rows contain no gold digit."""
        preds = []
        for i, (_, row) in enumerate(data.iterrows()):
            gold = str(row['answer']).strip()
            preds.append('no digits here' if i % wrong_every == 0 else gold)
        data['prediction'] = preds
        return data
