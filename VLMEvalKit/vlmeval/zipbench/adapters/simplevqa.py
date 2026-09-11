"""SimpleVQA adapter (batch C2) -- LLM-judged correct/incorrect/not-attempted
verdicts, headline F1.

The official scorer (``utils/simplevqa.py::SimpleVQAEval``) turns per-item
judge verdicts into three *proportions over all samples* (rounded to 4
decimals), then plugs them into the SimpleQA F1 formula. The weighted
counterpart replaces the three proportions with their subset-weighted versions
and reuses the formula line for line, including the ``round(., 4)`` calls.
The weighted correct-proportion is also reported as
``secondary.weighted_accuracy``.

Unmappable verdicts (e.g. the judge's parse-failure marker) stay in the denominator but in
none of the three buckets, exactly like upstream -- the three proportions may
sum to less than 1.
"""
import json
import os
import os.path as osp

from ..report import WEIGHT_COL, group_breakdown
from .base import ZipEvalMixin

VERDICTS = ('正确', '错误', '未尝试')


def _official_f1(c, i):
    """The SimpleVQAEval aggregation from the two rounded proportions."""
    ga = round(c + i, 4)
    a = c / ga if ga > 0 else 0
    f1 = 2 * a * c / (a + c) if (a + c) > 0 else 0
    return ga, a, f1


def _short_category(raw):
    """``vqa_category`` cells are JSON blobs; keep only ``task_category``."""
    try:
        return json.loads(raw).get('task_category', str(raw))
    except Exception:
        return str(raw)


class ZipSimpleVQAEvalMixin(ZipEvalMixin):

    ZIP_SCORE_COL = 'hit'                      # 1 == a "correct" verdict
    ZIP_GROUPS = ('language', 'vqa_category')  # diagnostics only
    ZIP_METRIC_NAME = 'official_f1'
    #: items-replay compares both headline entries.
    ZIP_HEADLINE_LABELS = {'f1': 'f1', 'acc': 'acc'}

    # ------------------------------------------------------------------ hooks

    def zip_item_file(self, eval_file, **judge_kwargs):
        from ...smp import get_intermediate_file_path

        return get_intermediate_file_path(eval_file, '_gpt_eval', 'json')

    def zip_prepare_item_frame(self, frame):
        """Normalise verdicts and materialise the three 0/1 columns.

        Accepts both the new ``_gpt_eval.json`` records (dict ``judge_res``
        plus a ``conclusion`` column) and the mining-side normalised files
        (``judge_res`` is a str repr of a dict, no ``conclusion``).
        """
        import ast

        from ...dataset.utils.simplevqa import mapper, normalize_judge_conclusion

        frame = frame.copy()
        if 'conclusion' in frame:
            conclusions = [str(c) for c in frame['conclusion']]
        else:
            conclusions = []
            for judge_res in frame['judge_res']:
                if isinstance(judge_res, str):
                    judge_res = ast.literal_eval(judge_res)
                if 'model_response' in judge_res:
                    conclusions.append(normalize_judge_conclusion(judge_res['model_response']))
                else:
                    conclusions.append('答案解析失败')
        buckets = []
        for concl in conclusions:
            if concl not in mapper:
                print(f'Error in Mapper Key: {concl}')
            buckets.append(mapper.get(concl))
        frame['conclusion'] = conclusions
        frame['hit'] = [1.0 if b == 'is_correct' else 0.0 for b in buckets]
        frame['is_incorrect'] = [1.0 if b == 'is_incorrect' else 0.0 for b in buckets]
        frame['is_not_attempted'] = [1.0 if b == 'is_not_attempted' else 0.0 for b in buckets]
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
        for col in self.ZIP_GROUPS:
            if col in self.data and col not in frame:
                values = {i: v for i, v in zip(self.data['index'], self.data[col])}
                frame[col] = [values[i] for i in frame['index']]
        if 'vqa_category' in frame:
            frame['vqa_category'] = [_short_category(v) for v in frame['vqa_category']]
        return frame

    def zip_aggregate(self, frame):
        import numpy as np
        import pandas as pd

        weights = np.asarray(frame[WEIGHT_COL], dtype=float)

        def proportions(w):
            total = w.sum()
            return tuple(
                round(float((np.asarray(frame[col], dtype=float) * w).sum() / total), 4)
                for col in ('hit', 'is_incorrect', 'is_not_attempted'))

        c, i, n = proportions(weights)
        ga, a, f1 = _official_f1(c, i)
        cu, iu, nu = proportions(np.ones(len(frame)))
        _, _, f1_u = _official_f1(cu, iu)

        # Same shape as the upstream return: pd.DataFrame(list(dict.items())).
        weighted = pd.DataFrame(list({
            'LVLM_name': 'model_response',
            'is_correct': c,
            'is_incorrect': i,
            'is_not_attempted': n,
            'is_given_attempted': ga,
            'accuracy_given_attempted': a,
            'f1': f1,
        }.items()))

        secondary = {
            'weighted_accuracy': c,
            'accuracy_unweighted': cu,
            'f1_unweighted': f1_u,
            'accuracy_given_attempted': a,
            'is_incorrect': i,
            'is_not_attempted': n,
            'headline': {'f1': f1, 'acc': c},
            'num_samples': int(len(frame)),
            'per_group': group_breakdown(frame, self.ZIP_SCORE_COL, groups=self.ZIP_GROUPS),
        }
        return weighted, f1, secondary

    # -------------------------------------------------------------- evaluate

    def evaluate(self, eval_file, **judge_kwargs):
        from ...smp import dump, get_intermediate_file_path

        weighted = super().evaluate(eval_file, **judge_kwargs)
        # The upstream evaluate() wrote its unweighted aggregation to _acc.csv;
        # move it aside and put the weighted official numbers in its place.
        acc_pth = get_intermediate_file_path(eval_file, '_acc', 'csv')
        if osp.exists(acc_pth):
            os.replace(acc_pth, get_intermediate_file_path(eval_file, '_acc_unweighted', 'csv'))
        dump(weighted, acc_pth)
        return weighted

    # ----------------------------------------------------------- smoke hooks

    @staticmethod
    def zip_smoke_fake(ds, data, wrong_every=3):
        """Echo the gold answer; the seeded judge file decides the verdicts."""
        data['prediction'] = [str(a) for a in data['answer']]
        return data

    @staticmethod
    def zip_smoke_seed(ds, eval_file, judge_kwargs):
        """Pre-write ``_gpt_eval.json`` (cycling through the three verdicts) so the
        upstream evaluate() skips the judge entirely -- a zero-API smoke."""
        from ...smp import get_intermediate_file_path, load

        data = load(eval_file)
        records = []
        for pos, (_, row) in enumerate(data.iterrows()):
            verdict = VERDICTS[pos % len(VERDICTS)]
            records.append({
                'index': int(row['index']),
                'question': row['question'],
                'answer': row['answer'],
                'model_response': row['prediction'],
                'judge_res': {'model_response': verdict},
                'conclusion': verdict,
            })
        target = get_intermediate_file_path(eval_file, '_gpt_eval', 'json')
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)

    @staticmethod
    def zip_smoke_expected(ds, scored, weights):
        """Independent recompute of the weighted F1 from raw conclusions."""
        acc = {v: 0.0 for v in VERDICTS}
        total = 0.0
        for idx, concl in zip(scored['index'], scored['conclusion']):
            w = weights[idx]
            total += w
            if concl in acc:
                acc[concl] += w
        c = round(acc['正确'] / total, 4)
        i = round(acc['错误'] / total, 4)
        ga = round(c + i, 4)
        a = c / ga if ga > 0 else 0
        return 2 * a * c / (a + c) if (a + c) > 0 else 0
