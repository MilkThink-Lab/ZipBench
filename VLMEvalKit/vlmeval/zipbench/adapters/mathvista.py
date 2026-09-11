"""ZipBench adapter for MathVista_MINI (batch B).

Upstream ``MathVista.evaluate_heuristic`` (image_vqa.py:311) dumps the judged
per-item frame to ``get_intermediate_file_path(eval_file, f'_{model}')`` with
``res``/``log`` columns but **no hit column** -- per-item correctness is
recomputed on the fly by ``utils/mathvista.post_check(line, prefetch=False)``
inside ``MathVista_acc``. We do the same here and materialise it as a ``hit``
column before weighting. The verifier path (``_detailed_results``) already
carries a ``verifier_match`` column.

``MathVista_acc`` returns a DataFrame with rows Overall + one per skill + one
per task (a question counts into every skill it lists), columns
``['Task&Skill', 'tot', 'prefetch', 'hit', 'prefetch_rate', 'acc']`` with
``acc`` in percent. The weighted counterpart keeps that exact shape: ``acc``
becomes the within-group weighted accuracy (the Overall row is the
headline; per-task/skill rows renormalise within the
group and are diagnostics only), while ``tot``/``prefetch``/``hit`` stay
unweighted counts.
"""
from collections import defaultdict

import pandas as pd

from ..report import WEIGHT_COL, plain_mean, weighted_mean
from .base import ZipEvalMixin


def _item_skills(item):
    try:
        return list(eval(item['skills']))
    except SyntaxError:
        return [item['skills']]


class ZipMathVistaEvalMixin(ZipEvalMixin):

    ZIP_SCORE_COL = 'hit'
    ZIP_GROUPS = ('task',)
    ZIP_METRIC_NAME = 'micro_accuracy'

    def zip_item_file(self, eval_file, **judge_kwargs):
        from ...smp import get_intermediate_file_path

        if judge_kwargs.get('use_verifier', False):
            # evaluate_verifier() path (image_vqa.py:361)
            return get_intermediate_file_path(eval_file, '_detailed_results')
        # evaluate_heuristic() path (image_vqa.py:316); `model` is a hard
        # KeyError upstream too, so mirror that.
        return get_intermediate_file_path(eval_file, f"_{judge_kwargs['model']}")

    def zip_item_frame(self, eval_file, base_result, **judge_kwargs):
        if judge_kwargs.get('use_verifier', False):
            self.ZIP_SCORE_COL = 'verifier_match'
        frame = super().zip_item_frame(eval_file, base_result, **judge_kwargs)
        if self.ZIP_SCORE_COL not in frame:
            from ...dataset.utils.mathvista import post_check
            frame[self.ZIP_SCORE_COL] = [
                float(post_check(frame.iloc[i], prefetch=False))
                for i in range(len(frame))
            ]
        return frame

    def zip_aggregate(self, frame):
        score_col = self.ZIP_SCORE_COL
        score = weighted_mean(frame, score_col)

        # Same row universe as MathVista_acc: Overall, then skills and tasks
        # in first-seen order; a question contributes to every skill it lists.
        groups = defaultdict(list)   # key -> list of row positions
        has_skills = 'skills' in frame
        for i in range(len(frame)):
            item = frame.iloc[i]
            groups['Overall'].append(i)
            if has_skills:
                for skill in _item_skills(item):
                    groups[skill].append(i)
            if 'task' in frame:
                groups[item['task']].append(i)

        prefetch_ok = (frame['log'] == 'Prefetch succeed') if 'log' in frame else None
        res = defaultdict(list)
        for k, rows in groups.items():
            sub = frame.iloc[rows]
            n_fetch = int(prefetch_ok.iloc[rows].sum()) if prefetch_ok is not None else 0
            res['Task&Skill'].append(k)
            res['tot'].append(len(sub))
            res['prefetch'].append(n_fetch)
            res['hit'].append(int(sub[score_col].sum()))
            res['prefetch_rate'].append(n_fetch / len(sub) * 100)
            res['acc'].append(weighted_mean(sub, score_col) * 100)
        weighted = pd.DataFrame(res)

        secondary = {
            'weighted_accuracy': score,
            'accuracy_unweighted': plain_mean(frame, score_col),
            'num_samples': int(len(frame)),
            'per_group': {
                'Task&Skill': {
                    str(k): {
                        'weighted': weighted_mean(frame.iloc[rows], score_col),
                        'unweighted': plain_mean(frame.iloc[rows], score_col),
                        'n': len(rows),
                        'weight_sum': float(frame.iloc[rows][WEIGHT_COL].sum()),
                    }
                    for k, rows in groups.items() if k != 'Overall'
                }
            },
        }
        return weighted, score, secondary

    # ------------------------------------------------------------ smoke hooks

    @staticmethod
    def zip_smoke_fake(ds, data, wrong_every=3):
        """Gold predictions with a deliberate miss every Nth row."""
        preds = []
        for i, (_, row) in enumerate(data.iterrows()):
            if row['question_type'] == 'multi_choice':
                gold, bad = str(row['answer_option']).strip(), 'Z'
            else:
                gold, bad = str(row['answer']).strip(), '999999999'
            preds.append(bad if i % wrong_every == 0 else gold)
        data['prediction'] = preds
        return data

    @staticmethod
    def zip_smoke_seed(ds, eval_file, judge_kwargs):
        """Pre-seed the judged storage file so evaluate() skips the LLM judge.

        ``res`` is what ``post_check(prefetch=False)`` scores, so echoing the
        prediction (which zip_smoke_fake made either exactly right or surely
        wrong) reproduces the intended hit pattern.
        """
        from ...smp import dump, load

        storage = ZipMathVistaEvalMixin.zip_item_file(ds, eval_file, **judge_kwargs)
        data = load(eval_file)
        data['res'] = [str(p) for p in data['prediction']]
        data['log'] = ['Prefetch succeed'] * len(data)
        dump(data, storage)
