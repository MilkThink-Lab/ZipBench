"""ZipBench adapter for LogicVista (batch B).

Upstream ``LogicVista.evaluate`` (image_vqa.py:1276) dumps the judged storage
to ``get_intermediate_file_path(eval_file, f'_{name_str}')`` with a ready-made
0/1 ``hit`` column (image_vqa.py:1335). ``evaluate_logicvista``
(utils/logicvista.py:75) aggregates it into a DataFrame
``['Task&Skill', 'tot', 'hit', 'acc']`` -- Overall plus five skill rows
selected by substring match on the comma-separated ``skill`` column (a
question can count into several skills), ``acc`` in percent.

The weighted counterpart keeps that shape: ``acc`` becomes the within-group
weighted accuracy (Overall is the headline; skill rows
renormalise within the group, diagnostics only), ``tot``/``hit`` stay
unweighted counts.

Caveat: with ``--judge exact_matching`` and no pre-existing storage the
upstream evaluate() returns ``None`` without writing anything -- the base
mixin then raises FileNotFoundError, which is the right signal (LogicVista
needs an LLM judge).
"""
from collections import defaultdict

import pandas as pd

from ..report import WEIGHT_COL, plain_mean, weighted_mean
from .base import ZipEvalMixin

#: Mirrors ``LogicVista.evaluate`` (image_vqa.py:1281-1286).
_JUDGE_NAME_MAP = {
    'gpt-4-0125': 'gpt4',
    'gpt-4-turbo': 'gpt4-turbo',
    'gpt-4o-mini': 'gpt4o-mini',
}

#: Skill rows of ``evaluate_logicvista``, in upstream order.
_SKILLS = ('inductive', 'deductive', 'numerical', 'spatial', 'mechanical')


class ZipLogicVistaEvalMixin(ZipEvalMixin):

    ZIP_SCORE_COL = 'hit'
    ZIP_GROUPS = ()
    ZIP_METRIC_NAME = 'micro_accuracy'

    def zip_item_file(self, eval_file, **judge_kwargs):
        from ...smp import get_intermediate_file_path

        judge = judge_kwargs.get('model', 'exact_matching')
        name_str = _JUDGE_NAME_MAP.get(judge, judge)
        return get_intermediate_file_path(eval_file, f'_{name_str}')

    def zip_aggregate(self, frame):
        score_col = self.ZIP_SCORE_COL
        score = weighted_mean(frame, score_col)

        groups = {'Overall': frame}
        if 'skill' in frame:
            for sk in _SKILLS:
                groups[sk] = frame[frame['skill'].str.contains(sk)]

        res = defaultdict(list)
        for k, sub in groups.items():
            if not len(sub):
                continue
            res['Task&Skill'].append(k)
            res['tot'].append(int(len(sub)))
            res['hit'].append(int(sub[score_col].sum()))
            res['acc'].append(weighted_mean(sub, score_col) * 100)
        weighted = pd.DataFrame(res)

        secondary = {
            'weighted_accuracy': score,
            'accuracy_unweighted': plain_mean(frame, score_col),
            'num_samples': int(len(frame)),
            'per_group': {
                'skill': {
                    k: {
                        'weighted': weighted_mean(sub, score_col),
                        'unweighted': plain_mean(sub, score_col),
                        'n': int(len(sub)),
                        'weight_sum': float(sub[WEIGHT_COL].sum()),
                    }
                    for k, sub in groups.items()
                    if k != 'Overall' and len(sub)
                }
            },
        }
        return weighted, score, secondary

    # ------------------------------------------------------------ smoke hooks

    @staticmethod
    def zip_smoke_fake(ds, data, wrong_every=3):
        """Gold letter predictions with a deliberate miss every Nth row."""
        preds = []
        for i, (_, row) in enumerate(data.iterrows()):
            gold = str(row['answer']).strip()
            preds.append(('A' if gold != 'A' else 'B') if i % wrong_every == 0 else gold)
        data['prediction'] = preds
        return data

    @staticmethod
    def zip_smoke_seed(ds, eval_file, judge_kwargs):
        """Pre-seed the judged storage so evaluate() skips the LLM judge.

        ``hit`` replicates ``LogicVista_auxeval``'s criterion: the sorted
        letter set of the prediction must equal the sorted gold set.
        """
        from ...smp import dump, load

        storage = ZipLogicVistaEvalMixin.zip_item_file(ds, eval_file, **judge_kwargs)
        data = load(eval_file)
        hits = []
        for _, row in data.iterrows():
            pred = sorted(str(row['prediction']).replace(',', ' ').split())
            gold = sorted(str(row['answer']).split(', '))
            hits.append(1 if pred == gold else 0)
        data['res'] = [str(p) for p in data['prediction']]
        data['log'] = ['smoke seed'] * len(data)
        data['hit'] = hits
        dump(data, storage)
