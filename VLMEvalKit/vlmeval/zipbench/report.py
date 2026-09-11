"""Weighted aggregation helpers + the standard ZipBench result shape.

Two things live here:

1. ``weighted_report_acc`` -- a drop-in weighted counterpart of
   ``vlmeval/dataset/utils/multiple_choice.py::report_acc``. It returns a
   DataFrame with the *same shape* as the upstream one (one row per ``split``,
   one column per group value) so everything downstream -- ``_acc.csv``,
   run.py's tabulate, external scripts -- keeps working unchanged.

2. ``zip_result`` -- the machine-readable side-car written to
   ``<eval_file>_zip.json``. Same shape as the OpenCompass port
   (``opencompass/zipbench/result.py``) so cross-framework tooling can read
   both::

       {"score": 68.42, "metric": "...", "secondary": {...}}

Note: the weights target the *overall* full-set metric. Group breakdowns
renormalise the weights *within* the group; per-category cells are
diagnostics.
"""
from collections import defaultdict

WEIGHT_COL = 'zip_weight'


def weighted_mean(frame, score_col, weight_col=WEIGHT_COL):
    """``sum(w_i * s_i) / sum(w_i)``; ``nan`` for an empty / zero-weight frame."""
    import numpy as np

    if not len(frame):
        return float('nan')
    scores = np.asarray(frame[score_col], dtype=float)
    weights = np.asarray(frame[weight_col], dtype=float)
    total = weights.sum()
    if total <= 0:
        return float('nan')
    return float((scores * weights).sum() / total)


def plain_mean(frame, score_col):
    import numpy as np

    if not len(frame):
        return float('nan')
    return float(np.asarray(frame[score_col], dtype=float).mean())


def weighted_report_acc(df, score_col='hit', weight_col=WEIGHT_COL,
                        groups=('l2-category', 'category')):
    """Weighted counterpart of ``report_acc``; identical output shape.

    Mirrors the upstream iteration order ``[None, 'l2-category', 'category']``
    so that a group value appearing in both columns resolves the same way.
    Values are fractions in ``[0, 1]``, like upstream.
    """
    import pandas as pd

    from ..dataset.utils.multiple_choice import MMB_abbrs

    df = df.copy()
    if weight_col not in df:
        raise ValueError(f'no `{weight_col}` column -- was the dataset built with a subset?')

    res = defaultdict(list)
    if 'split' in df:
        splits = sorted(set(df['split']), key=str)
    else:
        df['split'] = ['none'] * len(df)
        splits = ['none']
    res['split'] = splits

    for group in (None, *groups):
        if group is None:
            res['Overall'] = [
                weighted_mean(df[df['split'] == sp], score_col, weight_col) for sp in splits
            ]
        elif group not in df:
            continue
        else:
            for ab in sorted(set(df[group]), key=str):
                ab_name = MMB_abbrs[ab] if ab in MMB_abbrs else ab
                sub = df[df[group] == ab]
                res[ab_name] = [
                    weighted_mean(sub[sub['split'] == sp], score_col, weight_col) for sp in splits
                ]
    return pd.DataFrame(res)


def group_breakdown(df, score_col, weight_col=WEIGHT_COL, groups=()):
    """``{group_col: {value: {weighted, unweighted, n, weight_sum}}}``."""
    import numpy as np

    out = {}
    for group in groups:
        if group not in df:
            continue
        cells = {}
        for value in sorted(set(df[group]), key=str):
            sub = df[df[group] == value]
            cells[str(value)] = {
                'weighted': weighted_mean(sub, score_col, weight_col),
                'unweighted': plain_mean(sub, score_col),
                'n': int(len(sub)),
                'weight_sum': float(np.asarray(sub[weight_col], dtype=float).sum()),
            }
        out[group] = cells
    return out


def zip_result(score, metric, secondary=None):
    """The standard ZipBench result dict written to ``<eval_file>_zip.json``.

    Args:
        score (float): the headline number -- the benchmark's official metric,
            computed on the weighted subset.
        metric (str): ``'<name> -- <one-line description>'``. Name the
            aggregation (micro/macro) and what the benchmark's official metric
            is, in one clause.
        secondary (dict): every other number -- the unweighted counterpart,
            counters, per-group breakdowns, and the subset's provenance
            (``subset`` / ``anchor_mae`` / ``ratio``).
    """
    return {
        'score': float(score),
        'metric': metric,
        'secondary': dict(secondary or {}),
    }
