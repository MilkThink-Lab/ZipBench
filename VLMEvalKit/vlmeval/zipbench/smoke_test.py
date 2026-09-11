#!/usr/bin/env python3
"""End-to-end smoke test of the ZipBench evaluate() path on synthetic predictions.

``validate.py`` checks that the *spec* is right. This checks that the *adapter*
is right: it fabricates a prediction file from the gold answers (deliberately
wrong on every Nth row so the accuracy is not 1.0), runs the real
``dataset.evaluate()``, and asserts that

* the returned object has the same shape as the upstream one;
* ``<eval_file>_zip.json`` carries score / metric / secondary / provenance;
* the headline score equals a hand-computed ``sum(w*hit)/sum(w)`` over the
  upstream per-item file, bit-exactly;
* the weighted and unweighted numbers actually differ (i.e. the weights are
  reaching the aggregation at all);
* a ``--subset full`` run is untouched -- no ``_zip.json``, no mixin.

Costs nothing: the default judge is ``exact_matching``.

Usage::

    python -m vlmeval.zipbench.smoke_test --dataset MMStar --subset small
"""
import argparse
import json
import os.path as osp
import tempfile
import warnings

warnings.filterwarnings('ignore')

import numpy as np


def _fake_predictions(ds, answer_col='answer', wrong_every=3, hook=None):
    """Predict the gold answer, except on every Nth row (deterministic).

    An adapter may provide a ``zip_smoke_fake(ds, data, wrong_every)`` hook to
    fabricate predictions in the benchmark's own answer format (non-MCQ
    datasets); pass it as ``hook``. The default assumes MCQ letters.
    """
    data = ds.data.copy()
    if hook is not None:
        data = hook(ds, data, wrong_every)
    else:
        preds = []
        for i, (_, row) in enumerate(data.iterrows()):
            gold = str(row[answer_col]).strip()
            preds.append(('A' if gold != 'A' else 'B') if i % wrong_every == 0 else gold)
        data['prediction'] = preds
    if 'image' in data:
        data.pop('image')
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--subset', default='small')
    ap.add_argument('--judge', default='exact_matching')
    ap.add_argument('--score-col', default=None, help='default: the adapter\'s ZIP_SCORE_COL')
    ap.add_argument('--out-dir', default=None, help='default: a temp dir')
    ap.add_argument('--skip-full', action='store_true',
                    help='skip the full-set regression run (expensive scorers)')
    args = ap.parse_args()

    from ..dataset import build_dataset
    from ..smp import dump, get_intermediate_file_path, load

    out_dir = args.out_dir or tempfile.mkdtemp(prefix='zipbench_smoke_')
    judge_kwargs = dict(model=args.judge, nproc=4)

    sub = build_dataset(args.dataset, subset=args.subset)
    assert getattr(sub, 'zip_subset', None) == args.subset
    assert 'zip_weight' in sub.data, 'weight column missing from dataset.data'
    print(f'{args.dataset}/{args.subset}: {len(sub.data)} rows, class={type(sub).__name__}')

    fake_hook = getattr(type(sub), 'zip_smoke_fake', None)
    seed_hook = getattr(type(sub), 'zip_smoke_seed', None)

    preds = _fake_predictions(sub, hook=fake_hook)
    assert 'zip_weight' in preds, 'weight column did not survive into the prediction frame'
    eval_file = osp.join(out_dir, f'SmokeModel_{args.dataset}_ZIP_{args.subset}.xlsx')
    dump(preds, eval_file)
    if seed_hook is not None:
        # Pre-seed the upstream per-item file so evaluate() needs no judge API.
        seed_hook(sub, eval_file, judge_kwargs)

    result = sub.evaluate(eval_file, **judge_kwargs)
    report = json.load(open(get_intermediate_file_path(eval_file, '_zip', 'json')))
    sec = report['secondary']
    print(f"  score={report['score']:.6f}  unweighted={sec['accuracy_unweighted']:.6f}  "
          f"n={sec['num_samples']}")
    print(f"  metric: {report['metric']}")

    import pandas as pd

    score_col = args.score_col or sub.ZIP_SCORE_COL
    item_file = sub.zip_item_file(eval_file, **judge_kwargs)
    scored = load(item_file)
    if not isinstance(scored, pd.DataFrame):
        scored = pd.DataFrame(scored)
    if score_col not in scored:
        # e.g. MathVista: the upstream storage has no hit column; the adapter
        # recomputes it in zip_item_frame. Weights below still come from
        # sub.data, so the aggregation itself is recomputed independently.
        scored = sub.zip_item_frame(eval_file, None, **judge_kwargs)
    weights = dict(zip(sub.data['index'], sub.data['zip_weight']))
    values = np.asarray([float(v) for v in scored[score_col]])
    ws = np.asarray([weights[i] for i in scored['index']])
    expected_hook = getattr(type(sub), 'zip_smoke_expected', None)
    if expected_hook is not None:
        # Datasets whose headline is not the plain weighted mean (e.g.
        # OCRBench_v2's task macro-average) supply their own small,
        # independent implementation of the expected headline.
        manual = float(expected_hook(sub, scored, weights))
    else:
        manual = float((values * ws).sum() / ws.sum())

    assert abs(manual - report['score']) < 1e-12, f'score {report["score"]} != manual {manual}'
    assert sec['num_samples'] == len(sub.data)
    assert sec['weighted_accuracy'] != sec['accuracy_unweighted'], \
        'weighted == unweighted; the weights are not reaching the aggregation'
    assert sec['zipbench']['subset'] == args.subset
    print(f'  manual recompute matches ({manual:.12f})')

    if isinstance(result, pd.DataFrame) and 'Overall' in result:
        if len(result) == 1:
            assert abs(float(result['Overall'].iloc[0]) - manual) < 1e-12, \
                'returned DataFrame Overall != headline score'
        else:
            # Multi-split frame (e.g. MMMU dev/validation): each row's Overall
            # is the within-split weighted mean; the combined headline lives in
            # the _zip.json only. Recompute every row by hand.
            split_of = dict(zip(sub.data['index'], sub.data['split']))
            for sp, row_val in zip(result['split'], result['Overall']):
                m = np.asarray([split_of[i] == sp for i in scored['index']])
                expect = float((values[m] * ws[m]).sum() / ws[m].sum())
                assert abs(float(row_val) - expect) < 1e-12, \
                    f'split {sp!r}: Overall {row_val} != manual {expect}'
                sec_key = f'weighted_accuracy_{sp}'
                if sec_key in sec:
                    assert abs(sec[sec_key] - expect) < 1e-12, \
                        f'secondary {sec_key} != per-split manual value'
            print(f'  per-split rows verified: {list(result["split"])}')
        print(f'  returned DataFrame: {list(result.columns)[:6]}{"..." if len(result.columns) > 6 else ""}')
    elif isinstance(result, dict) and 'headline' in sec:
        # Multi-headline dict result (e.g. OCRBench_v2 en/cn): the returned
        # dict must carry the same numbers as the _zip.json headline.
        for key, val in sec['headline'].items():
            assert abs(float(result[key]) - val) < 1e-12, \
                f'returned dict {key!r} = {result[key]} != _zip.json headline {val}'
        print(f'  headline keys verified: {list(sec["headline"])}')

    if args.skip_full:
        print('\nALL SMOKE CHECKS PASSED (full-set regression skipped)')
        return

    # `full` must be byte-identical to pre-patch behaviour.
    full = build_dataset(args.dataset)
    assert getattr(full, 'zip_subset', None) is None
    full_file = osp.join(out_dir, f'SmokeModel_{args.dataset}.xlsx')
    dump(_fake_predictions(full, hook=fake_hook), full_file)
    if seed_hook is not None:
        seed_hook(full, full_file, judge_kwargs)
    full.evaluate(full_file, **judge_kwargs)
    assert not osp.exists(get_intermediate_file_path(full_file, '_zip', 'json')), \
        'a full run must not emit a _zip.json'
    print('  full-set path clean (no mixin, no _zip.json)')

    print('\nALL SMOKE CHECKS PASSED')


if __name__ == '__main__':
    main()
