#!/usr/bin/env python3
"""Offline acceptance checks for a ZipBench subset. No GPU, no judge API.

Three checks, in increasing strength:

1. **build** -- ``build_dataset(name, subset=...)`` yields exactly ``n`` rows,
   the weights sum to 1, the TSV md5 matches the manifest, no qhash drift.

2. **mae** -- replay the subset against a mining correctness matrix pkl: for
   every model in the matrix, compare its weighted-subset accuracy against its
   full-set accuracy. The mean absolute difference must equal the manifest's
   ``anchor_mae``.

   This is the strongest offline signal available: it round-trips
   ``spec index value -> row position -> matrix row``, so it fails loudly if the
   anchor was mined in a different row order than the TSV.

3. **replay** -- given an existing *full-set* prediction file, slice it down to
   the subset, score both, and report ``|weighted_subset - full|``. This is the
   end-to-end number that says whether the compression actually works for a
   held-out model, and it costs nothing because the predictions already exist.

4. **items-replay** (``--items-file``) -- like replay, but starts from the
   per-item scores file a *full-set* ``evaluate()`` already dumped, so it never
   re-runs the scoring at all: pure arithmetic. Essential for benchmarks with
   expensive scorers or paid judges.

Usage::

    python -m vlmeval.zipbench.validate --dataset RealWorldQA \\
        --record-pkl path/to/RealWorldQA_correctness_matrix.pkl \\
        --pred-file outputs/<model>/<eval_id>/<model>_RealWorldQA.xlsx

    # anchors with a non-row position space + merged en/cn spec:
    python -m vlmeval.zipbench.validate --dataset OCRBench_v2 \\
        --record ocrbench_v2_en=path/to/en_matrix.pkl \\
        --record ocrbench_v2_cn=path/to/cn_matrix.pkl \\
        --items-file outputs/.../<model>_OCRBench_v2_per_question.json
"""
import argparse
import os.path as osp
import pickle

import numpy as np

from .layouts import get_layout
from .spec import (available_subsets, check_qhash, load_manifest,
                   load_subset_spec)

OK, FAIL = '  [ok]  ', '  [FAIL]'


def _load_record(record_pkl, layout, dataset_key=None):
    """-> ``(matrix[n_items, n_models], model_names)`` in layout position order."""
    with open(osp.expanduser(record_pkl), 'rb') as f:
        payload = pickle.load(f)
    models = payload.get('models')
    data = payload['data']
    if dataset_key is not None:
        data = {dataset_key: data[dataset_key]}
    matrix = layout.stack_record(data)
    if models is not None and matrix.shape[0] == len(models) and matrix.shape[1] != len(models):
        matrix = matrix.T  # normalise to (n_items, n_models)
    return matrix, models


def check_build(dataset, subset, results):
    from ..dataset import build_dataset

    manifest = load_manifest(dataset)
    spec = load_subset_spec(dataset, subset)
    expected = manifest['subsets'][subset]

    full = build_dataset(dataset)
    n_total = len(full.data)
    if n_total != manifest['dataset_size']:
        results.append((False, f'build/{subset}: dataset has {n_total} rows, '
                               f"manifest says {manifest['dataset_size']}"))
        return None, None

    drift = check_qhash(full.data, spec, dataset=dataset)
    if drift:
        results.append((False, f'build/{subset}: {len(drift)} question(s) drifted, first={drift[:2]}'))

    ds = build_dataset(dataset, subset=subset)
    n = len(ds.data)
    weights = np.asarray(ds.data['zip_weight'], dtype=float)
    wsum = float(weights.sum())
    norm = manifest.get('weight_normalization', 'global')
    if norm == 'global':
        ok = (n == expected['n']) and abs(wsum - 1.0) < 1e-9
        results.append((ok, f'build/{subset}: n={n} (expected {expected["n"]}), weight_sum={wsum:.12f}'))
    else:
        # per_group: every task group sums to 1, the total equals the group count.
        groups = np.asarray(ds.zip_weight_groups(ds.data))
        names = sorted(set(groups.tolist()))
        bad = {g: float(weights[groups == g].sum()) for g in names
               if abs(weights[groups == g].sum() - 1.0) > 1e-6}
        ok = (n == expected['n']) and not bad and abs(wsum - len(names)) < 1e-6
        results.append((ok, f'build/{subset}: n={n} (expected {expected["n"]}), '
                            f'{len(names)} groups each summing to 1'
                            + (f', BAD groups: {bad}' if bad else '')
                            + f', total={wsum:.9f}'))

    if manifest.get('selection_unit') == 'pair':
        # Whole-pair invariants. Row count alone cannot catch a spec that was
        # converted without --selection-unit pair (tiny: 150 pair indices
        # misread as rows would also yield 150 rows), hence the completeness
        # check on every pair.
        idx = np.asarray(ds.data['index'], dtype=int)
        pair_ids = (idx - 1) // 2
        uniq, counts = np.unique(pair_ids, return_counts=True)
        incomplete = uniq[counts != 2].tolist()
        uneven = [int(p) for p in uniq[counts == 2]
                  if abs(np.diff(weights[pair_ids == p])[0]) > 1e-12]
        expected_pairs = expected.get('n_pairs')
        ok = (n % 2 == 0 and not incomplete and not uneven
              and (expected_pairs is None or len(uniq) == expected_pairs))
        results.append((ok, f'build/{subset}: {len(uniq)} pairs (expected {expected_pairs}), '
                            f'{len(incomplete)} incomplete, {len(uneven)} with unequal row weights'))

    tsv_md5 = getattr(type(full), 'DATASET_MD5', {}).get(dataset)
    if manifest.get('tsv_md5') and tsv_md5:
        results.append((manifest['tsv_md5'] == tsv_md5,
                        f'build/{subset}: tsv_md5 {tsv_md5} vs manifest {manifest["tsv_md5"]}'))
    return full, ds


def check_mae(dataset, subset, full, layout, record_pkl, record_key, results):
    manifest = load_manifest(dataset)
    spec = load_subset_spec(dataset, subset)
    expected_mae = manifest['subsets'][subset]['anchor_mae']
    label = layout.label
    if isinstance(expected_mae, dict):
        expected_mae = expected_mae[label]
    tag = f'mae/{subset}' + (f'/{label}' if label else '')

    matrix, models = _load_record(record_pkl, layout, record_key)
    if manifest.get('selection_unit') == 'pair':
        # Pair-mined subset: the record matrix lives in the PAIR space
        # (pair value = both rows correct). Fold the spec's rows back into pairs -- weight
        # is the two rows summed, position is (index - 1) // 2 -- which
        # end-to-end verifies the pair-index -> rows -> index-value mapping.
        n_pairs = len(full.data) // 2
        if matrix.shape[0] != n_pairs:
            results.append((False, f'{tag}: matrix has {matrix.shape[0]} items but the dataset '
                                   f'has {n_pairs} pairs -- not a pair-space record?'))
            return
        pair_w = {}
        for r in spec:
            p = (int(r['index']) - 1) // 2
            pair_w[p] = pair_w.get(p, 0.0) + r['weight']
        in_layout = sorted(pair_w)
        positions = np.array(in_layout)
        weights = np.array([pair_w[p] for p in in_layout], dtype=float)
        groups = None
    else:
        pos_index = layout.positions(full.data)
        pos_groups = layout.groups(full.data)
        if matrix.shape[0] != len(pos_index):
            results.append((False, f'{tag}: matrix has {matrix.shape[0]} items but the layout '
                                   f'has {len(pos_index)} positions -- different population'))
            return

        # A merged spec (e.g. OCRBench_v2 en+cn) holds rows outside this layout's
        # position space; only its own rows replay against this record.
        position = {idx: pos for pos, idx in enumerate(pos_index)}
        in_layout = [r for r in spec if r['index'] in position]
        if layout.GROUPS is None and len(in_layout) != len(spec):
            results.append((False, f'{tag}: {len(spec) - len(in_layout)} spec rows are not in the dataset'))
            return
        if not in_layout:
            results.append((False, f'{tag}: no spec rows fall inside this layout'))
            return
        positions = np.array([position[r['index']] for r in in_layout])
        weights = np.array([r['weight'] for r in in_layout], dtype=float)
        groups = None if pos_groups is None else np.asarray(pos_groups)

    gaps = []
    for m in range(matrix.shape[1]):
        full_acc = layout.aggregate(matrix[:, m], np.ones(matrix.shape[0]),
                                    None if groups is None else groups)
        sub_acc = layout.aggregate(matrix[positions, m], weights,
                                   None if groups is None else groups[positions])
        gaps.append(sub_acc - full_acc)
    gaps = np.asarray(gaps)
    mae = float(np.abs(gaps).mean())
    ok = abs(mae - expected_mae) < 5e-4
    results.append((ok, f'{tag}: reproduced {mae:.6f} vs manifest {expected_mae} '
                        f'over {matrix.shape[1]} models (n={len(in_layout)})'))
    if not ok:
        worst = np.argsort(-np.abs(gaps))[:3]
        detail = ', '.join(
            f'{(models[i] if models else i)}: gap {gaps[i]:+.4f}' for i in worst)
        results.append((False, f'{tag}: worst models -- {detail}'))


def check_replay(dataset, subset, pred_file, judge, results):
    """Slice an existing full-set prediction file down to the subset and score both."""
    from ..dataset import build_dataset
    from ..smp import dump, load

    from . import WEIGHT_COL, zip_tag

    spec = load_subset_spec(dataset, subset)
    weights = {r['index']: r['weight'] for r in spec}

    preds = load(pred_file)
    sub = preds[preds['index'].isin(weights)].copy()
    if len(sub) != len(spec):
        results.append((False, f'replay/{subset}: prediction file covers {len(sub)}/{len(spec)} '
                               'subset rows -- is it a full-set prediction file?'))
        return
    sub[WEIGHT_COL] = [weights[i] for i in sub['index']]

    root, ext = osp.splitext(pred_file)
    sub_file = f'{root}{zip_tag(subset)}{ext}'
    dump(sub, sub_file)

    judge_kwargs = dict(model=judge, nproc=4, verbose=False)
    full_res = build_dataset(dataset).evaluate(pred_file, **judge_kwargs)
    sub_ds = build_dataset(dataset, subset=subset)
    sub_res = sub_ds.evaluate(sub_file, **judge_kwargs)

    full_overall = _overall(full_res)
    sub_overall = _overall(sub_res)
    import pandas as pd
    if isinstance(sub_res, pd.DataFrame) and len(sub_res) > 1:
        # Multi-split acc frame (e.g. MMMU dev/validation): the row mean of
        # `Overall` is not the headline. Compare the combined micro instead --
        # weighted from the _zip.json side-car, plain from the full-set
        # per-item file (same path logic as the adapter, keyed on eval_file).
        from ..smp import get_intermediate_file_path, load as _load
        sub_overall = _load(get_intermediate_file_path(sub_file, '_zip', 'json'))['score']
        item_file = sub_ds.zip_item_file(pred_file, **judge_kwargs)
        full_items = _load(item_file)
        full_overall = float(np.asarray(full_items[sub_ds.ZIP_SCORE_COL], dtype=float).mean())
    if full_overall is None or sub_overall is None:
        results.append((False, f'replay/{subset}: could not read Overall from the results'))
        return
    gap = abs(sub_overall - full_overall)
    anchor_mae = load_manifest(dataset)['subsets'][subset]['anchor_mae']
    results.append((gap <= 3 * anchor_mae,
                    f'replay/{subset}: weighted={sub_overall:.4f} full={full_overall:.4f} '
                    f'|gap|={gap:.4f} (anchor_mae={anchor_mae})'))


def check_items_replay(dataset, subset, items_file, results):
    """Replay from a full-set run's per-item scores file: pure arithmetic.

    No evaluate() is re-run, so this costs seconds even for benchmarks whose
    scoring takes tens of minutes or burns judge tokens. The full-set headline
    is the adapter's own aggregation with unit weights; the subset headline is
    the same aggregation over the spec rows with the mined weights.
    """
    import pandas as pd

    from ..dataset import build_dataset
    from ..smp import load
    from .report import WEIGHT_COL

    manifest = load_manifest(dataset)
    spec = load_subset_spec(dataset, subset)

    sub_ds = build_dataset(dataset, subset=subset)
    frame = load(items_file)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    prepare = getattr(sub_ds, 'zip_prepare_item_frame', None)
    if prepare is not None:
        frame = prepare(frame)
    if 'index' not in frame:
        # Legacy per-item files carry no index; fall back to row position.
        # Only sound when the file's row order equals the TSV order.
        print('  note: items file has no `index` column; assuming row order == TSV order')
        frame['index'] = range(len(frame))

    if len(frame) != manifest['dataset_size']:
        results.append((False, f'items-replay/{subset}: items file has {len(frame)} rows, '
                               f"the full set has {manifest['dataset_size']} -- not a full-set run?"))
        return

    full_frame = frame.copy()
    full_frame[WEIGHT_COL] = 1.0
    _, full_score, full_sec = sub_ds.zip_aggregate(full_frame)

    weights = {r['index']: r['weight'] for r in spec}
    sub_frame = frame[frame['index'].isin(weights)].copy()
    if len(sub_frame) != len(spec):
        results.append((False, f'items-replay/{subset}: items file covers {len(sub_frame)}/'
                               f'{len(spec)} spec rows'))
        return
    sub_frame[WEIGHT_COL] = [weights[i] for i in sub_frame['index']]
    _, sub_score, sub_sec = sub_ds.zip_aggregate(sub_frame)

    full_head = full_sec.get('headline', {'score': full_score})
    sub_head = sub_sec.get('headline', {'score': sub_score})

    info = manifest['subsets'][subset]
    tol_mae = info['anchor_mae']
    labels = getattr(sub_ds, 'ZIP_HEADLINE_LABELS', None) or {}
    for key, sub_val in sub_head.items():
        full_val = full_head[key]
        mae = tol_mae[labels[key]] if isinstance(tol_mae, dict) else tol_mae
        gap = abs(sub_val - full_val)
        results.append((gap <= 3 * mae,
                        f'items-replay/{subset}/{key}: weighted={sub_val:.4f} full={full_val:.4f} '
                        f'|gap|={gap:.4f} (tolerance 3 x {mae})'))


def _overall(result):
    import pandas as pd

    if isinstance(result, pd.DataFrame) and 'Overall' in result:
        return float(np.asarray(result['Overall'], dtype=float).mean())
    if isinstance(result, dict):
        for key in ('Overall', 'overall', 'score'):
            if key in result:
                return float(result[key])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--subset', nargs='*', default=None, help='default: every subset in the manifest')
    ap.add_argument('--record-pkl', default=None, help='mining correctness matrix, enables the mae check')
    ap.add_argument('--record', action='append', default=None, metavar='LAYOUT=PKL',
                    help='repeatable; correctness matrix with a named layout '
                         '(--record-pkl X == --record row=X)')
    ap.add_argument('--record-key', default=None)
    ap.add_argument('--pred-file', default=None, help='full-set prediction file, enables the replay check')
    ap.add_argument('--items-file', default=None,
                    help="a full-set run's per-item scores file, enables the items-replay check")
    ap.add_argument('--judge', default='exact_matching', help='judge for the replay check')
    args = ap.parse_args()

    records = []
    if args.record_pkl:
        records.append(('row', args.record_pkl))
    for spec_arg in args.record or []:
        if '=' not in spec_arg:
            raise SystemExit(f'--record wants LAYOUT=PKL, got {spec_arg!r}')
        records.append(tuple(spec_arg.split('=', 1)))

    subsets = args.subset or [s for s in available_subsets(args.dataset) if s != 'full']
    results = []
    for subset in subsets:
        full, _ = check_build(args.dataset, subset, results)
        if full is None:
            continue
        for layout_name, record_pkl in records:
            check_mae(args.dataset, subset, full, get_layout(layout_name),
                      record_pkl, args.record_key, results)
        if args.pred_file:
            check_replay(args.dataset, subset, args.pred_file, args.judge, results)
        if args.items_file:
            check_items_replay(args.dataset, subset, args.items_file, results)

    print(f'\n=== ZipBench validation: {args.dataset} ===')
    for ok, msg in results:
        print(f'{OK if ok else FAIL} {msg}')
    failed = [m for ok, m in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} checks passed')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
