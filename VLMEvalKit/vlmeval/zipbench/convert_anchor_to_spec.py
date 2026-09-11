#!/usr/bin/env python3
"""Convert a mined anchor ``.pkl`` into a ZipBench spec + manifest.

Anchor pkls are the raw product of the subset-mining pipeline: binary and
Python-only. This script materialises one into a portable, diffable spec.

Anchor pkl structure::

    {dataset_key: {"indices": ndarray[int],    # 0-based ROW POSITIONS
                   "weights": ndarray[float],  # normalised
                   "by_sub":  {...}}}

**The positions are row positions, not `index` column values.** Several
VLMEvalKit datasets have non-positional ids (``MMMU_DEV_VAL``: sparse ints
1..11550 over 1050 rows; ``SpatialEval`` / ``CountBenchQA``: strings), so this
script resolves position -> ``index`` value by building the dataset exactly the
way ``build_dataset`` does. The resulting spec is then robust to upstream
re-ordering.

Example::

    python -m vlmeval.zipbench.convert_anchor_to_spec \\
        --dataset MMStar --name small \\
        --pkl path/to/anchor_weighted_MMStar_mae_<M>_ratio_<R>.pkl \\
        --adapter vlmeval.zipbench.adapters.mcq.ZipMCQEvalMixin \\
        --metric-note 'micro_accuracy -- ...'

Anchors whose position space is not the TSV row order use a named layout, and
several anchors can be merged into one spec (OCRBench_v2 keeps en + cn in a
single spec so one inference run yields both numbers)::

    python -m vlmeval.zipbench.convert_anchor_to_spec \\
        --dataset OCRBench_v2 --name small --weight-norm per_group \\
        --anchor ocrbench_v2_en=path/to/anchor_weighted_OCRBench_v2_en_mae_<M>_ratio_<R>.pkl \\
        --anchor ocrbench_v2_cn=path/to/anchor_weighted_OCRBench_v2_cn_mae_<M>_ratio_<R>.pkl \\
        --headline-key 'English Overall Score' --headline-key 'Chinese Overall Score' ...

``--pkl X`` is shorthand for ``--anchor row=X``. With ``--weight-norm
per_group`` the weights are kept as mined (each task group sums to 1, the
total equals the number of groups) instead of requiring ``sum(w) == 1``.

Always follow up with ``python -m vlmeval.zipbench.validate --dataset MMStar``.
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import re

import numpy as np

FNAME_RE = re.compile(r'mae_(?P<mae>[0-9]+(?:\.[0-9]+)?)_ratio_(?P<ratio>[0-9]+(?:\.[0-9]+)?)')


def parse_filename(path):
    """Pull ``(mae, ratio)`` out of ``anchor_weighted_<DS>_mae_X_ratio_Y.pkl``."""
    m = FNAME_RE.search(osp.basename(path))
    if not m:
        return None, None
    return float(m.group('mae')), float(m.group('ratio'))


def load_anchor(pkl_path, dataset_key=None):
    with open(osp.expanduser(pkl_path), 'rb') as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f'{pkl_path}: not a non-empty dict')
    if dataset_key is None:
        if len(payload) != 1:
            raise ValueError(
                f'{pkl_path}: holds {list(payload)}; pass --dataset-key to disambiguate')
        dataset_key = next(iter(payload))
    entry = payload[dataset_key]
    positions = np.asarray(entry['indices']).astype(int).tolist()
    weights = np.asarray(entry['weights']).astype(float).tolist()
    if len(positions) != len(weights):
        raise ValueError(f'{pkl_path}: indices/weights length mismatch')

    # Anchors occasionally list the same position twice; summing the weights is
    # the equivalent estimator.
    merged = {}
    for pos, w in zip(positions, weights):
        merged[pos] = merged.get(pos, 0.0) + w
    if len(merged) != len(positions):
        print(f'  note: merged {len(positions) - len(merged)} duplicate position(s)')
    return dataset_key, sorted(merged.items())


def build_full_dataset(dataset):
    """The full dataset in exactly the order inference would see it."""
    from ..dataset import build_dataset

    ds = build_dataset(dataset)
    if ds is None:
        raise ValueError(f'build_dataset({dataset!r}) returned None')
    if getattr(ds, 'zip_subset', None):
        raise RuntimeError('build_dataset returned an already-subsetted dataset')
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--dataset', required=True, help='VLMEvalKit dataset name, e.g. MMStar')
    ap.add_argument('--name', required=True, help="subset version name, e.g. 'small' / 'tiny'")
    ap.add_argument('--pkl', default=None, help='path to the anchor .pkl (== --anchor row=PKL)')
    ap.add_argument('--anchor', action='append', default=None, metavar='LAYOUT=PKL',
                    help='repeatable; anchor pkl with a named position-space layout '
                         '(see layouts.py). All anchors merge into one spec.')
    ap.add_argument('--weight-norm', choices=['global', 'per_group'], default='global',
                    help="'global': sum(w) == 1 (default); 'per_group': every task group "
                         'sums to 1, total == number of groups')
    ap.add_argument('--selection-unit', choices=['row', 'pair'], default='row',
                    help="'pair': anchor positions are PAIR indices p over adjacent TSV row "
                         'pairs (rows 2p, 2p+1); each pair expands to its two rows, each '
                         'carrying weight w_p / 2 (MMVP-style paired benchmarks)')
    ap.add_argument('--dataset-key', default=None, help='key inside the pkl (default: the only one)')
    ap.add_argument('--adapter', default='vlmeval.zipbench.adapters.mcq.ZipMCQEvalMixin')
    ap.add_argument('--metric-note', default=None, help="the `metric` string written into results")
    ap.add_argument('--official-metric', default='micro_accuracy')
    ap.add_argument('--headline-key', action='append', default=None,
                    help='repeatable; names of the headline entries in the upstream result dict')
    ap.add_argument('--mae', type=float, default=None, help='override anchor_mae (else from filename)')
    ap.add_argument('--ratio', type=float, default=None, help='override compression ratio')
    ap.add_argument('--renormalize', action='store_true',
                    help='rescale weights to sum to 1 (anchors whose weights sum to n)')
    ap.add_argument('--no-qhash', action='store_true',
                    help='do not store question hashes (datasets whose question lives in the image)')
    ap.add_argument('--no-dataset', action='store_true',
                    help='no VLMEvalKit TSV backs this dataset: skip build_dataset; positions '
                         'resolve through the layout alone (id_list), dataset_size = the '
                         'layout position count, qhash always null')
    ap.add_argument('--gt-json', default=None,
                    help='[--no-dataset] ground-truth JSON whose md5 is recorded as '
                         "the manifest's drift signal (gt_json_md5, replaces tsv_md5)")
    ap.add_argument('--manifest-extra', default=None,
                    help='JSON object merged into the manifest top level '
                         '(e.g. id_space / score_kind / aggregation fields)')
    ap.add_argument('--out-dir', default=None, help='default: vlmeval/zipbench/subsets/<dataset>')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    try:
        from .layouts import get_layout
        from .spec import question_hash, subsets_dir
    except ImportError:
        # Direct-path execution (python vlmeval/zipbench/convert_anchor_to_spec.py):
        # load the sibling stdlib-only modules without importing the (slow)
        # vlmeval package. Only valid together with --no-dataset.
        import importlib.util

        def _sibling(name):
            path = osp.join(osp.dirname(osp.abspath(__file__)), f'{name}.py')
            spec = importlib.util.spec_from_file_location(f'_zipbench_{name}', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        get_layout = _sibling('layouts').get_layout
        _spec_mod = _sibling('spec')
        question_hash, subsets_dir = _spec_mod.question_hash, _spec_mod.subsets_dir
        if not args.no_dataset:
            raise SystemExit('direct-path execution needs --no-dataset '
                             '(build_dataset requires the vlmeval package)')

    if bool(args.pkl) == bool(args.anchor):
        raise SystemExit('pass exactly one of --pkl or --anchor (repeatable)')
    anchor_args = args.anchor or [f'row={args.pkl}']
    anchors = []
    for spec_arg in anchor_args:
        if '=' not in spec_arg:
            raise SystemExit(f'--anchor wants LAYOUT=PKL, got {spec_arg!r}')
        layout_name, pkl_path = spec_arg.split('=', 1)
        anchors.append((get_layout(layout_name), pkl_path))
    if args.mae is not None and len(anchors) > 1:
        raise SystemExit('--mae only makes sense with a single anchor')
    if args.selection_unit == 'pair':
        if len(anchors) > 1 or anchors[0][0].name != 'row':
            raise SystemExit('--selection-unit pair supports a single row-layout anchor only')
        if args.weight_norm != 'global':
            raise SystemExit('--selection-unit pair only supports --weight-norm global')

    if args.no_dataset:
        if args.selection_unit == 'pair':
            raise SystemExit('--no-dataset does not support --selection-unit pair')
        ds, data = None, None
        n_total = len(anchors[0][0].positions(None))
        for layout, _ in anchors[1:]:
            if len(layout.positions(None)) != n_total:
                raise SystemExit('--no-dataset: merged anchors must share one position count')
        print(f'dataset {args.dataset}: {n_total} positions (no TSV, --no-dataset)')
        questions = {}
    else:
        ds = build_full_dataset(args.dataset)
        data = ds.data
        n_total = len(data)
        print(f'dataset {args.dataset}: {n_total} rows, index dtype={data["index"].dtype}')
        if args.selection_unit == 'pair' and n_total % 2:
            raise SystemExit(f'--selection-unit pair needs an even row count, got {n_total}')

        questions = {i: q for i, q in zip(data['index'], data['question'])} \
            if 'question' in data else {}

    records, seen = [], {}
    weight_sum = 0.0
    anchor_mae, n_by_label = {}, {}
    for layout, pkl_path in anchors:
        key, pairs = load_anchor(pkl_path, args.dataset_key)
        positions = [p for p, _ in pairs]
        weights = [w for _, w in pairs]
        print(f'anchor key={key!r} layout={layout.name}  n={len(positions)}  '
              f'positions=[{min(positions)}, {max(positions)}]  weight_sum={sum(weights):.6f}')

        n_pairs_sel = None
        if args.selection_unit == 'pair':
            # Positions are pair indices p; pair p == TSV rows 2p and 2p+1.
            n_pairs_sel = len(positions)
            if max(positions) >= n_total // 2:
                raise ValueError(
                    f'pair position {max(positions)} out of range for {n_total // 2} pairs')
            positions = [row for p in positions for row in (2 * p, 2 * p + 1)]
            weights = [half for w in weights for half in (w / 2, w / 2)]
            print(f'  pair unit: {n_pairs_sel} pairs -> {len(positions)} rows '
                  '(w_p split evenly over the two rows)')

        pos_to_index = layout.positions(data)
        pos_groups = layout.groups(data)
        if max(positions) >= len(pos_to_index):
            raise ValueError(
                f'anchor position {max(positions)} is out of range for a '
                f'{len(pos_to_index)}-position layout {layout.name!r} -- the anchor was mined '
                'against a different version or a different row order')

        wsum = float(sum(weights))
        if args.weight_norm == 'global':
            if args.renormalize:
                weights = [w / wsum for w in weights]
                wsum = float(sum(weights))
            elif abs(wsum - 1.0) > 1e-6:
                raise ValueError(
                    f'weights sum to {wsum:.6f}, not 1; pass --renormalize if that is expected')
        else:
            if pos_groups is None:
                raise ValueError(
                    f'--weight-norm per_group needs a grouped layout, {layout.name!r} is flat')
            by_group = {}
            for pos, w in zip(positions, weights):
                by_group[pos_groups[pos]] = by_group.get(pos_groups[pos], 0.0) + w
            bad = {g: s for g, s in by_group.items() if abs(s - 1.0) > 1e-6}
            if bad:
                raise ValueError(f'{layout.name}: per-group weight sums != 1: {bad}')
            if abs(wsum - len(layout.GROUPS)) > 1e-6:
                raise ValueError(
                    f'{layout.name}: total weight {wsum:.6f} != {len(layout.GROUPS)} groups -- '
                    'the anchor does not cover every group')
        weight_sum += wsum

        label = layout.label or layout.name
        for pos, w in zip(positions, weights):
            idx = pos_to_index[pos]
            idx = idx.item() if hasattr(idx, 'item') else idx
            if idx in seen:
                raise ValueError(
                    f'index {idx!r} selected by both {seen[idx]} and {pkl_path} -- '
                    'merged anchors must not overlap')
            seen[idx] = pkl_path
            records.append({
                'index': idx,
                'weight': w,
                'qhash': None if args.no_qhash else question_hash(questions.get(idx)),
            })

        mae, ratio = parse_filename(pkl_path)
        mae = args.mae if args.mae is not None else mae
        ratio = args.ratio if args.ratio is not None else ratio
        if mae is None or ratio is None:
            raise ValueError(
                f'{pkl_path}: mae/ratio not in the filename; pass --mae and --ratio explicitly')
        if args.selection_unit == 'pair':
            anchor_ratio = 1 - n_pairs_sel / (n_total // 2)
        else:
            anchor_ratio = 1 - len(positions) / len(pos_to_index)
        if abs(anchor_ratio - ratio) > 5e-3:
            print(f'  WARNING: filename ratio {ratio} != 1 - n/N = {anchor_ratio:.4f}')
        anchor_mae[label] = mae
        n_by_label[label] = len(positions)

    single = len(anchors) == 1 and anchors[0][0].label is None
    computed_ratio = 1 - len(records) / n_total

    out_dir = args.out_dir or subsets_dir(args.dataset)
    spec_file = f'{args.name}.jsonl'
    manifest_path = osp.join(out_dir, 'manifest.json')

    manifest = {}
    if osp.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    manifest.update({
        'dataset': args.dataset,
        'dataset_size': n_total,
        'selection_unit': args.selection_unit,
        'adapter': args.adapter,
        'official_metric': args.official_metric,
        'metric_note': args.metric_note or manifest.get('metric_note'),
        'weight_normalization': args.weight_norm,
        'weighting': (
            'score = sum(w_i * s_i) / sum(w_i); weights normalised so sum(w_i) == 1'
            if args.weight_norm == 'global' else
            'per group: sum(w_i * s_i) / sum(w_i) within each task group, macro-averaged; '
            'weights sum to 1 within each group'),
    })
    if args.no_dataset:
        manifest['tsv_md5'] = None
        if args.gt_json:
            with open(osp.expanduser(args.gt_json), 'rb') as f:
                manifest['gt_json_md5'] = hashlib.md5(f.read()).hexdigest()
    else:
        manifest['tsv_md5'] = getattr(type(ds), 'DATASET_MD5', {}).get(args.dataset)
    if args.manifest_extra:
        extra = json.loads(args.manifest_extra)
        if not isinstance(extra, dict):
            raise SystemExit('--manifest-extra must be a JSON object')
        manifest.update(extra)
    if args.selection_unit == 'pair':
        manifest['pair_space'] = {
            'n_pairs': n_total // 2,
            'rule': 'rows 2p, 2p+1 = pair p (0-based row positions; anchor indices are pair indices)',
        }
    if args.headline_key:
        manifest['headline_keys'] = args.headline_key
    subset_entry = {
        'file': spec_file,
        'n': len(records),
        'ratio': round(computed_ratio, 4),
        'anchor_mae': next(iter(anchor_mae.values())) if single else anchor_mae,
        'weight_sum': weight_sum,
    }
    if args.selection_unit == 'pair':
        subset_entry['n_pairs'] = n_pairs_sel
    if not single:
        subset_entry['n_by_lang'] = n_by_label
    manifest.setdefault('subsets', {})[args.name] = subset_entry

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f'[dry-run] would write {len(records)} records to {osp.join(out_dir, spec_file)}')
        return

    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, spec_file), 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'wrote {len(records)} records -> {osp.join(out_dir, spec_file)}')
    print(f'wrote manifest -> {manifest_path}')
    if not args.no_dataset:
        print(f'now run: python -m vlmeval.zipbench.validate --dataset {args.dataset}')


if __name__ == '__main__':
    main()
