"""ZipBench subset specs: open definitions of weighted benchmark subsets.

A spec is a JSONL file, one record per line::

    {"index": 295, "weight": 0.00066667, "qhash": "8f2c1a0b3d4e5f60"}

* ``index``  -- the value of the dataset TSV's ``index`` column (NOT a row
  position). VLMEvalKit keys predictions by this value throughout
  (``vlmeval/inference.py``), so selecting by value survives any upstream
  re-ordering of the TSV.
* ``weight`` -- normalised so ``sum(weight) == 1`` over the subset.
* ``qhash``  -- optional drift canary: ``sha1(normalised question)[:16]``.
  ``null`` for datasets whose question lives in the image (``MMMU_Pro_V``).

paired with a per-dataset ``manifest.json`` listing every subset version. See
``convert_anchor_to_spec.py`` for how specs are produced.

Deliberately stdlib-only at import time; ``pandas`` is imported inside the one
function that needs it.
"""
import hashlib
import json
import os
import os.path as osp

SUBSETS_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), 'subsets')

#: Sentinel meaning "no subsetting at all".
FULL = 'full'


def subsets_dir(dataset):
    return osp.join(SUBSETS_ROOT, dataset)


def has_zip_subsets(dataset):
    return osp.exists(osp.join(subsets_dir(dataset), 'manifest.json'))


def list_zip_datasets():
    """Every dataset name that ships a manifest under ``subsets/``."""
    if not osp.isdir(SUBSETS_ROOT):
        return []
    return sorted(d for d in os.listdir(SUBSETS_ROOT) if has_zip_subsets(d))


def load_manifest(dataset):
    path = osp.join(subsets_dir(dataset), 'manifest.json')
    if not osp.exists(path):
        raise FileNotFoundError(
            f'dataset {dataset!r} has no ZipBench manifest ({path}). '
            f'Datasets with subsets: {list_zip_datasets()}')
    with open(path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    if 'subsets' not in manifest:
        raise ValueError(f"{path}: missing 'subsets' section")
    return manifest


def available_subsets(dataset):
    return [FULL, *load_manifest(dataset)['subsets']]


def load_subset_spec(dataset, subset):
    """Read ``<dataset>/<subset>.jsonl`` into a list of dicts."""
    manifest = load_manifest(dataset)
    if subset not in manifest['subsets']:
        raise ValueError(
            f'unknown subset {subset!r} for {dataset!r}; '
            f'available: {available_subsets(dataset)}')
    path = osp.join(subsets_dir(dataset), manifest['subsets'][subset]['file'])
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{line_no}: invalid JSON ({e})') from e
            if 'index' not in rec or 'weight' not in rec:
                raise ValueError(f"{path}:{line_no}: record needs 'index' and 'weight'")
            records.append({
                'index': rec['index'],
                'weight': float(rec['weight']),
                'qhash': rec.get('qhash'),
            })
    if not records:
        raise ValueError(f'subset spec {path} is empty')
    return records


def question_hash(question):
    """Stable canary over a question string (whitespace-normalised)."""
    if question is None:
        return None
    text = ' '.join(str(question).split())
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


def select_rows(data, spec, weight_col='zip_weight', dataset=None):
    """Keep the spec rows of ``data`` (a DataFrame) and attach ``weight_col``.

    Selection is by the *value* of the ``index`` column. Raises if any spec
    row is missing from ``data`` -- a missing row means the spec is stale and
    silently scoring a smaller set would corrupt the weighted estimate.
    """
    import pandas as pd  # local import keeps module load cheap

    if 'index' not in data:
        raise ValueError(f'{dataset}: dataset has no `index` column')

    weights = {}
    for rec in spec:
        idx = rec['index']
        if idx in weights:
            raise ValueError(f'{dataset}: duplicate index {idx!r} in subset spec')
        weights[idx] = rec['weight']

    # The TSV's index dtype can be int or str depending on the dataset
    # (ImageBaseDataset casts back to int when every value looks like one).
    # Match the spec against whatever the built dataset uses.
    if len(data) and not isinstance(data['index'].iloc[0], str):
        try:
            weights = {type(data['index'].iloc[0])(k): v for k, v in weights.items()}
        except (TypeError, ValueError):
            pass

    selected = data[data['index'].isin(weights)].copy()
    if len(selected) != len(spec):
        missing = [k for k in weights if k not in set(selected['index'])]
        raise ValueError(
            f'{dataset}: {len(spec) - len(selected)} of {len(spec)} spec rows are not in the '
            f'dataset -- the spec is stale or the TSV changed. First missing indices: {missing[:5]}')

    selected[weight_col] = [weights[i] for i in selected['index']]
    return selected.reset_index(drop=True)


def check_qhash(data, spec, dataset=None, question_col='question'):
    """Compare each spec row's ``qhash`` against the live question text.

    Returns a list of ``(index, expected, actual)`` mismatches; empty when the
    spec has no hashes or the dataset has no question column.
    """
    if question_col not in data:
        return []
    live = {i: q for i, q in zip(data['index'], data[question_col])}
    mismatches = []
    for rec in spec:
        expected = rec.get('qhash')
        if expected is None:
            continue
        idx = rec['index']
        if idx not in live:
            continue
        actual = question_hash(live[idx])
        if actual != expected:
            mismatches.append((idx, expected, actual))
    return mismatches
