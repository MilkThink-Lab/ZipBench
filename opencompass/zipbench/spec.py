"""ZipBench subset *specs*: open definitions of weighted benchmark subsets.

A spec is a JSONL file (one record per line)::

    {"id": <stable id>, "index": <0-based position>, "weight": <float>}

paired with a per-dataset ``manifest.yaml`` that lists every subset version and
records how to rebuild the *full* dataset (``base_loader``, ``id_field``,
``breakdown_fields``). See ``convert_anchor_to_spec.py`` for how specs are
produced from the internal anchor ``.pkl`` files.

This module is deliberately lightweight at import time (standard library only);
``datasets`` / ``yaml`` are imported lazily inside the functions that need them
so configs can ``import zipbench`` without paying that cost.
"""
import json
import os.path as osp
from typing import Dict, List


def load_subset_spec(spec_path: str) -> List[Dict]:
    """Read a JSONL subset spec into a list of ``{id, index, weight}`` dicts."""
    spec_path = osp.expanduser(spec_path)
    records: List[Dict] = []
    with open(spec_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f'{spec_path}:{line_no}: invalid JSON ({e})') from e
            if 'index' not in rec or 'weight' not in rec:
                raise ValueError(
                    f'{spec_path}:{line_no}: record must contain '
                    "'index' and 'weight'")
            records.append({
                'id': rec.get('id'),
                'index': int(rec['index']),
                'weight': float(rec['weight']),
            })
    if not records:
        raise ValueError(f'subset spec {spec_path} is empty')
    return records


def load_manifest(manifest_path: str) -> Dict:
    """Load a subset manifest (``manifest.yaml`` or ``manifest.json``).

    A ``.json`` sibling is preferred when present (it can be read with the
    standard library only, which keeps lazy-import OpenCompass configs working);
    otherwise the YAML is parsed with PyYAML.
    """
    manifest_path = osp.expanduser(manifest_path)
    json_sibling = osp.splitext(manifest_path)[0] + '.json'
    if manifest_path.endswith('.json') or (
            not osp.exists(manifest_path) and osp.exists(json_sibling)):
        path = manifest_path if manifest_path.endswith('.json') \
            else json_sibling
        if not osp.exists(path):
            raise FileNotFoundError(f'subset manifest not found: {path}')
        with open(path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    else:
        import yaml  # local import; PyYAML ships with opencompass / mmengine

        if not osp.exists(manifest_path):
            raise FileNotFoundError(
                f'subset manifest not found: {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f) or {}
    if not isinstance(manifest, dict) or 'subsets' not in manifest:
        raise ValueError(f"{manifest_path}: missing 'subsets' section")
    return manifest


def validate_and_select(dataset, spec: List[Dict], id_field: str = '_id'):
    """Select the spec rows from ``dataset['test']`` and attach weights.

    The full dataset is produced by the base loader; ``dataset['train']``
    retains the original columns (including the stable id) while
    ``dataset['test']`` is the reconstructed split used for inference. Both are
    in the same natural order, so ``index`` selects the same row in each.

    Raises ``ValueError`` on duplicate / out-of-range indices, or if the stable
    ``id`` no longer matches the row at ``index`` (i.e. the upstream data drifted
    since the spec was created).
    """
    from datasets import DatasetDict  # local import keeps module load cheap

    test_split = dataset['test']
    n_total = len(test_split)

    # Stable ids (if available) for alignment validation, taken from the raw
    # 'train' split which preserves the original columns.
    raw_ids = None
    if 'train' in dataset and id_field in dataset['train'].column_names:
        raw_ids = dataset['train'][id_field]

    indices = [r['index'] for r in spec]
    weights = [r['weight'] for r in spec]

    if len(set(indices)) != len(indices):
        raise ValueError('subset spec contains duplicate indices')
    lo, hi = min(indices), max(indices)
    if lo < 0 or hi >= n_total:
        raise ValueError(
            f'subset index out of range: [{lo}, {hi}] not within '
            f'[0, {n_total}) -- dataset may have changed since the spec was '
            'created')

    # Stable-id alignment validation: catch silent drift if the upstream data
    # was re-ordered / re-hosted.
    if raw_ids is not None:
        mismatches = []
        for r in spec:
            spec_id = r['id']
            if spec_id is None:
                continue
            actual = str(raw_ids[r['index']])
            if actual != str(spec_id):
                mismatches.append((r['index'], spec_id, actual))
        if mismatches:
            preview = mismatches[:5]
            raise ValueError(
                f'subset id mismatch on {len(mismatches)} row(s); the dataset '
                f'no longer matches the spec. First mismatches '
                f'(index, expected_id, actual_id): {preview}')

    subset_test = test_split.select(indices)
    subset_test = subset_test.add_column('weight', weights)

    out = DatasetDict()
    # Preserve the base loader's original 'train' split (unchanged, full size)
    # so any few-shot retriever still works; only 'test' becomes the weighted
    # subset. OpenCompass's DatasetReader requires both splits to exist.
    if 'train' in dataset:
        out['train'] = dataset['train']
    out['test'] = subset_test
    return out
