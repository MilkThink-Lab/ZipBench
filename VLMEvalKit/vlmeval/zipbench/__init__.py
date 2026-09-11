"""ZipBench for VLMEvalKit -- weighted compressed subsets of a benchmark.

A subset reproduces the full-set score (and model ranking) from a fraction of
the questions: each kept row carries a ``weight``, and the score is a weighted
mean instead of a plain one.

Usage::

    python run.py --data MMStar --model <model> --subset tiny

Keep this module stdlib-only at import time -- ``vlmeval.inference``
imports ``naming`` at module level.
"""
from .naming import TAG_PREFIX, zip_file_key, zip_tag
from .spec import (FULL, available_subsets, check_qhash, has_zip_subsets,
                   list_zip_datasets, load_manifest, load_subset_spec,
                   question_hash, select_rows, subsets_dir)

__all__ = [
    'FULL', 'TAG_PREFIX', 'apply_zip_subset', 'available_subsets', 'check_qhash',
    'has_zip_subsets', 'list_zip_datasets', 'load_manifest', 'load_subset_spec',
    'question_hash', 'select_rows', 'subsets_dir', 'zip_dataset_names',
    'zip_file_key', 'zip_tag',
]

#: Column added to ``dataset.data`` holding each kept row's weight.
WEIGHT_COL = 'zip_weight'


def apply_zip_subset(ds, subset, strict_qhash=False):
    """Turn a freshly built dataset into its weighted ZipBench subset, in place.

    Two things happen:

    1. ``ds.data`` is filtered to the spec rows and gains a ``zip_weight``
       column. Inference then only runs on those rows, and the weight column
       rides along into the prediction file for free.
    2. ``ds.__class__`` gets the manifest's adapter mixin spliced in front, so
       ``ds.evaluate()`` runs the upstream scoring untouched and re-aggregates
       it with the weights.

    ``ds.dataset_name`` is deliberately left alone; only file names carry the
    subset tag (see ``naming.py``).
    """
    if subset in (None, FULL):
        return ds

    manifest = load_manifest(ds.dataset_name)
    spec = load_subset_spec(ds.dataset_name, subset)

    _check_tsv_md5(ds, manifest)

    mismatches = check_qhash(ds.data, spec, dataset=ds.dataset_name)
    if mismatches:
        msg = (f'{ds.dataset_name}: {len(mismatches)} question(s) no longer match the subset spec '
               f'(first: {mismatches[:3]}). The upstream data has drifted; rebuild the spec.')
        if strict_qhash:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)

    ds.data = select_rows(ds.data, spec, weight_col=WEIGHT_COL, dataset=ds.dataset_name)
    ds.zip_subset = subset
    ds.zip_manifest = manifest

    from .adapters import resolve_adapter
    mixin = resolve_adapter(manifest['adapter'])
    ds.__class__ = type(f'Zip{type(ds).__name__}', (mixin, type(ds)), {})
    return ds


def _check_tsv_md5(ds, manifest):
    """Warn when the local TSV differs from the one the spec was built on."""
    expected = manifest.get('tsv_md5')
    if not expected:
        return
    actual = getattr(type(ds), 'DATASET_MD5', {}).get(ds.dataset_name)
    if actual and actual != expected:
        import warnings
        warnings.warn(
            f'{ds.dataset_name}: DATASET_MD5 is {actual} but the ZipBench spec was built against '
            f'{expected}. The subset indices may no longer point at the same questions.')


def zip_dataset_names():
    """Every ``<dataset>_ZIP_<subset>`` name, for ``SUPPORTED_DATASETS``.

    Registering these makes ``fetch_aux_files``' prefix-exclusion rule
    (``smp/file.py:449``) treat ``MMStar_ZIP_tiny*`` as belonging to a
    *different* dataset than ``MMStar``, so ``--reuse`` never mixes them up.
    """
    names = []
    for dataset in list_zip_datasets():
        try:
            subsets = load_manifest(dataset)['subsets']
        except Exception:
            continue
        names.extend(f'{dataset}{zip_tag(s)}' for s in subsets)
    return names
