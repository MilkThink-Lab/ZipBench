"""Programmatic helper to switch an OpenCompass ``datasets`` list onto a
ZipBench weighted subset.

The custom dataset / evaluator are referenced by *fully-qualified string*
``type`` (not class objects) on purpose: ``run.py`` dumps each task config to a
plain ``.py`` file and re-loads it in a fresh subprocess that does **not** run
this config's imports. A string ``type`` is dumped verbatim and resolved in the
subprocess via ``import_module`` -- and the repo root is importable there
through the editable install (``pip install -e .``), so
``zipbench.dataset`` / ``zipbench.evaluator`` (and any benchmark adapter under
``zipbench.adapters``) are reachable.

The manifest fully describes *which* zipbench loader / evaluator a benchmark
uses, so this helper -- and the equivalent inline block in the configs -- is
benchmark-independent (see ``resolve_wiring`` for the schema). Row-level
exact-match benchmarks (LongBench v2) use the defaults; richer benchmarks
(SciCode: sub-step selection + code-execution scoring) point the manifest at a
``zipbench.adapters.*`` dataset/evaluator pair instead.

NOTE: this helper is for *non-lazy* contexts (scripts, tests, configs without
``read_base()``). A config that uses ``with read_base(): ...`` is parsed by
mmengine in lazy-import mode, where ``apply_zip_subset`` would be an uncallable
``LazyObject``; such configs must inline the equivalent dict-mutation with
string ``type``s instead (see ``configs/eval_longbenchv2.py`` /
``configs/eval_scicode.py`` -- the inline block mirrors ``apply_one`` below).
"""
import os.path as osp

from .spec import load_manifest

# Fully-qualified, importable references (defaults for row-level benchmarks).
ZIP_DATASET_TYPE = 'zipbench.dataset.ZipSubsetDataset'
WEIGHTED_EVALUATOR_TYPE = 'zipbench.evaluator.WeightedAccuracyEvaluator'

_SUBSETS_ROOT = osp.join(osp.dirname(osp.abspath(__file__)), 'subsets')


def resolve_wiring(manifest, subset):
    """Resolve the benchmark-independent wiring for one subset from a manifest.

    Returns a dict with everything a config needs to rewire a ``datasets`` list,
    decoupling the (benchmark-specific) manifest schema from the (generic)
    rewiring loop so the same loop serves every benchmark::

        {
          'spec_path':       <abs path to the <name>.jsonl spec>,
          'dataset_type':    <fully-qualified zipbench loader type string>,
          'loader_kwargs':   <extra ctor kwargs merged onto each ds dict>,
          'evaluator_type':  <fully-qualified evaluator type string>,
          'merge_base_kwargs': <bool: merge onto the original evaluator dict>,
          'inject_subset_spec': <bool: pass subset_spec= to the evaluator too>,
          'breakdown_fields': <list[str]>,
        }

    Manifest schema (all sections optional; defaults reproduce the original
    row-level LongBench behaviour, so pre-existing manifests keep working)::

        dataset:
          type: zipbench.dataset.ZipSubsetDataset
          loader_kwargs: {base_loader: ..., id_field: _id}
        evaluator:
          type: zipbench.evaluator.WeightedAccuracyEvaluator
          merge_base_kwargs: false
          inject_subset_spec: false
        # legacy top-level fallbacks (still honoured):
        base_loader: ...
        id_field: _id
        breakdown_fields: [...]
    """
    available = manifest.get('subsets', {})
    if subset not in available:
        raise ValueError(
            f'unknown subset {subset!r}; available: {["full", *available]}')

    ds_section = manifest.get('dataset') or {}
    loader_kwargs = dict(ds_section.get('loader_kwargs') or {})
    # Legacy fallback: top-level base_loader / id_field (pre-``dataset`` schema).
    if 'base_loader' in manifest and 'base_loader' not in loader_kwargs:
        loader_kwargs['base_loader'] = manifest['base_loader']
    if 'id_field' in manifest and 'id_field' not in loader_kwargs:
        loader_kwargs.setdefault('id_field', manifest.get('id_field', '_id'))

    ev_section = manifest.get('evaluator') or {}

    return {
        'spec_file': available[subset]['file'],
        'dataset_type': ds_section.get('type', ZIP_DATASET_TYPE),
        'loader_kwargs': loader_kwargs,
        'evaluator_type': ev_section.get('type', WEIGHTED_EVALUATOR_TYPE),
        'merge_base_kwargs': bool(ev_section.get('merge_base_kwargs', False)),
        'inject_subset_spec': bool(ev_section.get('inject_subset_spec', False)),
        'breakdown_fields': list(manifest.get('breakdown_fields') or []),
    }


def apply_one(ds, subset, wiring, spec_path):
    """Rewire a single dataset cfg dict in place onto a ZipBench subset.

    This is the canonical rewiring loop; the inline block in each config is a
    verbatim (lazy-import-safe) copy of it. ``wiring`` is the dict returned by
    :func:`resolve_wiring`; ``spec_path`` is the absolute path to its spec file.
    """
    ds['type'] = wiring['dataset_type']
    # Merge extra loader ctor kwargs (e.g. base_loader / id_field for the
    # generic loader); adapters that reuse the base ds kwargs need none.
    for key, value in wiring['loader_kwargs'].items():
        ds[key] = value
    ds['subset_spec'] = spec_path
    if 'abbr' in ds:
        ds['abbr'] = f"{ds['abbr']}_{subset}"

    ds.setdefault('eval_cfg', {})
    base_ev = (ds['eval_cfg'].get('evaluator', {})
               if wiring['merge_base_kwargs'] else {})
    new_ev = dict(base_ev)
    new_ev['type'] = wiring['evaluator_type']
    new_ev['breakdown_fields'] = wiring['breakdown_fields']
    if wiring['inject_subset_spec']:
        new_ev['subset_spec'] = spec_path
    ds['eval_cfg']['evaluator'] = new_ev
    return ds


def apply_zip_subset(datasets, dataset_key, subset, subsets_root=None):
    """Mutate ``datasets`` in place to evaluate on a ZipBench subset.

    Args:
        datasets: an OpenCompass ``datasets`` list (list of dataset cfg dicts).
        dataset_key: subset namespace, i.e. the directory under ``subsets/``
            holding ``manifest.yaml`` (e.g. ``'LongBenchv2_0shot'`` /
            ``'SciCode_with_background'``).
        subset: ``'full'`` (leave untouched, full unweighted eval) or a subset
            version name defined in the manifest (e.g. ``'tiny'`` / ``'small'``).
        subsets_root: override the directory that contains ``<dataset_key>/``
            (defaults to ``zipbench/subsets``).

    Returns:
        The same ``datasets`` list (mutated), for convenient chaining.
    """
    if subset == 'full':
        return datasets

    root = subsets_root or _SUBSETS_ROOT
    manifest_dir = osp.join(root, dataset_key)
    manifest = load_manifest(osp.join(manifest_dir, 'manifest.yaml'))
    wiring = resolve_wiring(manifest, subset)
    spec_path = osp.join(manifest_dir, wiring['spec_file'])

    for ds in datasets:
        apply_one(ds, subset, wiring, spec_path)
    return datasets
