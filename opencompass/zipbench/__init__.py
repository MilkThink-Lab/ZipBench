"""ZipBench: evaluate OpenCompass models on compressed, *weighted* benchmark
subsets (as well as the full set), reusably across benchmarks.

A subset is described by an open, language-neutral JSONL *spec* (one record per
line ``{"id", "index", "weight"}``) plus a per-dataset ``manifest.yaml``. At
eval time :func:`apply_zip_subset` rewires an existing OpenCompass ``datasets``
list onto the generic :class:`~zipbench.dataset.ZipSubsetDataset` loader and the
:class:`~zipbench.evaluator.WeightedAccuracyEvaluator`, both referenced by
fully-qualified *string* ``type`` so they survive being dumped to the
inference / evaluation subprocesses spawned by ``run.py``.

Importing this package is cheap on purpose: the heavy ``zipbench.dataset`` /
``zipbench.evaluator`` modules (which pull in ``datasets`` and ``opencompass``)
are imported lazily and resolved by their ``type`` strings inside the worker
subprocesses. A config only needs :func:`apply_zip_subset`.
"""
from .apply import (WEIGHTED_EVALUATOR_TYPE, ZIP_DATASET_TYPE, apply_one,
                    apply_zip_subset, resolve_wiring)
from .spec import load_manifest, load_subset_spec

__all__ = [
    'apply_zip_subset',
    'apply_one',
    'resolve_wiring',
    'load_manifest',
    'load_subset_spec',
    'ZIP_DATASET_TYPE',
    'WEIGHTED_EVALUATOR_TYPE',
]
