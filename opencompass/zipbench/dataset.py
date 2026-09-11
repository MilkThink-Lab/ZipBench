"""Generic, base-loader-driven ZipBench subset dataset for OpenCompass.

:class:`ZipSubsetDataset` wraps *any* existing ``BaseDataset`` loader: it loads
the full dataset with that loader and -- when a ``subset_spec`` is given --
keeps only the spec rows and attaches a per-row ``weight`` column. No
per-benchmark subclass is required; the underlying loader is chosen by a
fully-qualified ``base_loader`` string in the dataset config, e.g.::

    dict(type='zipbench.dataset.ZipSubsetDataset',
         base_loader='opencompass.datasets.LongBenchv2Dataset',
         path='./data/longbenchv2/data.json',
         subset_spec='.../tiny.jsonl',
         id_field='_id')
"""
import importlib

# Import an ``opencompass.datasets`` submodule before anything that pulls in
# ``opencompass.openicl`` so the datasets package finishes initialising first.
# This is the import order OpenCompass itself uses and avoids a circular import
# (openicl -> inferencer -> opencompass.datasets -> ... -> openicl).
from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET

from .spec import load_subset_spec, validate_and_select


def _resolve(dotted: str):
    """Import and return the object named by a fully-qualified dotted path."""
    if not isinstance(dotted, str) or '.' not in dotted:
        raise ValueError(
            'base_loader must be a fully-qualified dotted path '
            f"(e.g. 'opencompass.datasets.LongBenchv2Dataset'), got {dotted!r}")
    module_path, attr = dotted.rsplit('.', 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(
            f'cannot import {attr!r} from {module_path!r}') from e


@LOAD_DATASET.register_module()
class ZipSubsetDataset(BaseDataset):
    """Load a full dataset via ``base_loader`` then keep a weighted subset.

    When ``subset_spec`` is ``None`` the result is identical to the base loader
    (full, unweighted evaluation).
    """

    @staticmethod
    def load(base_loader, path=None, subset_spec=None, id_field='_id',
             test_split='test', **base_kwargs):
        loader = _resolve(base_loader)
        load_fn = getattr(loader, 'load', None)
        if not callable(load_fn):
            raise TypeError(
                f'base_loader {base_loader!r} has no callable load() method')
        if isinstance(loader, type):
            import inspect
            raw = inspect.getattr_static(loader, 'load')
            if not isinstance(raw, (staticmethod, classmethod)):
                # Plain instance method (e.g. ArenaHardDataset.load(self, ...)).
                # BaseDataset loaders never use ``self`` in load(), so bind a
                # dummy instead of running BaseDataset.__init__ (which would
                # recurse into load()).
                load_fn = raw.__get__(object.__new__(loader), loader)
        if path is not None:
            base_kwargs['path'] = path
        dataset = load_fn(**base_kwargs)
        if subset_spec is None:
            return dataset
        # Some loaders return a flat ``Dataset`` (one split) rather than a
        # ``DatasetDict``; normalise so ``validate_and_select`` can index
        # ``['test']``. With no separate 'train' split the stable-id check is
        # skipped (index-only selection), so prefer loaders that expose the id.
        from datasets import Dataset, DatasetDict
        if isinstance(dataset, Dataset):
            # Flat loaders (e.g. C3Dataset_V2) must stay flat on return:
            # DatasetReader reuses the single Dataset for every split, whereas
            # a test-only DatasetDict would KeyError on its 'train' lookup.
            selected = validate_and_select(
                DatasetDict({'test': dataset}), load_subset_spec(subset_spec),
                id_field=id_field)
            return selected['test']
        # Some loaders name their inference split differently (e.g.
        # commonsenseqa's 'validation'); ``test_split`` says which split the
        # spec indices select from. Map it onto 'test' for
        # ``validate_and_select`` and restore the original name afterwards so
        # the dataset's reader_cfg keeps working unchanged.
        if test_split != 'test':
            work = DatasetDict({'test': dataset[test_split]})
            selected = validate_and_select(
                work, load_subset_spec(subset_spec), id_field=id_field)
            out = DatasetDict({k: v for k, v in dataset.items()
                               if k != test_split})
            out[test_split] = selected['test']
            return out
        spec = load_subset_spec(subset_spec)
        return validate_and_select(dataset, spec, id_field=id_field)
