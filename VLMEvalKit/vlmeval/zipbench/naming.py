"""File-naming helpers for ZipBench subsets.

A ZipBench run must NOT reuse the full set's prediction file name: full and
subset runs would overwrite each other and ``--reuse`` would silently feed the
full-set predictions into a subset evaluation.

The tag is applied to *file names only*. ``dataset.dataset_name`` stays the
original name on purpose -- it is also fed to ``model.use_custom_prompt`` /
``model.build_prompt(..., dataset=...)`` / ``DATASET_TYPE`` (an exact-match
lookup) / ``img_root_map``, none of which would survive a rename. Keeping it
also means the image cache is shared with the full run.

This module is deliberately stdlib-only so ``vlmeval.inference`` can import it
at module level.
"""

TAG_PREFIX = '_ZIP_'


def zip_tag(subset):
    """``'tiny'`` -> ``'_ZIP_tiny'``; ``None`` / ``'full'`` -> ``''``."""
    if subset is None or subset == 'full':
        return ''
    return f'{TAG_PREFIX}{subset}'


def zip_file_key(dataset, dataset_name=None):
    """The name to use when building prediction / intermediate file paths.

    Args:
        dataset: a built dataset object (may carry ``zip_subset``), or ``None``.
        dataset_name: override the base name (defaults to
            ``dataset.dataset_name``).
    """
    name = dataset_name
    if name is None:
        name = getattr(dataset, 'dataset_name', None)
    return f'{name}{zip_tag(getattr(dataset, "zip_subset", None))}'
