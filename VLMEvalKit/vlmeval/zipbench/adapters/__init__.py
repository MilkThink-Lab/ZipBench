"""Per-benchmark ZipBench adapters.

Selecting rows is generic; **aggregating them is not**. Each adapter answers
one question -- where did the upstream ``evaluate()`` put its per-item scores,
and how does the benchmark officially aggregate them.

A manifest points at its adapter by fully-qualified dotted path::

    "adapter": "vlmeval.zipbench.adapters.mcq.ZipMCQEvalMixin"
"""
import importlib

from .base import ZipEvalMixin
from .countbenchqa import ZipCountBenchQAEvalMixin
from .logicvista import ZipLogicVistaEvalMixin
from .mathvista import ZipMathVistaEvalMixin
from .mcq import ZipMCQEvalMixin
from .mmvp import ZipMMVPEvalMixin
from .ocrbench_v2 import ZipOCRBenchV2EvalMixin
from .simplevqa import ZipSimpleVQAEvalMixin

__all__ = ['ZipEvalMixin', 'ZipMCQEvalMixin', 'ZipMathVistaEvalMixin',
           'ZipLogicVistaEvalMixin', 'ZipCountBenchQAEvalMixin',
           'ZipMMVPEvalMixin', 'ZipOCRBenchV2EvalMixin', 'ZipSimpleVQAEvalMixin',
           'resolve_adapter']


def resolve_adapter(dotted):
    """Import and return the mixin class named by a dotted path."""
    if not isinstance(dotted, str) or '.' not in dotted:
        raise ValueError(
            "manifest 'adapter' must be a fully-qualified dotted path "
            f"(e.g. 'vlmeval.zipbench.adapters.mcq.ZipMCQEvalMixin'), got {dotted!r}")
    module_path, attr = dotted.rsplit('.', 1)
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f'cannot import {attr!r} from {module_path!r}') from e
    if not issubclass(cls, ZipEvalMixin):
        raise TypeError(f'{dotted} is not a ZipEvalMixin subclass')
    return cls
