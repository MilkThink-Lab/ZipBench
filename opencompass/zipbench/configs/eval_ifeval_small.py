"""ZipBench evaluation for IFEval (full / small / tiny subsets).

This is a regular OpenCompass config and MUST be launched through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_ifeval.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- the complete official set (541 prompts, unweighted)
  * ``'small'`` -- compressed weighted subset (260 prompts, ratio 0.5194)
  * ``'tiny'``  -- compressed weighted subset (136 prompts, ratio 0.7486)

The dataset / prompt wiring is imported from ``IFEval_gen_353ae7`` (zero-shot,
the 541 prompts fed verbatim, scored by OpenCompass's verbatim copy of Google's
``instruction_following_eval``), so 'full' here reproduces
``examples/eval_ifeval.py``. ``353ae7`` is used rather than ``3321a3`` because
the latter pins ``max_out_len=1025`` on the inferencer while the
``length_constraints:number_words`` instructions ask for up to 1200 words
(~1600+ tokens). On 'small'/'tiny' the dataset is rewired onto
``ZipSubsetDataset`` and the evaluator onto
``zipbench.adapters.ifeval.WeightedIFEvalEvaluator``.

Implementation note (why this file mutates dicts inline instead of calling a
helper): ``with read_base()`` puts mmengine into *lazy-import* mode, where every
non-builtin import becomes an uncallable ``LazyObject`` (usable only as a
``type=`` reference). So we cannot call ``zipbench.apply_zip_subset`` here; we
read the manifest with the standard library (``json``) and set fully-qualified
*string* ``type``s for the custom dataset / evaluator, which dump cleanly into
the run.py worker subprocesses.
"""
import json
import os.path as osp

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import \
        ifeval_datasets
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import models

SUBSET = 'small'
# The zipbench/subsets/ directory name (CamelCase, matching its siblings); the
# manifest's own ``dataset_key`` is the lowercase 'ifeval'.
DATASET_KEY = 'IFEval'

datasets = ifeval_datasets

# ---- strip the chain of thought before scoring ------------------------------
# IFEval checks the *surface form* of the answer: comma count, all-lowercase,
# word count, wrapping quotes, JSON shape, section markers. A chain of thought
# left in the prediction is handed to the checkers verbatim and fails almost all
# of them, so <think>...</think> is removed first. Attached at EVALUATOR level
# rather than eval_cfg level so predictions/*.json keeps the full CoT and only
# the scored copy is trimmed; for non-thinking models it is a no-op beyond one
# strip (every official checker strips first anyway).
#
# ORDER MATTERS: with ``merge_base_kwargs: true`` the rewiring block below copies
# the *base* evaluator dict and overrides only ``type``. Setting this afterwards
# would still work for this file, but would silently drop the postprocessor for
# anything driving the same manifest through ``zipbench.apply.apply_zip_subset``.
# Keep it above the block.
#
# ``type`` is the registered STRING name, never the imported function: this dict
# is dumped into the run.py worker subprocess, where a LazyObject would not
# survive.
for _d in datasets:
    _d['eval_cfg']['evaluator']['pred_postprocessor'] = dict(
        type='extract-non-reasoning-content')

_SUBSETS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                        'subsets', DATASET_KEY)

# Rewire the full dataset(s) onto the ZipBench weighted-subset loader/evaluator
# (skipped entirely when SUBSET == 'full' -> plain full, unweighted eval).
#
# This block is benchmark-INDEPENDENT: it reads the manifest's dataset /
# evaluator wiring and mirrors ``zipbench.apply.apply_one`` with the standard
# library only (lazy-import-safe). To add a benchmark, copy this block verbatim
# and change only ``DATASET_KEY``, the imported ``datasets``, and the model.
if SUBSET != 'full':
    with open(osp.join(_SUBSETS_DIR, 'manifest.json'), encoding='utf-8') as _f:
        _manifest = json.load(_f)
    if SUBSET not in _manifest['subsets']:
        raise ValueError(
            f'unknown SUBSET={SUBSET!r}; available: '
            f"{['full', *_manifest['subsets']]}")
    _spec_path = osp.join(_SUBSETS_DIR, _manifest['subsets'][SUBSET]['file'])

    _ds_section = _manifest.get('dataset') or {}
    _loader_kwargs = dict(_ds_section.get('loader_kwargs') or {})
    if 'base_loader' in _manifest:  # legacy top-level fallback
        _loader_kwargs.setdefault('base_loader', _manifest['base_loader'])
        _loader_kwargs.setdefault('id_field', _manifest.get('id_field', '_id'))
    _dataset_type = _ds_section.get('type',
                                    'zipbench.dataset.ZipSubsetDataset')
    _ev_section = _manifest.get('evaluator') or {}
    _evaluator_type = _ev_section.get(
        'type', 'zipbench.evaluator.WeightedAccuracyEvaluator')
    _merge = bool(_ev_section.get('merge_base_kwargs', False))
    _inject = bool(_ev_section.get('inject_subset_spec', False))
    _breakdown = list(_manifest.get('breakdown_fields') or [])

    for _ds in datasets:
        _ds['type'] = _dataset_type
        for _k, _v in _loader_kwargs.items():
            _ds[_k] = _v
        _ds['subset_spec'] = _spec_path
        _ds['abbr'] = f"{_ds['abbr']}_{SUBSET}"
        _ds.setdefault('eval_cfg', {})
        _base_ev = _ds['eval_cfg'].get('evaluator', {}) if _merge else {}
        _new_ev = dict(_base_ev)
        _new_ev['type'] = _evaluator_type
        if _breakdown:
            _new_ev['breakdown_fields'] = _breakdown
        if _inject:
            _new_ev['subset_spec'] = _spec_path
        _ds['eval_cfg']['evaluator'] = _new_ev

# ---- model settings ---------------------------------------------------------
# Inference params (max_out_len / temperature / sampling) belong to the model
# config -- do NOT override them on the model here. IFEval_gen_353ae7 bakes no
# max_out_len into its inferencer, so the model-config value applies.

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = f'./outputs/zipbench/ifeval_{SUBSET}'

# run.py re-dumps this config to a .py file and re-parses it; module-level
# imports / file handles / temp vars are not valid when serialised. Keep only
# the keys OpenCompass consumes so the dumped file stays valid Python.
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
