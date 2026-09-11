"""ZipBench evaluation for MuSR (full / tiny / small subsets).

This is a regular OpenCompass config and MUST be launched through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_musr.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- all 3 MuSR scenarios concatenated (756 questions)
  * ``'small'`` -- compressed weighted subset (331 questions, ratio 0.5622)
  * ``'tiny'``  -- compressed weighted subset (190 questions, ratio 0.7487)

Unlike the upstream ``musr_gen_b47fd3`` (3 *separate* datasets, one ``name=`` per
scenario), this uses a single concatenated dataset (``MusrAllDataset``) so a
ZipBench anchor can select rows of one flat index space, scored by
``WeightedMusrEvaluator`` (last ``ANSWER: <int>`` line == gold choice number,
aggregated MACRO -- per-scenario weighted accuracy then an equal-weight mean over
the 3 scenarios, i.e. the official MuSR / ``musr_average`` metric; the flat
``micro_accuracy`` is reported alongside as a secondary metric). Prompt (cot+)
and max_out_len match ``examples/eval_musr_0shot.py`` (the full-set reference).

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

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from zipbench.adapters.musr import MusrAllDataset, WeightedMusrEvaluator

with read_base():
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import models

SUBSET = 'full'
DATASET_KEY = 'MuSR'

# ---- dataset (mirrors musr_gen_b47fd3's cot+ prompt / 2048 tokens, but as one
# concatenated dataset instead of 3). MusrAllDataset pre-renders the full cot+
# prompt into ``prompt`` and the system prompt into ``system_prompt`` (the heavy
# nested columns are pruned before concat), so the template just substitutes
# those two -> the rendered prompt is byte-identical to the upstream per-scenario
# HUMAN/SYSTEM prompt. The gold choice number is ``gold_answer``.
musr_reader_cfg = dict(input_columns=['prompt', 'system_prompt'],
                       output_column='gold_answer')

musr_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN',
                     prompt='{system_prompt}'),
            ],
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048),
)

musr_eval_cfg = dict(evaluator=dict(type=WeightedMusrEvaluator))

datasets = [
    dict(
        abbr='musr',
        type=MusrAllDataset,
        path='opencompass/musr',
        reader_cfg=musr_reader_cfg,
        infer_cfg=musr_infer_cfg,
        eval_cfg=musr_eval_cfg,
    )
]

# Inference params (max_out_len / temperature / sampling) belong to the model
# config -- do NOT override them on the model here.

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

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = f'./outputs/zipbench/musr_{SUBSET}'

# run.py re-dumps this config to a .py file and re-parses it; module-level
# imports / file handles / temp vars are not valid when serialised. Keep only
# the keys OpenCompass consumes so the dumped file stays valid Python.
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
