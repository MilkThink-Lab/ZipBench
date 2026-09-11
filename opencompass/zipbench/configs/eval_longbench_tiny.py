"""ZipBench evaluation for LongBench v1 (full / small / tiny subsets).

This is a regular OpenCompass config and MUST be launched through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_longbench.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- all 21 LongBench v1 tasks concatenated (4750 questions)
  * ``'small'`` -- compressed weighted subset (885 questions, ratio 0.8137)
  * ``'tiny'``  -- compressed weighted subset (151 questions, ratio 0.9682)

The 21 heterogeneous per-task configs of ``examples/eval_longbench.py`` are
folded into ONE dataset: LongbenchAllDataset pre-renders each task's official
prompt into the ``prompt`` column (so the single ``{prompt}`` template below
renders byte-identical prompts), and LongBenchGenInferencer honours the per-row
official ``max_out_len`` (32/64/128/512) by grouping each batch by that column.
Scoring is WeightedLongbenchEvaluator for full and subsets alike (on the full
set the weights are uniform): primary score = six-category equal-weight macro
of the per-item official LongBench scores; the flat micro and the official
EN/ZH leaderboard table averages are in ``secondary``.

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

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

from zipbench.adapters.longbench import (LongbenchAllDataset,
                                         LongBenchGenInferencer,
                                         WeightedLongbenchEvaluator)

with read_base():
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models

SUBSET = 'tiny'
DATASET_KEY = 'Longbench'

# ---------------------------------------------------------------------------
# Output length
# ---------------------------------------------------------------------------
# None = keep the official per-task max_out_len (32/64/128/512, stored in the
# dataset's ``max_out_len`` column, applied per row by LongBenchGenInferencer).
# LongBench does no answer extraction, so anything beyond the expected answer
# is penalised by precision -- raising the caps only lowers scores. Thinking
# models must override this (a 32-token cap cannot even emit ``</think>``),
# but doing so departs from the official setting and the results are no
# longer comparable to the official leaderboard / paper numbers.
THINKING_BUDGET = None  # e.g. 8192 for thinking models

# ---- dataset (mirrors the 21 upstream per-task configs, but as one
# concatenated dataset; prompts are pre-rendered per row so ``{prompt}`` is the
# whole template, and the per-task output cap lives in the ``max_out_len``
# column consumed by LongBenchGenInferencer; its ctor ``max_out_len`` is only
# the fallback for a missing column).
longbench_reader_cfg = dict(input_columns=['prompt'], output_column='answers')

longbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{prompt}')]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=LongBenchGenInferencer, max_out_len=512),
)

longbench_eval_cfg = dict(
    evaluator=dict(type=WeightedLongbenchEvaluator),
    pred_role='BOT',
)

datasets = [
    dict(
        abbr='longbench',
        type=LongbenchAllDataset,
        path='opencompass/Longbench',
        reader_cfg=longbench_reader_cfg,
        infer_cfg=longbench_infer_cfg,
        eval_cfg=longbench_eval_cfg,
    )
]

if THINKING_BUDGET is not None:
    for _d in datasets:
        _d['infer_cfg']['inferencer']['max_out_len_override'] = THINKING_BUDGET

_SUBSETS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                        'subsets', DATASET_KEY)

# Rewire the full dataset(s) onto the ZipBench weighted-subset loader/evaluator
# (skipped entirely when SUBSET == 'full' -> plain full, uniform weights).
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

# Sampling parameters (temperature etc.) belong to the model config file;
# this script does not override them. Note the official LongBench pred.py
# uses do_sample=False / num_beams=1, i.e. greedy decoding.
for model in models:
    # Do NOT set max_tokens in the model's generation_kwargs:
    # VLLMwithChatTemplate.generate seeds max_tokens from the per-call
    # max_out_len and then applies
    # ``sampling_kwargs.update(self.generation_kwargs)``, so the per-row caps
    # would be silently replaced by that one constant -- no error is raised.

    # CoT stripping is model-level (runs before scoring). strip=False is
    # required: the reference answers of the code tasks are indented code
    # lines and fuzz.ratio is whitespace-sensitive. Without think markers the
    # postprocessor is a strict no-op, so non-thinking models are unaffected.
    model['pred_postprocessor'] = dict(type=extract_non_reasoning_content,
                                       strip=False)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = f'./outputs/zipbench/longbench_{SUBSET}'

# run.py re-dumps this config to a .py file and re-parses it; module-level
# imports / file handles / temp vars are not valid when serialised. Keep only
# the keys OpenCompass consumes so the dumped file stays valid Python.
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
