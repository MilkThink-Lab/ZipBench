"""ZipBench evaluation for MMLU-Pro (full / small / tiny subsets).

This is a regular OpenCompass config and MUST be launched through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_mmlu_pro.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- 14 categories flattened, native HF test order (12032 rows,
                   unweighted)
  * ``'small'`` -- compressed weighted subset (388 rows, ratio 0.9678)
  * ``'tiny'``  -- compressed weighted subset (259 rows, ratio 0.9785)


Mirrors ``examples/eval_mmlu_pro_0shot_cot.py`` (OpenCompass default alias
``mmlu_pro_gen`` -> ``mmlu_pro_0shot_cot_gen_08c1de``, openai simple-evals style):
0-shot CoT prompt ending "ANSWER: $LETTER", ``match_answer_pattern`` extraction
over A-P, single-pass generation, qwen3-30b-think.

Generation params (max_out_len,
temperature) are NOT set here -- they come from the model config .

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
from opencompass.utils.text_postprocessors import match_answer_pattern

with read_base():
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import models

SUBSET = 'small'
DATASET_KEY = 'MMLU_Pro'

# ---- dataset (mirrors mmlu_pro_0shot_cot_gen_08c1de QUERY_TEMPLATE) -----------
QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.

Question:\n
{question}

Options:\n
{options_str}

""".strip()

datasets = [dict(
    abbr='mmlu_pro',
    type='zipbench.adapters.mmlu_pro.MMLUProAllDataset',
    path='opencompass/mmlu_pro',
    reader_cfg=dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer'),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    ),
    eval_cfg=dict(
        evaluator=dict(type='zipbench.evaluator.WeightedAccuracyEvaluator'),
        pred_postprocessor=dict(type=match_answer_pattern,
                                answer_pattern=r'(?i)ANSWER\s*:\s*([A-P])'),
    ),
)]

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
        _new_ev['breakdown_fields'] = _breakdown
        if _inject:
            _new_ev['subset_spec'] = _spec_path
        _ds['eval_cfg']['evaluator'] = _new_ev

# ---- model settings ---------------------------------------------------------
# No generation params are set here: max_out_len AND temperature both belong in
# the model config (opencompass/configs/models/qwen3/vllm_qwen3_30b_think.py),
# not in eval scripts.

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = f'./outputs/zipbench/mmlu_pro_{SUBSET}'

# run.py re-dumps this config to a .py file and re-parses it; module-level
# imports / file handles / temp vars are not valid when serialised. Keep only
# the keys OpenCompass consumes so the dumped file stays valid Python.
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
