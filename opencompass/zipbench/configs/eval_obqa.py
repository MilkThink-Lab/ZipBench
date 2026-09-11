"""ZipBench evaluation for OpenBookQA (full / tiny / small subsets).

This is a regular OpenCompass config and MUST be launched through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_obqa.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- complete OpenBookQA test set (500 questions per variant)
  * ``'small'`` -- compressed weighted subsets (244 / 219 questions)
  * ``'tiny'``  -- compressed weighted subsets (170 / 125 questions)

OpenBookQA ships two variants that are ALWAYS evaluated together, one
``datasets`` entry each, compressed independently:
  * ``openbookqa``       -- question only            (Main/test.jsonl)
  * ``openbookqa_fact``  -- question + gold fact1    (Additional/test_complete.jsonl)
so every SUBSET run produces two weighted scores, one per variant.

Implementation note (why this file mutates dicts inline instead of calling a
helper): ``with read_base()`` puts mmengine into *lazy-import* mode, where every
non-builtin import becomes an uncallable ``LazyObject`` (usable only as a
``type=`` reference). So we cannot call ``zipbench.apply_zip_subset`` here; we
read the manifests with the standard library (``json``) and set fully-qualified
*string* ``type``s for the custom dataset / evaluator, which dump cleanly into
the run.py worker subprocesses.
"""
import json
import os.path as osp

from mmengine.config import read_base

from opencompass.datasets import OBQADataset_Local
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.utils.text_postprocessors import first_option_postprocess

with read_base():
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import models

SUBSET = 'full'

# subsets dir per dataset abbr -- the two variants are compressed separately, so
# each has its own anchor specs / manifest.
_DATASET_KEYS = {'openbookqa': 'OpenBookQA', 'openbookqa_fact': 'OpenBookQA_fact'}

# ---- dataset (mirrors examples/eval_obqa.py, the full-set reference) --------
_DATA_ROOT = ('./data/openbookqa')

_input_columns = [
    ['question_stem', 'A', 'B', 'C', 'D'],
    ['question_stem', 'A', 'B', 'C', 'D', 'fact1'],
]
_template = [
    dict(
        round=[
            dict(
                role='HUMAN',
                prompt='Question: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'
            ),
            dict(role='BOT', prompt='<think>\n'),
        ], ),
    dict(
        round=[
            dict(
                role='HUMAN',
                prompt='Given the fact: {fact1}\nQuestion: {question_stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'
            ),
            dict(role='BOT', prompt='<think>\n'),
        ], ),
]

datasets = [
    dict(
        abbr='openbookqa',
        type=OBQADataset_Local,
        path=f'{_DATA_ROOT}/Main/test.jsonl',
        name='main',
    ),
    dict(
        abbr='openbookqa_fact',
        type=OBQADataset_Local,
        path=f'{_DATA_ROOT}/Additional/test_complete.jsonl',
        name='additional',
    ),
]

for _i in range(2):
    datasets[_i]['reader_cfg'] = dict(
        input_columns=_input_columns[_i], output_column='answerKey')
    datasets[_i]['infer_cfg'] = dict(
        prompt_template=dict(type=PromptTemplate, template=_template[_i]),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )
    datasets[_i]['eval_cfg'] = dict(
        evaluator=dict(type=AccEvaluator),
        pred_role='BOT',
        pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    )

_SUBSETS_ROOT = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                         'subsets')

# Rewire the full dataset(s) onto the ZipBench weighted-subset loader/evaluator
# (skipped entirely when SUBSET == 'full' -> plain full, unweighted eval).
#
# This block is benchmark-INDEPENDENT: it reads each manifest's dataset /
# evaluator wiring and mirrors ``zipbench.apply.apply_one`` with the standard
# library only (lazy-import-safe). Same block as eval_commonsenseqa.py, except
# the manifest is looked up per dataset abbr (two independently-compressed
# variants) instead of from a single DATASET_KEY.
if SUBSET != 'full':
    for _ds in datasets:
        _subsets_dir = osp.join(_SUBSETS_ROOT, _DATASET_KEYS[_ds['abbr']])
        with open(osp.join(_subsets_dir, 'manifest.json'),
                  encoding='utf-8') as _f:
            _manifest = json.load(_f)
        if SUBSET not in _manifest['subsets']:
            raise ValueError(
                f'unknown SUBSET={SUBSET!r}; available: '
                f"{['full', *_manifest['subsets']]}")
        _spec_path = osp.join(_subsets_dir,
                              _manifest['subsets'][SUBSET]['file'])

        _ds_section = _manifest.get('dataset') or {}
        _loader_kwargs = dict(_ds_section.get('loader_kwargs') or {})
        if 'base_loader' in _manifest:  # legacy top-level fallback
            _loader_kwargs.setdefault('base_loader', _manifest['base_loader'])
            _loader_kwargs.setdefault('id_field',
                                      _manifest.get('id_field', '_id'))
        _dataset_type = _ds_section.get('type',
                                        'zipbench.dataset.ZipSubsetDataset')
        _ev_section = _manifest.get('evaluator') or {}
        _evaluator_type = _ev_section.get(
            'type', 'zipbench.evaluator.WeightedAccuracyEvaluator')
        _merge = bool(_ev_section.get('merge_base_kwargs', False))
        _inject = bool(_ev_section.get('inject_subset_spec', False))
        _breakdown = list(_manifest.get('breakdown_fields') or [])

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

# Inference params (max_out_len / temperature / sampling) belong to the model
# config -- do NOT override them on the model here.

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = f'./outputs/zipbench/obqa_{SUBSET}'

# run.py re-dumps this config to a .py file and re-parses it; module-level
# imports / file handles / temp vars are not valid when serialised. Keep only
# the keys OpenCompass consumes so the dumped file stays valid Python.
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
