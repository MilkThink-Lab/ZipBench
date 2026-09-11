"""ZipBench evaluation for MBPP+ (full / tiny / small subsets).

This is a regular OpenCompass config and MUST be launched through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_mbpp_plus.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- complete MBPP+ test set (399 problems, MbppPlus v0.1.0)
  * ``'small'`` -- compressed weighted subset (200 problems, ratio 0.4995)
  * ``'tiny'``  -- compressed weighted subset (88 problems, ratio 0.7805)

The dataset/prompt/eval wiring (3-shot [BEGIN]/[DONE] prompt over the
original MBPP problem text and asserts; scored by the evalplus base+plus
harness) is copied verbatim from the reference config
``opencompass/configs/datasets/mbpp_plus/mbpp_plus_gen_0b836a.py``. Only
the dataset side is replicated here; the model below is a swappable example.
Note the dataset-level inferencer ``max_out_len`` takes precedence over the
model config's value.

Implementation note (why this file mutates dicts inline instead of calling a
helper): ``with read_base()`` puts mmengine into *lazy-import* mode, where
every non-builtin import becomes an uncallable ``LazyObject`` (usable only as
a ``type=`` reference). So we cannot call ``zipbench.apply_zip_subset`` here;
we read the manifest with the standard library (``json``) and set
fully-qualified *string* ``type``s for the custom dataset / evaluator, which
dump cleanly into the run.py worker subprocesses.
"""
import json
import os.path as osp

from mmengine.config import read_base

from opencompass.datasets import MBPPEvaluator, MBPPPlusDataset
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import \
        models as qwen3_4b_instruct_model

SUBSET = 'small'
DATASET_KEY = 'MbppPlus'

# -- dataset wiring replicated from
#    opencompass/configs/datasets/mbpp_plus/mbpp_plus_gen_0b836a.py
mbpp_plus_reader_cfg = dict(
    input_columns=['text', 'test_list'], output_column='task_id')

mbpp_plus_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                'You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n'
            ),
            dict(
                role='BOT',
                prompt=
                "[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\n  return tuple(set(test_tup1) & set(test_tup2))' \n[DONE] \n\n "
            ),
            dict(
                role='HUMAN',
                prompt=
                'You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n'
            ),
            dict(
                role='BOT',
                prompt=
                "[BEGIN]\n 'import math\ndef is_not_prime(n):\n    if n == 1:\n        return True\n    for i in range(2, int(math.sqrt(n))+1):\n        if n % i == 0:\n            return True\n    return False' \n[DONE] \n\n "
            ),
            dict(
                role='HUMAN',
                prompt=
                'You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n'
            ),
            dict(
                role='BOT',
                prompt=
                "[BEGIN]\n 'import heapq as hq\ndef heap_queue_largest(nums: list,n: int) -> list:\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums' \n[DONE] \n\n "
            ),
            dict(
                role='HUMAN',
                prompt=
                'You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_list}  \n '
            ),
            dict(role='BOT', prompt='[BEGIN]\n'),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

mbpp_plus_eval_cfg = dict(
    evaluator=dict(type=MBPPEvaluator, metric='MBPPPlus'),
    pred_role='BOT',
)

datasets = [
    dict(
        abbr='mbpp_plus',
        type=MBPPPlusDataset,
        path='./data/mbpp_plus/mbpp_plus.jsonl',
        reader_cfg=mbpp_plus_reader_cfg,
        infer_cfg=mbpp_plus_infer_cfg,
        eval_cfg=mbpp_plus_eval_cfg,
    )
]

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

# ---- model settings ----
# Example model only: swap in whatever model you want to evaluate. The subset
# selection and evaluation above are model-independent, so the choice of model
# here does not affect the ZipBench wiring.
models = qwen3_4b_instruct_model

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = f'./outputs/zipbench/MbppPlus_{SUBSET}'

# run.py re-dumps this config to a .py file and re-parses it; module-level
# imports / file handles / temp vars are not valid when serialised. Keep only
# the keys OpenCompass consumes so the dumped file stays valid Python.
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
