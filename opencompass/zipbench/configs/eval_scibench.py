"""ZipBench evaluation for SciBench (full / small / tiny subsets).

This is a regular OpenCompass config and MUST be launched through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_scibench.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- all 10 subjects concatenated (583 questions, unweighted)
  * ``'small'`` -- compressed weighted subset (272 questions, ratio 0.5307;
                   one item, fund-16 / "Question 23.55", was dropped
                   because upstream SciBench deleted it)
  * ``'tiny'``  -- compressed weighted subset (146 questions, ratio 0.7492)

Prompt / tolerances / postprocessing mirror the full-set reference config
(``examples/eval_scibench.py``, cot_prompt variant, max_out_len=4096,
NumericAccEvaluator(rtol=1e-2, atol=1e-3) + scibench_postprocess). The only
structural difference: the 10 per-subject datasets are replaced by ONE
concatenated dataset (``zipbench.adapters.scibench.ScibenchZipDataset``) so
row indices match the anchor spec's index space; 'full' therefore reports one
overall accuracy instead of per-subject rows (per-subject merging was the job
of ScibenchSummarizer, which the weighted subsets don't need).

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

from opencompass.openicl.icl_evaluator import NumericAccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_14b import \
        models

SUBSET = 'full'
DATASET_KEY = 'Scibench'

# ---- dataset (mirrors examples/eval_scibench.py, concatenated) --------------
cot_prompt = "Let's think step by step."

datasets = [dict(
    abbr='scibench',
    type='zipbench.adapters.scibench.ScibenchZipDataset',
    path='./data/scibench',
    reader_cfg=dict(input_columns=['question'], output_column='answer'),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=(
                        'Please provide a clear and step-by-step solution '
                        'for a scientific problem in the categories of '
                        'Chemistry, Physics, or Mathematics. The problem '
                        'will specify the unit of measurement, which should '
                        'not be included in the answer. Express the final '
                        'answer as a decimal number with three digits after '
                        'the decimal point. Conclude the answer by stating '
                        "'Therefore, the answer is \\boxed[ANSWER].'\\n\\n"
                        'Problem: {question}\\nAnswer: ' + cot_prompt),
                ),
                dict(role='BOT'),
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=4096),
    ),
    eval_cfg=dict(
        evaluator=dict(type=NumericAccEvaluator, rtol=1e-2, atol=1e-3),
        pred_postprocessor=dict(
            type='opencompass.datasets.scibench.scibench_postprocess'),
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
        if _breakdown:
            _new_ev['breakdown_fields'] = _breakdown
        if _inject:
            _new_ev['subset_spec'] = _spec_path
        _ds['eval_cfg']['evaluator'] = _new_ev

# Inference params (max_out_len / temperature / sampling) belong to the model
# config -- do NOT override them on the model here. The dataset inferencer
# already carries the answer budget (max_out_len=4096).

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = f'./outputs/zipbench/Scibench_{SUBSET}'

# run.py re-dumps this config to a .py file and re-parses it; module-level
# imports / file handles / temp vars are not valid when serialised. Keep only
# the keys OpenCompass consumes so the dumped file stays valid Python.
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
