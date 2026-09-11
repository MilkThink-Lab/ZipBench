"""ZipBench evaluation for SimpleQA (full / tiny / small subsets).

Launch through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_simpleqa.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- complete SimpleQA test set (4326 questions, unweighted)
  * ``'small'`` -- compressed weighted subset (1085 questions)
  * ``'tiny'``  -- compressed weighted subset (274 questions)

SimpleQA is scored by an LLM judge (grade A/B/C per answer), so this config
runs the *subjective* eval pipeline (``SubjectiveEvalTask`` injects
``judge_cfg`` / ``output_path`` into ``LMEvaluator``). Unlike the row-level
exact-match benchmarks, the subset rewiring keeps ``LMEvaluator`` and only
swaps its ``dict_postprocessor`` for the weighted ZipBench one
(``zipbench.adapters.simpleqa.simpleqa_zip_postprocess``); the dataset side
uses the generic ``ZipSubsetDataset``.

The default judge is Qwen3-30B-Instruct. See ``eval_longbenchv2.py`` for why the rewiring below is written
inline with stdlib-only imports (mmengine lazy-import mode under
``read_base()``).
"""
import json
import os.path as osp

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import DefaultSubjectiveSummarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

with read_base():
    from opencompass.configs.datasets.SimpleQA.simpleqa_gen import \
        simpleqa_datasets as datasets
    # Generation model (infer stage) -- swap for the model under test.
    from opencompass.configs.models.qwen2_5.vllm_qwen2_5_3b_instruct import \
        models
    # Judge model (eval stage).
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_instruct import \
        models as judge_models

SUBSET = 'full'  # 'full' | 'tiny' | 'small'
DATASET_KEY = 'SimpleQA'

_SUBSETS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                        'subsets', DATASET_KEY)

# Rewire the full dataset(s) onto the ZipBench weighted subset (skipped when
# SUBSET == 'full' -> plain full, unweighted eval). Mirrors the generic block
# in eval_longbenchv2.py, except the evaluator stays LMEvaluator and only its
# dict_postprocessor is replaced by the manifest's weighted postprocessor.
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
    _dataset_type = _ds_section.get('type',
                                    'zipbench.dataset.ZipSubsetDataset')
    _postproc_type = (_manifest.get('evaluator') or {}).get(
        'type', 'zipbench.adapters.simpleqa.simpleqa_zip_postprocess')

    for _ds in datasets:
        _ds['type'] = _dataset_type
        for _k, _v in _loader_kwargs.items():
            _ds[_k] = _v
        _ds['subset_spec'] = _spec_path
        _ds['abbr'] = f"{_ds['abbr']}_{SUBSET}"
        _ds['eval_cfg']['evaluator']['dict_postprocessor'] = dict(
            type=_postproc_type, subset_spec=_spec_path)

summarizer = dict(type=DefaultSubjectiveSummarizer)

# ------------- Inference Stage ----------------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

# ------------- Evaluation Stage ---------------------------------------
eval = dict(
    partitioner=dict(
        type=SubjectiveNaivePartitioner,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner,
                max_num_workers=256,
                task=dict(type=SubjectiveEvalTask)),
)

work_dir = f'./outputs/zipbench/simpleqa_{SUBSET}'

# --- run.py config-dump fix -------------------------------------------------
# run.py re-serialises the resolved config; module objects / file handles /
# temporaries dump as invalid Python, so drop everything OpenCompass does not
# consume (see eval_longbenchv2.py for the full explanation).
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'judge_models', 'infer', 'eval',
                         'summarizer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
