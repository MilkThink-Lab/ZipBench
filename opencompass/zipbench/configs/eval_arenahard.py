"""ZipBench evaluation for ArenaHard (full / small / tiny subsets).

Launch through ``run.py``::

    cd opencompass
    python run.py zipbench/configs/eval_arenahard.py

Switch ``SUBSET`` below between:
  * ``'full'``  -- complete ArenaHard set (500 questions, unweighted; official
    MLE-ELO leaderboard summarizer)
  * ``'small'`` -- compressed weighted subset (n=250, ratio 0.4995)
  * ``'tiny'``  -- compressed weighted subset (n=123, ratio 0.7532)

``ratio`` is the compression rate 1 - n_subset/n_total.

Pipeline mirrors ``examples/eval_arenahard_fast.py``: the subjective eval
pipeline (``SubjectiveEvalTask``) with the ``arena_hard_compare`` dataset
config (m2n vs the gpt4-0314 baseline, ``infer_order='double'``), judge =
Qwen3-235B-Instruct. For subsets, the dataset is rewired onto
``ZipSubsetDataset`` and ``LMEvaluator`` gets the weighted
``zipbench.adapters.arenahard.arenahard_zip_postprocess`` (per-question soft
win rate; see that module); the dataset-level ``ArenaHardSummarizer`` (an ELO
fit over the full battle set) is removed for subsets.

Baseline answers footgun: ``SubjectiveEvalTask`` loads the gpt4-0314 answers
from ``<given_pred path>/<dataset_abbr>.json`` and joins them BY POSITION, so
each subset needs its own baseline file with the spec's rows re-keyed
'0'..'n-1' (``data/subjective/arena_hard/arenahard_{small,tiny}.json``,
extracted from the full ``arenahard.json`` in spec order). Build them with
``python zipbench/make_arenahard_subset_baselines.py`` (re-run if the specs
change).

Model inference parameters (max_out_len, temperature, ...) belong to the
model config files, not here. See ``eval_longbenchv2.py`` for why the
rewiring below is written inline with stdlib-only imports (mmengine
lazy-import mode under ``read_base()``).
"""
import json
import os.path as osp

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import (DefaultSubjectiveSummarizer,
                                     SubjectiveSummarizer)
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

with read_base():
    from opencompass.configs.datasets.subjective.arena_hard.arena_hard_compare import \
        arenahard_datasets as datasets
    # Generation model (infer stage) -- swap for the model under test.
    from opencompass.configs.models.qwen3.vllm_qwen3_4b_instruct import \
        models
    # Judge model (eval stage).
    from opencompass.configs.models.qwen3.vllm_qwen3_235b_instruct import \
        models as judge_models

SUBSET = 'full'  # 'full' | 'small' | 'tiny'
DATASET_KEY = 'ArenaHard'

_SUBSETS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                        'subsets', DATASET_KEY)

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
        'type', 'zipbench.adapters.arenahard.arenahard_zip_postprocess')

    for _ds in datasets:
        _ds['type'] = _dataset_type
        for _k, _v in _loader_kwargs.items():
            _ds[_k] = _v
        _ds['subset_spec'] = _spec_path
        # The abbr doubles as the baseline-answer filename (see docstring).
        _ds['abbr'] = f"{_ds['abbr']}_{SUBSET}"
        _ds['eval_cfg']['evaluator']['dict_postprocessor'] = dict(
            type=_postproc_type, subset_spec=_spec_path)
        # ELO summarizer needs the full battle set; drop it for subsets.
        _ds.pop('summarizer', None)
    summarizer = dict(type=DefaultSubjectiveSummarizer)
else:
    summarizer = dict(type=SubjectiveSummarizer, function='subjective')

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

work_dir = f'./outputs/zipbench/arenahard_{SUBSET}'

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
