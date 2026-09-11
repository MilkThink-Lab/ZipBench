"""ZipBench evaluation for SciCode (full / tiny / small weighted subsets).

A regular OpenCompass config -- launch through ``run.py`` from the repo root::

    cd opencompass
    python run.py zipbench/configs/eval_scicode.py

Two knobs:
  * ``VARIANT``  -- ``'wo_bg'`` or ``'with_bg'`` (selects the dataset file and
    the subset namespace ``SciCode`` / ``SciCode_with_background``);
  * ``SUBSET``   -- ``'full'`` (complete 65-problem set, unweighted) or a
    compressed weighted subset (``'small'`` / ``'tiny'``).

Unlike LongBench v2, SciCode's anchor selects *sub-steps*, not rows, and is
scored by code execution, so the subset is wired to the SciCode adapter
(``zipbench.adapters.scicode``) -- but the rewiring block below is the *same*
benchmark-independent block as ``eval_longbenchv2.py``: it just reads the
manifest's dataset/evaluator wiring. The base datasets are built inline (real
classes, non-lazy) like ``examples/eval_scicode.py``; the block then overwrites
the dataset/evaluator ``type`` with fully-qualified strings that survive being
dumped into the run.py worker subprocesses.

``SUBSET != 'full'`` keeps every *touched* problem at full length by default
(predictions identical to a full run; only untouched problems are skipped). Set
``TRUNCATE = True`` to additionally cut each problem to its last selected step
-- safe (a step only depends on earlier steps) and it saves the trailing
generations too.
"""
import json
import os.path as osp

from mmengine.config import read_base

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import SciCodeChatInferencer
from opencompass.datasets import SciCodeDataset, SciCodeEvaluator

with read_base():
    from opencompass.configs.models.deepseek.vllm_deepseek_r1_32b import \
        models as deepseek_r1_32b_model
    from opencompass.configs.models.qwen3.vllm_qwen3_30b_think import \
        models as qwen3_30b_think_model

# ---------------- User-editable knobs ----------------
VARIANT = 'with_bg'   # 'wo_bg' | 'with_bg'
SUBSET = 'tiny'       # 'full' | 'small' | 'tiny'
TRUNCATE = True      # True: also drop each problem's steps after its last
#                       selected one (extra generation savings, still exact for
#                       the selected steps).

# cot_prompt = " Let's think step by step."
PROMPT_SUFFIX = None
GEN_PRESET = 'deterministic'   # 'deterministic' | 'sampling'

DATASET_PATH = './data/scicode'
INFER_MAX_OUT_LEN = 10000
OVERRIDE_MODEL_LIMITS = dict(max_out_len=4096)
OVERRIDE_GEN_KWARGS = dict(temperature=0)

GEN_PRESET_KWARGS = dict(
    deterministic=dict(temperature=0),
    sampling=dict(temperature=0.6, top_p=0.95),
)
VARIANT_TO_WITH_BG = dict(wo_bg=False, with_bg=True)
# The subset namespace == the anchor's dataset key == the base abbr.
VARIANT_TO_KEY = dict(wo_bg='SciCode', with_bg='SciCode_with_background')

if VARIANT not in VARIANT_TO_WITH_BG:
    raise ValueError(f'Unsupported VARIANT: {VARIANT}')
if GEN_PRESET not in GEN_PRESET_KWARGS:
    raise ValueError(f'Unsupported GEN_PRESET: {GEN_PRESET}')

with_bg = VARIANT_TO_WITH_BG[VARIANT]
DATASET_KEY = VARIANT_TO_KEY[VARIANT]
prompt_text = '{prompt}' if not PROMPT_SUFFIX else '{prompt}\n' + PROMPT_SUFFIX

# -------------- Base (full) SciCode dataset, built inline --------------
SciCode_reader_cfg = dict(input_columns=['prompt'], output_column=None)
SciCode_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt=prompt_text)]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=SciCodeChatInferencer,
        infer_mode='every',
        max_out_len=INFER_MAX_OUT_LEN,
    ),
)
SciCode_eval_cfg = dict(
    evaluator=dict(
        type=SciCodeEvaluator,
        dataset_path=DATASET_PATH,
        with_bg=with_bg,
    ))

datasets = [
    dict(
        abbr=DATASET_KEY,
        type=SciCodeDataset,
        path=DATASET_PATH,
        with_bg=with_bg,
        reader_cfg=SciCode_reader_cfg,
        infer_cfg=SciCode_infer_cfg,
        eval_cfg=SciCode_eval_cfg,
    )
]

# ---- Rewire onto the ZipBench weighted subset (benchmark-independent) ----
# Identical to the block in ``eval_longbenchv2.py``: read the manifest's
# dataset/evaluator wiring and apply it with the standard library only.
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
        _ds['truncate'] = TRUNCATE
        _ds['abbr'] = f"{_ds['abbr']}_{SUBSET}"
        _ds.setdefault('eval_cfg', {})
        _base_ev = _ds['eval_cfg'].get('evaluator', {}) if _merge else {}
        _new_ev = dict(_base_ev)
        _new_ev['type'] = _evaluator_type
        _new_ev['breakdown_fields'] = _breakdown
        if _inject:
            _new_ev['subset_spec'] = _spec_path
        _ds['eval_cfg']['evaluator'] = _new_ev

# -------------------------------- Models --------------------------------
models = qwen3_30b_think_model
generation_kwargs = GEN_PRESET_KWARGS[GEN_PRESET].copy()
generation_kwargs.update(OVERRIDE_GEN_KWARGS)
for _m in models:
    for _key, _value in OVERRIDE_MODEL_LIMITS.items():
        _m[_key] = _value
    _m['generation_kwargs'] = generation_kwargs.copy()

# ------------------------------ Inference -------------------------------
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

work_dir = f'./outputs/zipbench/scicode_{VARIANT}_{SUBSET}'

# --- run.py config-dump fix -------------------------------------------------
# run.py does NOT use this module's namespace directly: it re-serialises the
# resolved config to a temporary .py file and re-parses it. That dump emits one
# `name = <repr>` line per surviving module-level name, so any object whose
# repr is not valid Python breaks the re-parse with a SyntaxError -- e.g. the
# `json`/`os.path` modules imported at the top, the manifest file handle (`_f`),
# and the `VARIANT`/`SUBSET`/`_*` temporaries used above:
#     SyntaxError: invalid syntax
#     _f=<_io.TextIOWrapper name='.../manifest.json' mode='r' ...>
# Drop everything except the keys OpenCompass actually consumes so the dumped
# file stays valid Python. (`_n` is excluded from the sweep, then deleted, so it
# doesn't delete itself mid-loop.)
for _n in [n for n in list(globals())
           if not n.startswith('__')
           and n not in ('datasets', 'models', 'infer', 'eval',
                         'summarizer', 'work_dir', '_n')]:
    globals().pop(_n, None)
del _n
