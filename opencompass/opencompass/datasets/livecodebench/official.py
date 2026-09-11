"""Wrappers around the upstream LiveCodeBench `lcb_runner` package.

The upstream repo lives under `third_party/livecodebench/` (cloned from
https://github.com/LiveCodeBench/LiveCodeBench). This module exposes:

- ``LCBOfficialCodeGenerationDataset``: a BaseDataset whose ``load`` delegates
  to ``lcb_runner.benchmarks.code_generation.load_code_generation_dataset``,
  i.e. the official HF dataset ``livecodebench/code_generation_lite`` resolved
  by ``version_tag=<release_version>``.
- ``LCBOfficialCodeGenerationEvaluator``: a BaseEvaluator that uses
  ``lcb_runner.evaluation.codegen_metrics`` + the official
  ``lcb_runner.utils.extraction_utils.extract_code`` for code extraction.
  After scoring, it also dumps lcb_runner-compatible
  ``Scenario.codegeneration_<n>_<temp>{,_eval,_eval_all}.json`` files next
  to OpenCompass's standard results JSON.

Use them through ``opencompass/configs/datasets/livecodebench/livecodebench_official_gen.py``.
"""

import gc
import json
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
_THIRD_PARTY = _REPO_ROOT / 'third_party' / 'livecodebench'
if not _THIRD_PARTY.exists():
    raise ImportError(
        f'third_party/livecodebench not found at {_THIRD_PARTY}. '
        "Run `git clone https://github.com/LiveCodeBench/LiveCodeBench.git "
        "third_party/livecodebench` from the repo root.")
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

# If the pre-populated HF cache for LiveCodeBench exists (mirroring the layout
# used by the standalone `examples/eval_livecodebench.py`), point HF at it.
# Uses setdefault so an explicit user-supplied HF_HOME/HF_DATASETS_CACHE wins.
#
# The env vars alone are NOT enough here: `datasets` (and huggingface_hub)
# snapshot them into module constants at import time, and by the time this
# module loads through the `opencompass` package chain, `datasets` has long
# been imported with the defaults (~/.cache/huggingface). So in addition to
# the env vars (which cover spawned worker processes), patch the already
# imported `datasets.config` / `huggingface_hub.constants` in place.
_LCB_CACHE_DIR = _REPO_ROOT / 'data' / 'hf_cache_livecodebench_full'
if _LCB_CACHE_DIR.exists():
    _user_set_cache = any(
        v in os.environ
        for v in ('HF_HOME', 'HF_DATASETS_CACHE'))
    os.environ.setdefault('HF_HOME', str(_LCB_CACHE_DIR))
    os.environ.setdefault('HF_DATASETS_CACHE',
                          str(_LCB_CACHE_DIR / 'datasets'))
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE',
                          str(_LCB_CACHE_DIR / 'hub'))
    os.environ.setdefault('HF_HUB_CACHE', str(_LCB_CACHE_DIR / 'hub'))
    # Force offline so load_dataset() doesn't waste ~23s on huggingface.co
    # HEAD-retry storms when the host has no proxy / blocked egress. setdefault
    # keeps user overrides intact (e.g. HF_DATASETS_OFFLINE=0 to refresh).
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    if not _user_set_cache:
        from datasets import config as _ds_config
        _ds_config.HF_DATASETS_CACHE = str(_LCB_CACHE_DIR / 'datasets')
        _ds_config.HF_CACHE_HOME = str(_LCB_CACHE_DIR)
        if os.environ.get('HF_DATASETS_OFFLINE') == '1' \
                or os.environ.get('HF_HUB_OFFLINE') == '1':
            _ds_config.HF_HUB_OFFLINE = True
            try:
                from huggingface_hub import constants as _hub_constants
                _hub_constants.HF_HUB_OFFLINE = True
            except ImportError:
                pass

# Upstream imports — these modules have no module-load file I/O.
# (We deliberately avoid `lcb_runner.prompts.code_generation`, which opens
# few-shot JSON files relative to cwd at import time.)
from lcb_runner.benchmarks.code_generation import (  # noqa: E402
    load_code_generation_dataset, )
from lcb_runner.evaluation import (  # noqa: E402
    codegen_metrics, extract_instance_results)
from lcb_runner.lm_styles import LMStyle  # noqa: E402
from lcb_runner.utils.extraction_utils import extract_code  # noqa: E402

# Mirrored verbatim from lcb_runner/prompts/code_generation.py:35-37 to avoid
# importing that module (which does file-relative loads at module load).
_FORMATTING_WITH_STARTER_CODE = (
    'You will use the following starter code to write the solution to the '
    'problem and enclose your code within delimiters.')
_FORMATTING_WITHOUT_STARTER_CODE = (
    'Read the inputs from stdin solve the problem and write the answer to '
    'stdout (do not directly test on the sample inputs). Enclose your code '
    'within delimiters as follows. Ensure that when the python program runs, '
    'it reads the inputs, runs the algorithm and writes output to STDOUT.')


def _build_format_prompt(starter_code: str) -> str:
    if starter_code:
        return (f'### Format: {_FORMATTING_WITH_STARTER_CODE}\n'
                f'```python\n{starter_code}\n```\n\n')
    return (f'### Format: {_FORMATTING_WITHOUT_STARTER_CODE}\n'
            '```python\n# YOUR CODE HERE\n```\n\n')


def _problem_to_row(problem) -> dict:
    return {
        'question_id': problem.question_id,
        'question_content': problem.question_content,
        'starter_code': problem.starter_code,
        'contest_date': problem.contest_date.isoformat(),
        'format_prompt': _build_format_prompt(problem.starter_code),
    }


@LOAD_DATASET.register_module()
class LCBOfficialCodeGenerationDataset(BaseDataset):
    """LiveCodeBench codegeneration loaded via upstream lcb_runner."""

    @staticmethod
    def load(path: str = 'livecodebench/code_generation_lite',
             release_version: str = 'release_latest',
             start_date: str = None,
             end_date: str = None,
             **kwargs):
        del path, kwargs  # path is fixed by upstream loader; kept for symmetry
        problems = load_code_generation_dataset(
            release_version=release_version,
            start_date=start_date,
            end_date=end_date,
        )
        # Match upstream runner order; see scenario_router.py:60.
        problems.sort(key=lambda p: p.question_id)
        rows = [_problem_to_row(p) for p in problems]
        del problems
        gc.collect()
        ds = Dataset.from_list(rows)
        del rows
        gc.collect()
        return DatasetDict({'test': ds, 'train': ds})


@ICL_EVALUATORS.register_module()
class LCBOfficialCodeGenerationEvaluator(BaseEvaluator):
    """Score predictions with upstream `lcb_runner.evaluation.codegen_metrics`.

    On top of returning OpenCompass's standard result dict, this evaluator also
    writes lcb_runner-compatible
    ``Scenario.codegeneration_<n>_<temperature>{,_eval,_eval_all}.json``
    files next to OpenCompass's results JSON (so downstream tools that expect
    upstream's layout, e.g. ``lcb_runner.evaluation.compute_scores``, can read
    them directly).
    """

    def __init__(self,
                 release_version: str = 'release_latest',
                 start_date: str = None,
                 end_date: str = None,
                 k_list=(1, ),
                 num_process_evaluate: int = 4,
                 timeout: int = 6,
                 lm_style: str = 'OpenAIChat',
                 dump_lcb_runner_format: bool = True,
                 lcb_runner_n: int = 1,
                 lcb_runner_temperature: float = 0.0):
        super().__init__()
        self.release_version = release_version
        self.start_date = start_date
        self.end_date = end_date
        self.k_list = list(k_list)
        self.num_process_evaluate = num_process_evaluate
        self.timeout = timeout
        self.lm_style = LMStyle(lm_style)
        self.dump_lcb_runner_format = dump_lcb_runner_format
        self.lcb_runner_n = lcb_runner_n
        self.lcb_runner_temperature = lcb_runner_temperature
        self._problems = load_code_generation_dataset(
            release_version=release_version,
            start_date=start_date,
            end_date=end_date,
        )
        self._by_qid = {p.question_id: p for p in self._problems}

    def evaluate(self, k, n, original_dataset, predictions, references):
        """Aggregate the framework's dataset-level replicas ourselves.

        With dataset ``n>1`` OpenCompass repeats the test set n times
        (replica blocks concatenated in order) and normally splits
        predictions back per replica, calling ``score()`` once each. We
        override the whole hook instead: group the n predictions per
        question into one ``code_list`` and run ``codegen_metrics`` once,
        yielding the official ``pass@k`` metric names and a single set of
        lcb_runner-format dump files.
        """
        del original_dataset
        predictions = self.pred_postprocess(predictions)
        if len(predictions) != len(references):
            return {
                'error':
                f'len(predictions)={len(predictions)} vs '
                f'len(references)={len(references)}'
            }
        if n <= 0 or len(references) % n != 0:
            return {
                'error':
                f'len(references)={len(references)} not divisible by n={n}'
            }
        real_size = len(references) // n
        qids = list(references[:real_size])
        output_lists = []
        for j in range(real_size):
            outs = []
            for i in range(n):
                idx = i * real_size + j
                if references[idx] != qids[j]:
                    return {
                        'error':
                        f'replica row-order mismatch at index {idx}: '
                        f'{references[idx]!r} != {qids[j]!r}'
                    }
                outs.append(predictions[idx])
            output_lists.append(outs)

        del k  # k_list is derived from n; see _official_k_list
        return self._grouped_eval(qids, output_lists,
                                  self._official_k_list(n), n)

    @staticmethod
    def _official_k_list(n):
        """Official metric set, matching examples/eval_livecodebench.py:
        pass@k for k in (1, 5, 10) filtered to k <= n. When n is too small
        for pass@5, fall back to pass@n (the largest unbiased k) so a
        multi-sample metric is still reported; n=1 yields just pass@1.
        estimate_pass_at_k silently returns 1.0 for k > n, so k must never
        exceed n."""
        k_list = [k for k in (1, 5, 10) if k <= n]
        if 5 not in k_list and n > 1:
            k_list.append(n)
        return sorted(set(k_list))

    def score(self, predictions, references):
        """Single-sample-per-question entry point (direct calls / smokes)."""
        if len(predictions) != len(references):
            return {
                'error':
                f'len(predictions)={len(predictions)} vs '
                f'len(references)={len(references)}'
            }
        return self._grouped_eval(list(references),
                                  [[p] for p in predictions],
                                  list(self.k_list), self.lcb_runner_n)

    def _grouped_eval(self, qids, output_lists, k_list, n):
        """Score one n-element candidate list per question.

        ``qids``/``output_lists`` are parallel; each ``output_lists[j]`` holds
        the raw model outputs for question ``qids[j]``.
        """
        # Official extraction: last fenced ```...``` block.
        samples_list = []
        filtered_codes = []
        filtered_outputs = []
        kept_qids = []
        for qid, outs in zip(qids, output_lists):
            prob = self._by_qid.get(qid)
            if prob is None:
                continue
            samples_list.append(prob.get_evaluation_sample())
            filtered_codes.append(
                [extract_code(o, self.lm_style) for o in outs])
            filtered_outputs.append(list(outs))
            kept_qids.append(qid)

        metrics, eval_results, final_metadata = codegen_metrics(
            samples_list,
            filtered_codes,
            k_list=k_list,
            num_process_evaluate=self.num_process_evaluate,
            timeout=self.timeout,
        )

        instance_results = extract_instance_results(eval_results)
        per_task_pass = metrics.get('detail', {})
        details = []
        for i, qid in enumerate(kept_qids):
            graded = (instance_results[i]
                      if i < len(instance_results) else [])
            detail = {
                'question_id': qid,
                'code_list': filtered_codes[i],
                'graded': graded,
                'metadata':
                final_metadata[i] if i < len(final_metadata) else None,
            }
            for k in k_list:
                task_detail = per_task_pass.get(f'pass@{k}', {})
                if i in task_detail:
                    detail[f'pass@{k}'] = task_detail[i]
            details.append(detail)

        # Rename upstream `detail` (per-task pass@k floats keyed by integer
        # index) to `pass_at_k_per_task` so it doesn't collide with our
        # `details` (per-task debug info).
        result = {}
        for key, v in metrics.items():
            if key == 'detail':
                result['pass_at_k_per_task'] = v
            elif key != 'details':
                result[key] = v
        result['details'] = details

        if self.dump_lcb_runner_format:
            self._dump_lcb_runner_files(
                kept_qids=kept_qids,
                filtered_outputs=filtered_outputs,
                filtered_codes=filtered_codes,
                instance_results=instance_results,
                final_metadata=final_metadata,
                metrics=metrics,
                n=n,
            )

        return result

    # ------------------------------------------------------------------
    # lcb_runner-compatible dumps (Scenario.codegeneration_<n>_<temp>*.json)
    # Format mirrors upstream `lcb_runner.benchmarks.code_generation.CodeGenerationProblem.insert_output_evaluation`.
    # ------------------------------------------------------------------
    def _dump_lcb_runner_files(self, *, kept_qids, filtered_outputs,
                               filtered_codes, instance_results,
                               final_metadata, metrics, n):
        out_root = getattr(self, '_out_dir', None)
        if not out_root:
            return  # not called from openicl_eval task; skip
        out_dir = Path(out_root).parent  # results/<model_abbr>/
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = (f'Scenario.codegeneration_{n}_'
                  f'{self.lcb_runner_temperature}')

        # 1) <prefix>.json — raw outputs + extracted code per question
        raw_records = []
        # 2) <prefix>_eval_all.json — full upstream record
        eval_all_records = []
        for i, qid in enumerate(kept_qids):
            prob = self._by_qid[qid]
            output_list = filtered_outputs[i]
            code_list = filtered_codes[i]
            graded = instance_results[i] if i < len(instance_results) else []
            md = final_metadata[i] if i < len(final_metadata) else []
            raw_records.append({
                'question_id': qid,
                'question_title': prob.question_title,
                'platform': prob.platform.value,
                'contest_id': prob.contest_id,
                'contest_date': prob.contest_date.isoformat(),
                'difficulty': prob.difficulty.value,
                'starter_code': prob.starter_code,
                'output_list': output_list,
                'code_list': code_list,
            })
            eval_all_records.append(
                prob.insert_output_evaluation(
                    output_list=output_list,
                    code_list=code_list,
                    graded_list=graded,
                    metadata=md,
                ))

        self._write_json(out_dir / f'{prefix}.json', raw_records)
        self._write_json(out_dir / f'{prefix}_eval.json', metrics)
        self._write_json(out_dir / f'{prefix}_eval_all.json',
                         eval_all_records)

    @staticmethod
    def _write_json(path, obj):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
