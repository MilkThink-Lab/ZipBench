import contextlib
import io
import itertools
import json
import multiprocessing
import os
import os.path as osp
import signal
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import environ
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import regex as re
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MBPPDataset(BaseDataset):

    @staticmethod
    def load(path: str, local_mode: bool = False):
        path = get_data_path(path, local_mode=local_mode)

        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_list_2'] = example['test_list']
            return example

        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            train = MsDataset.load(path,
                                   subset_name='full',
                                   split='train[:10]').map(processing_test)
            test = MsDataset.load(path,
                                  subset_name='full',
                                  split='train[10:510]').map(processing_test)
        else:
            train = load_dataset('json', data_files=path,
                                 split='train[:10]').map(processing_test)
            test = load_dataset('json', data_files=path,
                                split='train[10:510]').map(processing_test)
        return DatasetDict({'train': train, 'test': test})


class MBPPDatasetV2(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1):
        """Load mbpp dataset for pass k mode.

        Note that you can use num_repeats > 1 when your model does not support
        `num_return_sequence` in generation, otherwise use the raw
        mbpp dataset and set `num_return_sequence` in model config to
        generate multiple responses for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """

        path = get_data_path(path)

        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_column'] = dict(test_list_2=example['test_list'],
                                          task_id=example['task_id'])
            return example

        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            train = MsDataset.load(path,
                                   subset_name='full',
                                   split='train[:10]').map(processing_test)
            test = MsDataset.load(path,
                                  subset_name='full',
                                  split='train[10:510]').map(processing_test)
        else:
            train = load_dataset('json', data_files=path,
                                 split='train[:10]').map(processing_test)
            test = load_dataset('json', data_files=path,
                                split='train[10:510]').map(processing_test)
        test = concatenate_datasets([test] * num_repeats)
        return DatasetDict({'train': train, 'test': test})


class SanitizedMBPPDataset(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1):
        """Load mbpp dataset for pass k mode.

        Note that you can use num_repeats > 1 when your model does not support
        `num_return_sequence` in generation, otherwise use the raw
        mbpp dataset and set `num_return_sequence` in model config to
        generate multiple responses for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """
        path = get_data_path(path)

        def processing_test(example):
            example['text'] = example.pop('prompt')
            # used for prompt
            example['test_list'] = '\n'.join(example['test_list'])
            # used for eval
            example['test_list_2'] = example['test_list']
            example['test_column'] = dict(test_list_2=example['test_list'],
                                          task_id=example['task_id'])
            return example

        # train : test = 7 : 257
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            train = MsDataset.load(path,
                                   subset_name='sanitized',
                                   split='train[:7]').map(processing_test)
            test = MsDataset.load(path,
                                  subset_name='sanitized',
                                  split='train[7:264]').map(processing_test)
        else:
            train = load_dataset('json', data_files=path,
                                 split='train[:7]').map(processing_test)
            test = load_dataset('json', data_files=path,
                                split='train[7:264]').map(processing_test)
        test = concatenate_datasets([test] * num_repeats)
        return DatasetDict({'train': train, 'test': test})


class MBPPPlusDataset(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1):
        """Load mbpp dataset for pass k mode. Note that you can use
        num_repeats.

        > 1 when your model does not support `num_return_sequence` in
        generation, otherwise use the raw mbpp dataset and set
        `num_return_sequence` in model config to generate multiple responses
        for testing pass@k>1.

        It better to change your dataset abbr correspondingly if you want to
        change num_repeats>1, otherwise the number in
        `.cache/dataset_size.json` might be inconsistent.

        Args:
            num_repeats(int): Number of repetition for this dataset to get
        multiple responses in special cases.
        """

        path = get_data_path(path)

        def processing_test(example):
            example['test_case'] = example['test_list']
            example['test_list'] = '\n'.join(example['test_list'])
            example['test_list_2'] = example['test_list']
            example['test_column'] = dict(test_list_2=example['test_list'],
                                          task_id=example['task_id'])
            return example

        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                example = processing_test(example)
                dataset.extend([example for _ in range(num_repeats)])
        return Dataset.from_list(dataset)


class MBPPPlusOfficialDataset(BaseDataset):
    """MBPP+ loaded from the official EvalPlus release jsonl.

    Unlike ``MBPPPlusDataset`` (original MBPP text + original asserts), this
    exposes the official ``prompt`` field (docstring + fixed assertion), so
    the assertion shown to the model matches the semantics the EvalPlus
    harness actually tests.
    """

    @staticmethod
    def load(path: str, num_repeats: int = 1):
        path = get_data_path(path)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                # keep only flat columns; nested input fields have
                # heterogeneous types that break Arrow inference
                example = {
                    'task_id': example['task_id'],
                    'prompt': example['prompt'].strip(),
                    'assertion': example.get('assertion', ''),
                }
                dataset.extend([example for _ in range(num_repeats)])
        return Dataset.from_list(dataset)


class TimeOutException(Exception):
    pass


def _sorted_task_ids(task_ids: Iterable[str]) -> List[str]:
    def sort_key(task_id: str) -> Tuple[str, int]:
        if "/" in task_id:
            prefix, suffix = task_id.rsplit("/", 1)
            if suffix.isdigit():
                return (prefix, int(suffix))
        return (task_id, -1)

    return sorted(task_ids, key=sort_key)


def _export_mbpp_plus_per_item(eval_results_path: str,
                               output_jsonl: str) -> None:
    with open(eval_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    items = []
    for task_id in _sorted_task_ids(results["eval"].keys()):
        task_res = results["eval"][task_id]
        base_list = task_res.get("base", [])
        plus_list = task_res.get("plus", [])

        def _is_success(idx: int) -> bool:
            if idx >= len(base_list):
                return False
            base_ok = base_list[idx][0] == "success"
            if not plus_list:
                return base_ok
            if idx >= len(plus_list):
                return False
            return base_ok and (plus_list[idx][0] == "success")

        any_correct = any(_is_success(i) for i in range(task_res["nfiles"]))
        first_base = base_list[0] if base_list else None
        first_plus = plus_list[0] if plus_list else None

        items.append({
            "task_id": task_id,
            "any_correct": any_correct,
            "base_status_first": first_base[0] if first_base else None,
            "plus_status_first": first_plus[0] if first_plus else None,
            "base_details_first": first_base[1] if first_base else None,
            "plus_details_first": first_plus[1] if first_plus else None,
        })

    os.makedirs(osp.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_mbpp_plus_any_correct(per_item_path: str) -> dict:
    any_correct_map = {}
    with open(per_item_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            if not task_id:
                continue
            any_correct_map[task_id] = bool(record.get("any_correct"))
    return any_correct_map


def _normalize_reference(reference) -> Optional[str]:
    if isinstance(reference, str):
        return reference
    if isinstance(reference, list) and len(reference) == 1:
        if isinstance(reference[0], str):
            return reference[0]
    return None


def _inject_mbpp_plus_any_correct(details: dict,
                                  any_correct_map: dict) -> None:
    for entry in details.values():
        if not isinstance(entry, dict):
            continue
        reference = _normalize_reference(entry.get("references"))
        if not reference:
            continue
        if reference in any_correct_map:
            entry["any_correct"] = any_correct_map[reference]

def _strip_after_think(text: str) -> str:
    """Only keep content after </think> if present."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):

    def signal_handler(signum, frame):
        raise TimeOutException('Time out!')

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@ICL_EVALUATORS.register_module()
class MBPPEvaluator(BaseEvaluator):
    """Evaluator for MBPP or MBPPPlus."""

    def __init__(self,
                 metric: str = 'MBPP',
                 export_mbpp_plus_details: bool = True,
                 mbpp_plus_details_dir: Optional[str] = None) -> None:
        self.metric = metric
        self.export_mbpp_plus_details = export_mbpp_plus_details
        self.mbpp_plus_details_dir = mbpp_plus_details_dir
        assert self.metric in ['MBPP', 'MBPPPlus']

    def _resolve_mbpp_plus_details_dir(self) -> Optional[str]:
        if not self.export_mbpp_plus_details:
            return None
        if self.mbpp_plus_details_dir:
            base_dir = self.mbpp_plus_details_dir
        else:
            base_dir = getattr(self, '_out_dir', None)
            if base_dir:
                base_dir = f'{base_dir}_details'
        if not base_dir:
            return None
        replica_idx = getattr(self, '_dataset_replica_idx', 0)
        if replica_idx:
            return f'{base_dir}_replica{replica_idx}'
        return base_dir

    def _run_evalplus(self, samples_path: str, mbpp_preds: List[dict],
                      test_details: Union[bool, float]):
        self.write_jsonl(samples_path, mbpp_preds)
        flags = dict(dataset='mbpp',
                     samples=samples_path,
                     base_only=None,
                     parallel=None,
                     i_just_wanna_run=None,
                     test_details=test_details,
                     min_time_limit=0.2,
                     gt_time_limit_factor=4.0,
                     mini=None)
        return self.eval(flags)

    def _maybe_export_mbpp_plus_details(self, samples_path: str,
                                        details_dir: str) -> None:
        eval_results_path = samples_path.replace(".jsonl",
                                                 "_eval_results.json")
        if osp.exists(eval_results_path):
            per_item_path = osp.join(details_dir, "mbpp_plus_per_item.jsonl")
            _export_mbpp_plus_per_item(eval_results_path, per_item_path)

    def postprocess_results(self, result: dict) -> None:
        if self.metric != "MBPPPlus":
            return
        details = result.get("details")
        if not isinstance(details, dict):
            return
        details_dir = self._resolve_mbpp_plus_details_dir()
        if not details_dir:
            return
        per_item_path = osp.join(details_dir, "mbpp_plus_per_item.jsonl")
        if not osp.exists(per_item_path):
            return
        any_correct_map = _load_mbpp_plus_any_correct(per_item_path)
        if not any_correct_map:
            return
        _inject_mbpp_plus_any_correct(details, any_correct_map)

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        if self.metric == 'MBPP':
            result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
            details = {}
            with ProcessPoolExecutor() as executor:
                futures = []
                for i, (refer, pred) in enumerate(zip(references,
                                                      predictions)):
                    pred = self._process_answer(pred)
                    programs = self._process_test(refer, pred)
                    future = executor.submit(execution, programs, i, 10)
                    futures.append(future)
                    details[str(i)] = {}
                    details[str(i)]['origin'] = predictions[i]
                    details[str(i)]['programs'] = programs

                from tqdm import tqdm
                for future in tqdm(as_completed(futures), total=len(futures)):
                    index, ret = future.result()
                    result[ret] += 1
                    details[str(index)]['result'] = ret
                    details[str(index)]['is_correct'] = (ret == 'pass')

            result['score'] = result['pass'] / len(predictions) * 100
            result['details'] = details
            return result
        else:
            try:
                from evalplus.data import write_jsonl
                from evalplus.evaluate import evaluate
                self.write_jsonl = write_jsonl
                self.eval = evaluate
            except ImportError:
                raise ImportError(
                    'Please install evalplus use following steps:\n'
                    'git clone --recurse-submodules git@github.com:open-compass/human-eval.git\n'  # noqa
                    'cd human-eval\n'
                    'pip install -e .\n'
                    'pip install -e evalplus\n')
            mbpp_preds = []
            for preds, refer in zip(predictions, references):
                if not isinstance(preds, list):
                    preds = [preds]
                for pred in preds:
                    pred = self._process_answer(pred)
                    mbpp_preds.append({'task_id': refer, 'solution': pred})
            details_dir = self._resolve_mbpp_plus_details_dir()
            if details_dir:
                os.makedirs(details_dir, exist_ok=True)
                samples_path = osp.join(details_dir, "mbpp_plus_samples.jsonl")
                score = self._run_evalplus(samples_path, mbpp_preds, True)
                self._maybe_export_mbpp_plus_details(samples_path, details_dir)
            else:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    samples_path = osp.join(tmp_dir, "mbpp_eval.jsonl")
                    score = self._run_evalplus(samples_path, mbpp_preds, 0.2)
            return {f'mbpp_plus_{k}': score[k] * 100 for k in score}

    def _process_answer(self, text):
        text = _strip_after_think(text)
        patterns = [
            r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
            r"\[BEGIN\]\s*```(.*)```\s*\[DONE\]",
            r"BEGIN\s*'(.*)'\s*\[DONE\]",
            r"\[BEGIN\]\s*'(.*)'\s*DONE",
            r"BEGIN\s*'(.*)'\s*DONE",
            r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
            r"BEGIN\s*'(.*)\s*\[DONE\]",
            r"\[BEGIN\]\s*'(.*)\s*DONE",
            r"BEGIN\s*'(.*)\s*DONE",
            r'\[BEGIN\]\s*(.*)\s*\[DONE\]',
            r'BEGIN\s*(.*)\s*\[DONE\]',
            r'\[BEGIN\]\s*(.*)\s*DONE',
            r'BEGIN\s*(.*)\s*DONE',
            r'```python\s*(.*)\s*```',
            r'```\s*(.*)\s*```',
            r'```python\s*(.*)\s*$',
            r'```\s*(.*)\s*$',
            r'(.*)\s*```.*',
            r"\[BEGIN\]\s*'(.*)",
            r'\[BEGIN\](.*)',
            r"'(.*)'\s*\[DONE\]",
        ]
        for p in patterns:
            try:
                match = re.search(p, text, re.DOTALL, timeout=10.0)
            except TimeoutError:
                match = None

            if match:
                text = match.group(1)
                break
        text = text.split('```')[0]
        text = re.split(r"'?\s*\[?DONE\]?", text)[0]
        text = text.replace('\\_', '_')
        text = text.lstrip("ল")
        text = text.lstrip("python")
        text = text.replace("\\n", "\n")
        text = text.strip()
        return text

    def _process_test(self, test_case, pred):
        formatted = pred + '\n'
        formatted += test_case
        return formatted


@ICL_EVALUATORS.register_module()
class MBPPEvaluator2(MBPPEvaluator):
    """Better use for WizardCoder evaluation."""

    def _process_answer(self, text):
        text = _strip_after_think(text)
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                if not text.startswith(
                        '\n'):  # in case starting with ```python
                    text = text[max(text.find('\n') + 1, 0):]
        else:
            match = re.search(r'Here(.*?)\n', text)
            if match:
                text = re.sub('Here(.*?)\n', '', text, count=1)

        # remove test in generation
        test_list = ['# Test', '#Test', '#test', '# test']
        for s in test_list:
            if s in text:
                text = text[:text.find(s)]

        text = text.strip()
        match = re.search(r"('\s*|)(\[DONE\]|DONE)", text)
        if match:
            text = text[:match.start()]
        match = re.search(r"(\[BEGIN\]|BEGIN)('\s*|)", text)
        if match:
            text = text[match.end():]
        text = text.strip()
        if text.startswith("'"):
            text = text[1:]
        return text


def _execution(programs, timeout, key):
    try:
        # Add exec globals to prevent the exec to raise
        # unnecessary NameError for correct answer
        exec_globals = {}
        with swallow_io():
            with time_limit(timeout):
                exec(programs, exec_globals)
        key.append('pass')
    except TimeOutException:
        key.append('timeout')
    except AssertionError:
        key.append('wrong_answer')
    except BaseException as e:
        print(e)
        key.append('failed')


def execution(programs, task_id, timeout):
    """Execution function for running generation code.

    Args:
        programs(str): Python code to be executed.
        task_id(int): Task id of the current example.
        timeout(int): Time limit for execution, avoid unnecessary
            blocking.

    In pass@k scenario, a lot of programs should be executed.
    Some internal error cannot be handled properly, such as
    `RecursionError` might cause system break. It is better to
    separate the execution in thread or multiprocess to better
    control the process.
    """

    manager = multiprocessing.Manager()
    key = manager.list()
    # `signal` cannot be used in child thread, therefore, we
    # need to create a process in the thread.
    p = multiprocessing.Process(target=_execution,
                                args=(programs, timeout - 1, key))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        # key might not have value if killed
        return task_id, 'timeout'
    return task_id, key[0]


class MBPPPassKEvaluator(MBPPEvaluator):
    """Better use for pass k evaluation.

    Args:
        k(Tuple[int]): Choices of Pass@k. Defaults to (1, 10, 100)
    """

    def __init__(self, k=(1, 10, 100)) -> None:
        if not isinstance(k, Sequence):
            k = (k, )
        self.k = k

    @staticmethod
    def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int,
    ) -> np.ndarray:
        """Estimates pass@k of each problem and returns them in an array."""

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array([
            estimator(int(n), int(c), k)
            for n, c in zip(num_samples_it, num_correct)
        ])

    def score(self, predictions, references):
        assert len(predictions) == len(references)

        task_pass = defaultdict(int)
        task_total = defaultdict(int)

        result = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
        with ProcessPoolExecutor() as executor:
            futures = []
            for refer, preds in zip(references, predictions):
                # suits for two case
                # 1. use repeated dataset
                # 2. use `num_return_sequences` to generate multiple responses
                if not isinstance(preds, list):
                    preds = [preds]
                test_case = refer['test_list_2']
                task_id = refer['task_id']
                # create empty task_pass in case all example failed
                if task_id not in task_pass:
                    task_pass[task_id] = 0
                for pred in preds:
                    pred = self._process_answer(pred)
                    programs = self._process_test(test_case, pred)
                    future = executor.submit(execution, programs, task_id, 10)
                    futures.append(future)

            from tqdm import tqdm
            for future in tqdm(as_completed(futures), total=len(futures)):
                task_id, key = future.result()
                result[key] += 1
                task_total[task_id] += 1
                if key == 'pass':
                    task_pass[task_id] += 1

        def get_number(tasks):
            return np.array([
                task[1] for task in sorted(tasks.items(), key=lambda x: x[0])
            ])

        task_pass = get_number(task_pass)
        task_total = get_number(task_total)
        pass_at_k = {
            f'pass@{k}':
            self.estimate_pass_at_k(task_total, task_pass, k).mean() * 100
            for k in self.k if (task_total >= k).all()
        }
        result.update(pass_at_k)
        return result
