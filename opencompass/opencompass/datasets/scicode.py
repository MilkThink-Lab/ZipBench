import ast
import concurrent.futures
import json
import os
import os.path as osp
import re
import subprocess
import sys

import h5py
import numpy as np
import scipy
import scipy.sparse
import sympy
from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

IMPORT_LINE_PATTERN = re.compile(
    r'^\s*(import .*|from .*\s+import\s+.*)', re.MULTILINE)
PYTHON_FENCE_PATTERN = re.compile(r'```python\s*\n?(.*?)```',
                                  re.DOTALL | re.IGNORECASE)
GENERIC_FENCE_PATTERN = re.compile(r'```\s*\n?(.*?)```', re.DOTALL)
TOP_LEVEL_SYMBOL_PATTERN = re.compile(
    r'(?m)^(def|class)\s+([A-Za-z_]\w*)\s*(?:\(|:)')
SCICODE_PREPROVIDED_CODE = {
    ('13', 5): '13.6.txt',
    ('62', 0): '62.1.txt',
    ('76', 2): '76.3.txt',
}


@LOAD_DATASET.register_module()
class SciCodeDataset(BaseDataset):

    @staticmethod
    def load(path, with_bg, dataset_filename=None, **kwargs):
        test_data = []
        path = get_data_path(path, local_mode=True)
        if dataset_filename:
            file_path = osp.join(path, dataset_filename)
        elif with_bg:  # test with background
            file_path = osp.join(path, 'SciCode_datasets_with_background.json')
        else:  # test w/o background
            file_path = osp.join(path, 'SciCode_datasets.json')

        with open(file_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)

        dataset = Dataset.from_list(test_data)
        return dataset

    def return_dataset(self):
        return self.dataset


def strip_import_lines(code: str) -> str:
    return re.sub(IMPORT_LINE_PATTERN, '', code).strip()


def trim_to_parsable_suffix_safe(code: str) -> str:
    code = code.strip()
    if not code:
        return ''

    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    lines = code.splitlines()
    for end in range(len(lines) - 1, 0, -1):
        snippet = '\n'.join(lines[:end]).strip()
        if not snippet:
            continue
        try:
            ast.parse(snippet)
            return snippet
        except SyntaxError:
            continue
    return ''


def extract_candidate_code(response: str) -> str:
    if '</think>' in response:
        response = response.split('</think>')[-1]

    match = PYTHON_FENCE_PATTERN.search(response)
    if match is None:
        match = GENERIC_FENCE_PATTERN.search(response)

    candidate = match.group(1) if match is not None else response
    candidate = strip_import_lines(candidate)
    if not candidate:
        return ''

    if match is None and not trim_to_parsable_suffix_safe(candidate):
        return ''
    return candidate


def extract_named_definition(code: str, symbol_name: str) -> str:
    if not code:
        return ''

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ''

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.name == symbol_name:
            segment = ast.get_source_segment(code, node)
            if segment:
                return segment
            return ast.unparse(node)
    return ''


def extract_scicode_symbol_from_prompt(prompt: str):
    matches = list(TOP_LEVEL_SYMBOL_PATTERN.finditer(prompt))
    if not matches:
        raise ValueError('Failed to parse SciCode step symbol from prompt')
    kind, symbol_name = matches[-1].groups()
    return kind, symbol_name


def append_code_chunk(python_code: str, code_chunk: str) -> str:
    if not code_chunk:
        return python_code
    if python_code and not python_code.endswith('\n'):
        python_code += '\n'
    return python_code + code_chunk.rstrip() + '\n'


def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            dict[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group):
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr),
                                       shape=shape,
                                       blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == 'list':
            return process_hdf5_list(group[key])
        if key == 'sparse_matrix':
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num, h5py_file_folder=None):
    if h5py_file_folder is None:
        h5py_file_folder = './data/scicode/test_data'
    data_lst = []
    h5py_file = os.path.join(h5py_file_folder, f'{step_id}.h5')
    assert os.path.exists(
        h5py_file
    ), (f"Please manually download 'test_data.h5' from "
        f"https://github.com/open-compass/storage/releases/download/"
        f"v0.1.0/scicode_test_data.zip and put the file in {h5py_file}")

    with h5py.File(h5py_file, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode(
                                'utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode(
                                    'utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(
                                process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(
                    f'Path {group_path} not found in the file.')
    return data_lst


def are_dicts_close(dict1, dict2, atol=1e-8, rtol=1e-5):
    dict1 = process_symbol_in_dict(dict1)
    dict2 = process_symbol_in_dict(dict2)
    # Check if both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        return False

    # Check if the corresponding values are close
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, (sympy.Symbol, str)):
            if not value1 == value2:
                return False
        elif isinstance(value1,
                        (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix,
                         scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix)):
            value1 = value1.toarray()
            value2 = value2.toarray()
            if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                return False
        # Use np.allclose to compare values
        else:
            try:
                if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                    return False
            except ValueError:
                if not value1 == value2:
                    return False

    return True


def process_symbol_in_dict(dict):
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value
        if isinstance(value, sympy.Symbol):
            new_dict[key] = str(value)
        if isinstance(key, sympy.Symbol):
            new_dict[str(key)] = dict[key]
            new_dict.pop(key)
    return new_dict


def are_csc_matrix_close(matrix1, matrix2):
    dense1 = matrix1.toarray()
    dense2 = matrix2.toarray()
    return np.allclose(dense1, dense2)


def cmp_tuple_or_list(var1, var2):
    if len(var1) != len(var2):
        return False
    for v1, v2 in zip(var1, var2):
        if isinstance(v1, dict):
            if not are_dicts_close(v1, v2):
                return False
        elif isinstance(v1,
                        (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            if not are_csc_matrix_close(v1, v2):
                return False
        elif isinstance(v1, bool):
            if not v1 == v2:
                return False
        else:
            try:
                if not np.allclose(v1, v2):
                    return False
            except ValueError as e:
                print(e)
                if not v1 == v2:
                    return False
    return True


@ICL_EVALUATORS.register_module()
class SciCodeEvaluator(BaseEvaluator):

    PREPROVIDED_CODE = SCICODE_PREPROVIDED_CODE

    def __init__(self, dataset_path, with_bg, dataset_filename=None):
        super().__init__()
        test_data = []
        dataset_path = get_data_path(dataset_path, local_mode=True)
        if dataset_filename:
            file_path = osp.join(dataset_path, dataset_filename)
        elif with_bg:  # test with background
            file_path = osp.join(dataset_path,
                                 'SciCode_datasets_with_background.json')
        else:  # test w/o background
            file_path = osp.join(dataset_path, 'SciCode_datasets.json')
        with open(file_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)
        self.dataset = Dataset.from_list(test_data)
        self.eval_data_path = get_data_path('./data/scicode/eval_data',
                                            local_mode=True)
        for filename in self.PREPROVIDED_CODE.values():
            code_file = osp.join(self.eval_data_path, filename)
            if not osp.exists(code_file):
                raise FileNotFoundError(
                    f'Expected SciCode eval data file at {code_file}')

    def extract_python_script(self, response: str):
        return trim_to_parsable_suffix_safe(extract_candidate_code(response))

    def run_script(self, script_path):
        try:
            subprocess.run([sys.executable, script_path],
                           check=True,
                           capture_output=True,
                           text=True,
                           timeout=1800)
            return 0
        except subprocess.CalledProcessError:
            return 1
        except subprocess.TimeoutExpired:
            return 2

    def score(self, predictions, references):
        # generate all python test codes
        os.makedirs(self._out_dir, exist_ok=True)
        ordered_sub_ids = []
        for idx, prediction_list in enumerate(predictions):
            # traverse each test sample
            problem_id = self.dataset[idx]['id']
            prompt_list = self.dataset[idx]['prompt']
            num_tests = len(self.dataset[idx]['test'])
            expected_predictions = sum(
                1 for test_idx in range(num_tests)
                if (problem_id, test_idx) not in self.PREPROVIDED_CODE)

            if len(prompt_list) != expected_predictions:
                raise ValueError(
                    f'Problem {problem_id} prompt count mismatch: '
                    f'{len(prompt_list)} prompts for {expected_predictions} '
                    'generated steps')
            if len(prediction_list) != expected_predictions:
                raise ValueError(
                    f'Problem {problem_id} prediction count mismatch: '
                    f'{len(prediction_list)} predictions for '
                    f'{expected_predictions} generated steps')

            # create dir for each test sample
            testdir_path = os.path.join(self._out_dir, str(problem_id))
            os.makedirs(testdir_path, exist_ok=True)

            python_code = ''
            # add import statement
            python_code += self.dataset[idx]['import']

            pred_idx = 0  # index into prediction_list
            for test_idx in range(num_tests):
                preprovided_key = (problem_id, test_idx)
                test_lst = self.dataset[idx]['test'][test_idx]

                if preprovided_key in self.PREPROVIDED_CODE:
                    if test_lst:
                        raise ValueError(
                            f'Problem {problem_id} step {test_idx + 1} '
                            'should be a preprovided step with empty tests')
                    # Load preprovided code instead of LLM generation
                    code_file = os.path.join(
                        self.eval_data_path,
                        self.PREPROVIDED_CODE[preprovided_key])
                    with open(code_file, 'r', encoding='utf-8') as f:
                        python_code = append_code_chunk(python_code,
                                                        f.read())
                    continue
                if not test_lst:
                    raise ValueError(
                        f'Problem {problem_id} step {test_idx + 1} has empty '
                        'tests but is not marked as preprovided')

                response = prediction_list[pred_idx]
                python_code = append_code_chunk(
                    python_code, self.extract_python_script(response))
                pred_idx += 1

                step_id = f'{problem_id}.{test_idx + 1}'
                sub_id = f'{problem_id}-{test_idx + 1}'
                testfile_path = os.path.join(testdir_path,
                                             f'{sub_id}.py')
                # track the sub-problem ID in dataset order
                ordered_sub_ids.append(sub_id)
                # write python code and test cases to a real python file
                with open(testfile_path, 'w', encoding='utf-8') as f:
                    f.write(python_code)
                    f.write(
                        '\nfrom opencompass.datasets.scicode '
                        'import process_hdf5_to_tuple\n')
                    f.write(f"targets = process_hdf5_to_tuple("
                            f"'{step_id}', {len(test_lst)})\n")
                    for idx2 in range(len(test_lst)):
                        f.write(f'target = targets[{idx2}]\n\n')
                        for line in test_lst[idx2].split('\n'):
                            f.write(line + '\n')
            if pred_idx != len(prediction_list):
                raise ValueError(
                    f'Problem {problem_id} consumed {pred_idx} predictions '
                    f'but received {len(prediction_list)}')

        # find all scripts
        python_scripts = []
        for root, dirs, files in os.walk(self._out_dir):
            for file in files:
                if file.endswith('.py'):
                    python_scripts.append(os.path.join(root, file))
        python_scripts.sort()

        # Use ThreadPoolExecutor to concurrently execute scripts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.run_script, script)
                for script in python_scripts
            ]

        # Collect results in submission order to match python_scripts
        results = [future.result() for future in futures]

        # Build a lookup: sub_id -> 1 (correct) / 0 (incorrect)
        sub_results_lookup = {}
        all_results = {}
        for script_path, result in zip(python_scripts, results):
            basename = os.path.basename(script_path)
            main_id = basename.split('-')[0]
            sub_id = basename.replace('.py', '')  # e.g. "56-3"
            if all_results.get(main_id):
                all_results[main_id].append(result)
            else:
                all_results[main_id] = [result]
            # run_script returns: 0=pass, 1=fail, 2=timeout
            # convert to: 1=correct, 0=incorrect
            sub_results_lookup[sub_id] = 1 if result == 0 else 0

        correct, sub_correct = 0, 0
        count, sub_count = 0, 0

        for main_id in all_results:
            correct += sum(all_results[main_id]) == 0
            count += 1
            for sub in all_results[main_id]:
                sub_correct += sub == 0
                sub_count += 1

        # Build ordered detail following original dataset order
        ordered_detail = {
            sid: sub_results_lookup[sid] for sid in ordered_sub_ids
        }

        detail_name = ('sub_results_detail.json'
                       if self.dataset_replica_idx == 0 else
                       f'sub_results_detail_replica{self.dataset_replica_idx}.json')
        detail_path = os.path.join(self._out_dir, detail_name)
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_detail, f, indent=2, ensure_ascii=False)

        result = {
            'accuracy': 100 * correct / count,
            'sub_accuracy': 100 * sub_correct / sub_count,
        }
        return result
