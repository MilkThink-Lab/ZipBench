"""ZipBench adapter for SciCode (sub-step selection + code-execution scoring).

SciCode breaks ZipBench's default "one dataset row == one selection unit == one
exact-match score" assumption on every axis:

  * a *row* is a multi-step problem; the model generates code step-by-step, and
    each step sees the previous steps' *extracted* code as chat history
    (``SciCodeChatInferencer``), so steps are a dependent chain -- you can drop
    trailing steps but not interior ones;
  * the anchor selects *sub-steps* -- the 288 generated steps across the 65
    problems, in dataset order, skipping the 3 pre-provided steps -- so an
    anchor ``index`` is sub-step granular and the stable id is the sub-step id
    ``'<problem>-<step>'`` (the same id SciCode's ``sub_results_detail.json``
    uses);
  * scoring runs the assembled program against hidden numeric targets, not
    ``pred == ref``.

This module supplies the three benchmark-specific pieces the generic core
cannot:

  * :func:`enumerate_sub_ids` -- the canonical sub-step enumeration (used by the
    spec converter and to validate a spec against the live dataset);
  * :class:`ZipSciCodeDataset` -- loads the full dataset, then keeps only the
    problems *touched* by the subset (optionally truncating each to its last
    selected step), so untouched problems are never inferred;
  * :class:`WeightedSciCodeEvaluator` -- subclasses the upstream evaluator,
    writes/runs only the selected sub-steps and reports a *weighted* sub-step
    accuracy.

Everything is wired through the manifest (``--units-from scicode``); no core
ZipBench code is SciCode-aware.
"""
import concurrent.futures
import json
import os
import os.path as osp

from datasets import Dataset

# Importing the upstream SciCode module initialises ``opencompass.datasets``
# (and pulls h5py/scipy/sympy); doing it first avoids the openicl<->datasets
# circular import, exactly as the ZipBench core modules do.
from opencompass.datasets.scicode import (SCICODE_PREPROVIDED_CODE,
                                          SciCodeEvaluator, append_code_chunk,
                                          process_hdf5_to_tuple)
from opencompass.datasets.base import BaseDataset
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import get_data_path

from zipbench.result import ZipResultShapeMixin, zip_result

from ..spec import load_subset_spec


# ----------------------------------------------------------------------------
# Shared helpers (used by the converter, the loader and the evaluator so the
# sub-step enumeration / subset construction can never drift between them).
# ----------------------------------------------------------------------------
def _resolve_dataset_file(path, with_bg, dataset_filename=None):
    """Mirror ``SciCodeDataset.load``'s file resolution."""
    path = get_data_path(path, local_mode=True)
    if dataset_filename:
        return osp.join(path, dataset_filename)
    if with_bg:
        return osp.join(path, 'SciCode_datasets_with_background.json')
    return osp.join(path, 'SciCode_datasets.json')


def load_scicode_test_data(path, with_bg, dataset_filename=None):
    """Return the raw SciCode problem list (mirrors ``SciCodeDataset.load``)."""
    with open(_resolve_dataset_file(path, with_bg, dataset_filename),
              'r', encoding='utf-8') as f:
        return json.load(f)


def enumerate_sub_ids(test_data):
    """Canonical sub-step ids in dataset order, skipping pre-provided steps.

    Each generated step of each problem yields ``'<problem_id>-<step_no>'``
    (1-based ``step_no``); pre-provided steps (``SCICODE_PREPROVIDED_CODE``) are
    skipped because the model never generates them. The result is index-aligned
    with the *spec*: ``enumerate_sub_ids(...)[i]`` is the sub-step a spec row
    refers to as ``index == i``. (The internal anchor pkl uses a different item
    order -- sub_ids sorted lexicographically; the converter translates that to
    these natural-order indices, see ``convert_anchor_to_spec.py``.)
    ``test_data`` may be a raw list or a ``datasets.Dataset`` (both iterate to
    per-problem dicts).
    """
    sub_ids = []
    for rec in test_data:
        pid = str(rec['id'])
        for test_idx in range(len(rec['test'])):
            if (pid, test_idx) in SCICODE_PREPROVIDED_CODE:
                continue
            sub_ids.append(f'{pid}-{test_idx + 1}')
    return sub_ids


def load_selected_sub_ids_and_weights(subset_spec, test_data):
    """Load a spec, validating each row against the live sub-step enumeration.

    Returns ``(selected_set, weight_by_sub_id)``. Raises if a spec row's stable
    id no longer matches the sub-step at its ``index`` (dataset drift) -- the
    same guarantee ``zipbench.spec.validate_and_select`` gives the row path.
    """
    spec = load_subset_spec(subset_spec)
    all_sub_ids = enumerate_sub_ids(test_data)
    n_total = len(all_sub_ids)
    selected, weights, mismatches = set(), {}, []
    for r in spec:
        idx = r['index']
        if idx < 0 or idx >= n_total:
            raise ValueError(
                f'subset index out of range: {idx} not within [0, {n_total}) '
                '-- SciCode dataset may have changed since the spec was made')
        actual = all_sub_ids[idx]
        if r['id'] is not None and str(r['id']) != actual:
            mismatches.append((idx, r['id'], actual))
        selected.add(actual)
        weights[actual] = r['weight']
    if mismatches:
        raise ValueError(
            f'subset id mismatch on {len(mismatches)} sub-step(s); the SciCode '
            'dataset no longer matches the spec. First (index, expected, '
            f'actual): {mismatches[:5]}')
    return selected, weights


def build_scicode_subset(test_data, selected_sub_ids, truncate=False):
    """Keep only the problems touched by ``selected_sub_ids`` (dataset order).

    ``truncate=False`` (default): each kept problem keeps its full ``prompt`` /
    ``test`` lists, so the selected steps' predictions are byte-identical to a
    full run -- only untouched problems are dropped.

    ``truncate=True``: each kept problem is cut to its last selected step
    (trailing steps removed). Safe because a step only depends on *earlier*
    steps' code, so the selected steps still see identical chat history; this
    additionally saves the trailing generations.
    """
    rows = []
    for rec in test_data:
        pid = str(rec['id'])
        sel_steps = [t for t in range(len(rec['test']))
                     if (pid, t) not in SCICODE_PREPROVIDED_CODE
                     and f'{pid}-{t + 1}' in selected_sub_ids]
        if not sel_steps:
            continue
        if not truncate:
            rows.append(dict(rec))
            continue
        cut = max(sel_steps)
        n_gen_kept = sum(1 for t in range(cut + 1)
                         if (pid, t) not in SCICODE_PREPROVIDED_CODE)
        row = dict(rec)
        row['test'] = list(rec['test'])[:cut + 1]
        row['prompt'] = list(rec['prompt'])[:n_gen_kept]
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Dataset loader
# ----------------------------------------------------------------------------
@LOAD_DATASET.register_module()
class ZipSciCodeDataset(BaseDataset):
    """Load SciCode, then keep only the problems a subset spec touches.

    Same ctor surface as ``SciCodeDataset`` (``path`` / ``with_bg`` /
    ``dataset_filename``) plus ``subset_spec`` (and an optional ``truncate``);
    returns a flat ``Dataset`` of the kept problems. ``subset_spec=None``
    reproduces the full dataset.
    """

    @staticmethod
    def load(path, with_bg, subset_spec=None, dataset_filename=None,
             truncate=False, **kwargs):
        test_data = load_scicode_test_data(path, with_bg, dataset_filename)
        if subset_spec is None:
            return Dataset.from_list(test_data)
        selected, _ = load_selected_sub_ids_and_weights(subset_spec, test_data)
        rows = build_scicode_subset(test_data, selected, truncate=truncate)
        return Dataset.from_list(rows)


# ----------------------------------------------------------------------------
# Weighted evaluator
# ----------------------------------------------------------------------------
@ICL_EVALUATORS.register_module()
class WeightedSciCodeEvaluator(ZipResultShapeMixin, SciCodeEvaluator):
    """Score only the subset's sub-steps; report weighted sub-step accuracy.

    Reuses the upstream code-assembly / execution machinery but, for each
    problem in the (already subset-filtered) ``test_set``, writes and runs a
    test file *only* for the selected sub-steps -- while still accumulating
    every generated step's code, since later steps depend on earlier ones.

    Headline metric ``weighted_sub_accuracy = sum(w_i * correct_i) * 100`` over
    the selected sub-steps (anchor weights are normalised to sum to 1). The
    problem-level ``accuracy`` is reported only over *fully covered* problems
    (every generated step selected), since a sub-step subset cannot faithfully
    reconstruct the standard SciCode main accuracy.
    """

    def __init__(self, dataset_path, with_bg, subset_spec,
                 breakdown_fields=None, dataset_filename=None):
        super().__init__(dataset_path=dataset_path, with_bg=with_bg,
                         dataset_filename=dataset_filename)
        # ``self.dataset`` (full problem list) is set by the parent; reuse it to
        # resolve + validate the selected sub-steps and their weights.
        self.selected_sub_ids, self.sub_weights = \
            load_selected_sub_ids_and_weights(subset_spec, self.dataset)
        self.breakdown_fields = list(breakdown_fields or [])

    def score(self, predictions, references, test_set):
        if len(predictions) != len(test_set):
            raise ValueError(
                f'predictions ({len(predictions)}) and test_set '
                f'({len(test_set)}) length mismatch')
        os.makedirs(self._out_dir, exist_ok=True)

        ordered_sub_ids = []
        # problem_id -> {'gen': [sub_id...], 'sel': [sub_id...]}
        main_steps = {}

        for idx, prediction_list in enumerate(predictions):
            row = test_set[idx]
            problem_id = str(row['id'])
            prompt_list = row['prompt']
            num_tests = len(row['test'])
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

            testdir_path = os.path.join(self._out_dir, str(problem_id))
            os.makedirs(testdir_path, exist_ok=True)

            info = main_steps.setdefault(problem_id, {'gen': [], 'sel': []})
            python_code = row['import']
            pred_idx = 0
            for test_idx in range(num_tests):
                preprovided_key = (problem_id, test_idx)
                test_lst = row['test'][test_idx]

                if preprovided_key in self.PREPROVIDED_CODE:
                    # Always inject pre-provided code (later steps depend on it).
                    code_file = os.path.join(
                        self.eval_data_path,
                        self.PREPROVIDED_CODE[preprovided_key])
                    with open(code_file, 'r', encoding='utf-8') as f:
                        python_code = append_code_chunk(python_code, f.read())
                    continue

                # Accumulate every generated step's code regardless of selection.
                response = prediction_list[pred_idx]
                python_code = append_code_chunk(
                    python_code, self.extract_python_script(response))
                pred_idx += 1

                sub_id = f'{problem_id}-{test_idx + 1}'
                info['gen'].append(sub_id)
                if sub_id not in self.selected_sub_ids:
                    continue  # generated for context, but not part of the subset

                info['sel'].append(sub_id)
                step_id = f'{problem_id}.{test_idx + 1}'
                testfile_path = os.path.join(testdir_path, f'{sub_id}.py')
                ordered_sub_ids.append(sub_id)
                with open(testfile_path, 'w', encoding='utf-8') as f:
                    f.write(python_code)
                    f.write('\nfrom opencompass.datasets.scicode '
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

        # Execute only the selected sub-step scripts.
        python_scripts = []
        for root, _dirs, files in os.walk(self._out_dir):
            for file in files:
                if file.endswith('.py'):
                    python_scripts.append(os.path.join(root, file))
        python_scripts.sort()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.run_script, s)
                       for s in python_scripts]
        results = [future.result() for future in futures]

        # sub_id -> 1 (pass) / 0 (fail or timeout)
        sub_results_lookup = {}
        for script_path, result in zip(python_scripts, results):
            sub_id = os.path.basename(script_path).replace('.py', '')
            sub_results_lookup[sub_id] = 1 if result == 0 else 0

        # Weighted sub-step accuracy over the selected sub-steps.
        w_correct = w_total = 0.0
        n_correct = n_total = 0
        for sub_id in ordered_sub_ids:
            correct = sub_results_lookup.get(sub_id, 0)
            weight = float(self.sub_weights.get(sub_id, 0.0))
            w_correct += weight * correct
            w_total += weight
            n_correct += correct
            n_total += 1

        # Problem-level accuracy, restricted to fully covered problems.
        fully_correct = fully_count = 0
        for _pid, info in main_steps.items():
            if not info['sel'] or set(info['gen']) != set(info['sel']):
                continue
            fully_count += 1
            fully_correct += int(
                all(sub_results_lookup.get(s, 0) == 1 for s in info['sel']))

        ordered_detail = {sid: sub_results_lookup[sid]
                          for sid in ordered_sub_ids}
        detail_name = ('sub_results_detail.json'
                       if self.dataset_replica_idx == 0 else
                       f'sub_results_detail_replica{self.dataset_replica_idx}'
                       '.json')
        with open(os.path.join(self._out_dir, detail_name), 'w',
                  encoding='utf-8') as f:
            json.dump(ordered_detail, f, indent=2, ensure_ascii=False)

        weighted_sub_accuracy = (w_correct / w_total * 100) if w_total else 0.0
        return zip_result(
            weighted_sub_accuracy,
            'weighted_sub_accuracy — subset-weighted SciCode subproblem '
            "accuracy. SciCode's other official number, main-problem "
            'accuracy, is reported under '
            'secondary.accuracy_fully_covered_mains, restricted to '
            'problems whose every generated step is present',
            {
                'weighted_sub_accuracy': weighted_sub_accuracy,
                'sub_accuracy_unweighted':
                (n_correct / n_total * 100) if n_total else 0.0,
                'num_sub_samples': n_total,
                'weight_sum': w_total,
                'accuracy_fully_covered_mains':
                (fully_correct / fully_count * 100) if fully_count else 0.0,
                'num_fully_covered_mains': fully_count,
            },
        )
