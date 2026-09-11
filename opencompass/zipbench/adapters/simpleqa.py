"""ZipBench adapter for SimpleQA (LLM-judge scoring, weighted subset).

SimpleQA differs from the row-level exact-match benchmarks in *how* it is
scored, not in how rows are selected:

  * selection is plain row-level, so the dataset side reuses the generic
    :class:`zipbench.dataset.ZipSubsetDataset` with
    ``base_loader='opencompass.datasets.SimpleQADataset'`` -- no subclass here;
  * scoring runs through the *subjective* pipeline: ``LMEvaluator`` sends every
    (question, gold, prediction) triple to a judge model which replies with a
    grade letter (A=CORRECT, B=INCORRECT, C=NOT_ATTEMPTED), and a
    ``dict_postprocessor`` turns the judge outputs into metrics
    (``opencompass/datasets/simpleqa.py::simpleqa_postprocess``).

So the only benchmark-specific piece ZipBench needs is a *weighted* dict
postprocessor. ``LMEvaluator.postprocess`` calls it as
``proc(output, output_path, **kwargs)``, which lets the config pass the subset
spec path::

    dict_postprocessor=dict(
        type='zipbench.adapters.simpleqa.simpleqa_zip_postprocess',
        subset_spec='.../zipbench/subsets/SimpleQA/tiny.jsonl')

``output`` is keyed by the sample's 0-based position in the *subset* test
split, which by construction (``zipbench.spec.validate_and_select`` selects
rows in spec order) is exactly the spec's line order -- so weights are joined
by position.

SimpleQA's official report is the *pair* ``accuracy_given_attempted`` / ``f1``
(simple-evals prints exactly these two side by side), so this postprocessor
puts both on top level: ``score`` = weighted f1 (headline row) and
``weighted_accuracy_given_attempted`` as a second table row -- the one
sanctioned two-row exception to the single-``score`` shape (see
``zipbench.result``). Per-item ``correct_i = (grade_i == 'A')``; the weighted
plain accuracy and the unweighted counterparts live under ``secondary``.
All percentages are on the 0-100 scale.
"""
import re

from ..result import zip_result
from ..spec import load_subset_spec


def _grade_letter(judgement: str) -> str:
    """Mirror ``opencompass.datasets.simpleqa._single_simpleqa_postprocess``."""
    match = re.search(r'(A|B|C)', judgement)
    return match.group(0) if match else 'C'  # default: NOT_ATTEMPTED


def simpleqa_zip_postprocess(output: dict, output_path: str,
                             subset_spec: str) -> dict:
    """Weighted SimpleQA metrics from LM-judge outputs.

    The weighted metrics follow the upstream formulas with counts replaced by
    weight mass over ``correct_i = (grade_i == 'A')``:
    ``acc = sum(w_i*correct_i)/sum(w_i)``, ``aga = w_A/(w_A+w_B)``,
    ``f1 = 2*aga*acc/(aga+acc)``. ``score`` = weighted f1;
    ``weighted_accuracy_given_attempted`` is also returned top-level (both
    official metrics get a table row). Unweighted counterparts are included so
    a subset run can always be compared against the plain postprocessor.
    """
    spec = load_subset_spec(subset_spec)
    if len(output) != len(spec):
        raise ValueError(
            f'judge output has {len(output)} samples but subset spec '
            f'{subset_spec} has {len(spec)}; the evaluated dataset does not '
            'match the spec')

    weights = [r['weight'] for r in spec]
    w_sum = sum(weights)

    # Align by position: sample keys are '0'..'n-1' in subset (== spec) order.
    keys = sorted(output.keys(), key=int)

    w_correct = w_incorrect = 0.0
    n_correct = n_incorrect = 0
    for pos, key in enumerate(keys):
        sample = output[key]
        grade = _grade_letter(sample['prediction'])
        w = weights[pos]
        sample['grade_letter'] = grade
        sample['correct'] = grade == 'A'
        sample['weight'] = w
        if grade == 'A':
            w_correct += w
            n_correct += 1
        elif grade == 'B':
            w_incorrect += w
            n_incorrect += 1

    def _final(correct, incorrect, total):
        """Upstream ``get_final_results`` formulas on (weighted) counts."""
        is_correct = correct / total
        given_attempted = correct + incorrect
        acc_given_attempted = (correct / given_attempted
                               if given_attempted > 0 else 0)
        f1 = (2 * acc_given_attempted * is_correct /
              (acc_given_attempted + is_correct)
              if (acc_given_attempted + is_correct) > 0 else 0)
        return is_correct, acc_given_attempted, f1

    w_acc, w_aga, w_f1 = _final(w_correct, w_incorrect, w_sum)
    u_acc, u_aga, u_f1 = _final(float(n_correct), float(n_incorrect),
                                float(len(keys)))

    result = zip_result(
        w_f1 * 100,
        'weighted_f1 — subset-weighted SimpleQA F1 (harmonic mean of '
        'accuracy_given_attempted and the grade-A correct rate; official '
        'micro, 0-100). The official report is the pair f1 + '
        'accuracy_given_attempted, so weighted_accuracy_given_attempted is '
        'also top-level as a second table row',
        {
            'weighted_f1': w_f1 * 100,
            'weighted_accuracy_given_attempted': w_aga * 100,
            'weighted_accuracy': w_acc * 100,
            # Unweighted references (upstream metric definitions on the subset).
            'f1_unweighted': u_f1 * 100,
            'accuracy_given_attempted_unweighted': u_aga * 100,
            'accuracy_unweighted': u_acc * 100,
            'num_samples': len(keys),
        },
        output,
    )
    # Second official metric: its own summary-table row next to score.
    result['weighted_accuracy_given_attempted'] = w_aga * 100
    return result
