"""ZipBench adapter for IFEval (weighted instruction-following scoring).

IFEval selection is plain row-level over the natural ``input_data.jsonl`` order
(541 prompts); the generic :class:`zipbench.dataset.ZipSubsetDataset` with
``base_loader='opencompass.datasets.IFEvalDataset'`` provides the full dataset
the anchor index space refers to. Each row carries the official unique ``key``,
used stringified as the spec id (see
``convert_anchor_to_spec._load_ifeval_ids``); the loader emits it nested inside
the ``reference`` dict rather than as a top-level column, so selection is by
index and the drift check is skipped at runtime.

Scoring reuses Google's ``instruction_following_eval`` checkers *verbatim* --
``test_instruction_following_strict`` / ``test_instruction_following_loose``
from ``opencompass.datasets.IFEval.evaluation_main`` -- and changes only the
aggregation, from unweighted totals to subset-weighted ones. IFEval reports
four numbers on two independent axes:

* *prompt-level* -- a prompt counts only if **every** one of its verifiable
  instructions is followed, so the denominator is the number of prompts;
* *inst-level* -- each individual instruction counts on its own, so the
  denominator is the number of instructions (834 over the full set, since a
  prompt carries 1-3 of them);
* *strict* -- the response is checked exactly as emitted;
* *loose* -- the response is additionally checked with its first line, its last
  line and markdown emphasis markers removed, and passes if any variant does.

``Prompt-level-strict-accuracy`` is the headline (the same number OpenCompass's
own OC15 summarizers select for IFEval); the other three are reported under
``secondary`` alongside their unweighted counterparts.

The upstream evaluator is subclassed so the class stays declared as
"IFEvaluator, re-aggregated", but ``score`` re-runs the per-item loop instead of
delegating to it: upstream's ``details`` keep only ``is_strict_correct``
(``all(...)``, lossy for the 236 multi-instruction prompts) and drop the
per-instruction ``follow_instruction_list`` that the two inst-level numerators
need.
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets import IFEvaluator
from opencompass.datasets.IFEval.evaluation_main import (
    InputExample, test_instruction_following_loose,
    test_instruction_following_strict)
from opencompass.registry import ICL_EVALUATORS

from zipbench.result import ZipResultShapeMixin, zip_result


@ICL_EVALUATORS.register_module()
class WeightedIFEvalEvaluator(ZipResultShapeMixin, IFEvaluator):
    """Official IFEval instruction checking with ZipBench subset weights.

    With ``w_i`` the subset weight of prompt ``i``, ``f_ij`` whether its ``j``-th
    verifiable instruction was followed and ``L_i`` its instruction count::

        Prompt-level-*  =  sum(w_i * all_j(f_ij)) / sum(w_i)      * 100
        Inst-level-*    =  sum(w_i * sum_j f_ij)  / sum(w_i * L_i) * 100

    Both are ratios of weighted totals, so the instruction-count denominator
    keeps tracking the prompts that were actually selected, and both are
    invariant to a global rescaling of the weights. At ``w_i == 1`` they reduce
    term-for-term to the official unweighted definitions.
    """

    def __init__(self, breakdown_fields=None, **kwargs):
        # ``zipbench.apply.apply_one`` sets ``breakdown_fields`` on every rewired
        # evaluator dict; IFEval defines no sub-groups, so it is accepted and
        # ignored. ``**kwargs`` forwards ``pred_postprocessor`` (merged in from
        # the base eval_cfg) to ``BaseEvaluator.__init__``, which is what makes
        # ``pred_postprocess`` run before scoring.
        super().__init__(**kwargs)
        self.breakdown_fields = list(breakdown_fields or [])

    def score(self, predictions, references, test_set, origin_prompt):
        # OpenCompass passes exactly the arguments this signature names, so no
        # ``**kwargs`` may appear in it; ``test_set`` carries the ``weight``
        # column that ``ZipSubsetDataset`` adds.
        if not test_set:
            raise ValueError('test set is empty')
        if not (len(predictions) == len(references) == len(test_set)):
            raise ValueError(
                'predictions / references / test_set length mismatch')

        w_prompts = w_insts = 0.0        # sum(w_i), sum(w_i * L_i)
        w_prompt_strict = w_prompt_loose = 0.0
        w_inst_strict = w_inst_loose = 0.0
        n_prompts = n_insts = 0
        n_prompt_strict = n_prompt_loose = 0
        n_inst_strict = n_inst_loose = 0
        n_empty = 0

        details = []
        for index, (pred, refer, sample) in enumerate(
                zip(predictions, references, test_set)):
            if isinstance(pred, list):
                pred = pred[-1]
            weight = float(sample.get('weight', 1.0))

            inp = InputExample(key=refer['key'],
                               instruction_id_list=refer['instruction_id_list'],
                               prompt=refer['prompt'],
                               kwargs=refer['kwargs'])
            # The checkers take each row's kwargs as ``**kwargs``. Arrow unifies
            # the per-row struct schema and fills the fields a row does not use
            # with None, so those have to be dropped first (as upstream does).
            for kwarg in inp.kwargs:
                for k in list(kwarg.keys()):
                    if kwarg[k] is None:
                        kwarg.pop(k, None)

            strict_list = list(
                test_instruction_following_strict(
                    inp, pred).follow_instruction_list)
            loose_list = list(
                test_instruction_following_loose(
                    inp, pred).follow_instruction_list)
            n_inst = len(inp.instruction_id_list)
            is_strict_correct = all(strict_list)
            is_loose_correct = all(loose_list)

            w_prompts += weight
            w_insts += weight * n_inst
            w_prompt_strict += weight * is_strict_correct
            w_prompt_loose += weight * is_loose_correct
            w_inst_strict += weight * sum(strict_list)
            w_inst_loose += weight * sum(loose_list)

            n_prompts += 1
            n_insts += n_inst
            n_prompt_strict += int(is_strict_correct)
            n_prompt_loose += int(is_loose_correct)
            n_inst_strict += sum(strict_list)
            n_inst_loose += sum(loose_list)
            n_empty += int(not pred.strip())

            details.append({
                'prompt': origin_prompt[index] if origin_prompt else None,
                'pred': pred,
                'refer': refer,
                'weight': weight,
                # Kept (upstream drops them) so all four reported numbers can be
                # re-derived from the details file alone.
                'instruction_id_list': list(inp.instruction_id_list),
                'follow_instruction_list_strict': strict_list,
                'follow_instruction_list_loose': loose_list,
                'is_strict_correct': is_strict_correct,
                'is_loose_correct': is_loose_correct,
                'is_correct': is_strict_correct,
                'grade': ('strict' if is_strict_correct else
                          'loose' if is_loose_correct else 'none'),
            })

        def _pct(num, den):
            return (num / den * 100) if den else 0.0

        prompt_strict = _pct(w_prompt_strict, w_prompts)
        inst_strict = _pct(w_inst_strict, w_insts)
        prompt_loose = _pct(w_prompt_loose, w_prompts)
        inst_loose = _pct(w_inst_loose, w_insts)

        return zip_result(
            prompt_strict,
            'Prompt-level-strict-accuracy — subset-weighted share of prompts '
            'whose every verifiable instruction is followed under the official '
            'strict checkers; official micro over the 541 prompts. The three '
            'other official IFEval numbers (inst-level strict, prompt- and '
            'inst-level loose) are under secondary',
            {
                'Prompt-level-strict-accuracy': prompt_strict,
                'Inst-level-strict-accuracy': inst_strict,
                'Prompt-level-loose-accuracy': prompt_loose,
                'Inst-level-loose-accuracy': inst_loose,
                'Prompt-level-strict-accuracy_unweighted':
                _pct(n_prompt_strict, n_prompts),
                'Inst-level-strict-accuracy_unweighted':
                _pct(n_inst_strict, n_insts),
                'Prompt-level-loose-accuracy_unweighted':
                _pct(n_prompt_loose, n_prompts),
                'Inst-level-loose-accuracy_unweighted':
                _pct(n_inst_loose, n_insts),
                'num_samples': n_prompts,
                'num_instructions': n_insts,
                'num_empty_predictions': n_empty,
            },
            details,
        )
