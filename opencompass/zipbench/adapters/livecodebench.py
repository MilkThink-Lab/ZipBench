"""ZipBench adapter for LiveCodeBench codegeneration (weighted official
pass@k scoring).

LiveCodeBench selection is plain row-level, so the dataset side reuses the
generic :class:`zipbench.dataset.ZipSubsetDataset` with
``base_loader='opencompass.datasets.LCBOfficialCodeGenerationDataset'``
(release_v6; the loader returns a DatasetDict whose rows carry the native
``question_id``, so the stable-id drift check is active).

Scoring delegates to the upstream lcb_runner codegen harness through
``LCBOfficialCodeGenerationEvaluator.evaluate``: the framework's n dataset
replicas are grouped into one n-sample candidate list per question and run
through ``codegen_metrics`` once, yielding per-question ``pass@k`` floats
(0-100). This adapter only changes the final aggregation from an unweighted
mean to ``sum(w_i * pass@k_i) / sum(w_i)`` over the ZipBench subset weights
(read from the ``weight`` column ``ZipSubsetDataset`` adds).

``dump_lcb_runner_format`` defaults to False here: the lcb_runner-format dump
files cover only the subset's questions, so they are not comparable to a
full-set ``Scenario.codegeneration_*`` record.
"""
# Ensure ``opencompass.datasets`` initialises before ``opencompass.openicl``
# (circular import through the inferencers otherwise).
import opencompass.datasets  # noqa: F401
from opencompass.datasets.livecodebench.official import \
    LCBOfficialCodeGenerationEvaluator
from opencompass.registry import ICL_EVALUATORS

from zipbench.result import ZipResultShapeMixin, zip_result


@ICL_EVALUATORS.register_module()
class WeightedLCBCodeGenerationEvaluator(ZipResultShapeMixin,
                                         LCBOfficialCodeGenerationEvaluator):
    """LCBOfficialCodeGenerationEvaluator with ZipBench subset weights."""

    def __init__(self, breakdown_fields=None, pred_postprocessor=None,
                 dump_lcb_runner_format=False, **kwargs):
        # ``zipbench.apply.apply_one`` sets ``breakdown_fields`` on every
        # rewired evaluator dict; LCB defines no sub-groups, so it is accepted
        # and ignored.
        super().__init__(dump_lcb_runner_format=dump_lcb_runner_format,
                         **kwargs)
        # Assigned AFTER the upstream __init__ (which chains to
        # ``BaseEvaluator.__init__`` and would reset the postprocessor to
        # None). Accepting the kwarg is what lets a ``merge_base_kwargs``
        # manifest carry an eval_cfg postprocessor through.
        self.pred_postprocessor = pred_postprocessor
        self.breakdown_fields = list(breakdown_fields or [])

    def evaluate(self, k, n, original_dataset, predictions, references):
        if n <= 0 or len(references) % n != 0:
            return {
                'error':
                f'len(references)={len(references)} not divisible by n={n}'
            }
        real_size = len(references) // n
        # ``weight`` is the column ZipSubsetDataset adds; with dataset n>1 the
        # test set is replica blocks concatenated in order, so the first block
        # aligns positionally with references[:real_size].
        if 'weight' not in original_dataset.column_names:
            return {
                'error': "original_dataset has no 'weight' column; "
                'is this a ZipSubsetDataset subset?'
            }
        weights = [float(w) for w in original_dataset['weight'][:real_size]]

        result = super().evaluate(k, n, original_dataset,
                                  predictions=predictions,
                                  references=references)
        if 'error' in result:
            return result

        qid_weight = {qid: w for qid, w in zip(references[:real_size],
                                               weights)}
        details = result['details']
        pass_keys = sorted(
            {key for d in details for key in d if key.startswith('pass@')},
            key=lambda s: int(s.split('@')[1]))
        w_sums = {key: 0.0 for key in pass_keys}
        w_total = 0.0
        for detail in details:
            weight = qid_weight[detail['question_id']]
            detail['weight'] = weight
            w_total += weight
            for key in pass_keys:
                if key in detail:
                    w_sums[key] += weight * float(detail[key])
        if w_total <= 0:
            return {'error': 'subset weights sum to zero'}
        weighted = {f'{key}_weighted': w_sums[key] / w_total
                    for key in pass_keys}

        secondary = dict(weighted)
        for key in pass_keys:
            if key in result:
                secondary[f'{key}_unweighted'] = float(result[key])
        secondary['num_samples'] = len(details)

        return zip_result(
            weighted['pass@1_weighted'],
            'weighted_pass@1 — official LiveCodeBench codegeneration pass@1 '
            '(micro over questions), importance-weighted; pass@5 (n>=5) under '
            'secondary',
            secondary,
            details,
        )
