#!/usr/bin/env python3
"""Convert an anchor ``.pkl`` subset definition into an open, language-neutral
JSONL spec (``id`` + ``index`` + ``weight``) plus a small YAML manifest.

The anchor ``.pkl`` files are an *internal* intermediate product of subset
construction. They are Python-only, binary and unsafe to redistribute
(``pickle.load`` can execute arbitrary code). This script materialises them into
a portable spec that is safe to open-source and easy to diff:

  * ``<name>.jsonl`` -- one record per line: ``{"id", "index", "weight"}``
        - ``id``     : the dataset's stable per-question id (LongBench v2 ``_id``)
                       so the subset survives upstream re-ordering / re-hosting.
        - ``index``  : the 0-based position of the question in the *full* dataset
                       (used for fast selection; validated against ``id``).
        - ``weight`` : the (normalised) weight used for weighted scoring.
  * ``manifest.yaml`` -- metadata for every subset version (n, ratio,
        dataset fingerprint, weighting formula).

Anchor ``.pkl`` structure (per dataset key)::

    {dataset_key: {"indices": ndarray[int],   # 0-based
                   "weights": ndarray[float],  # normalised, sums to 1
                   "by_sub": {...}}}

Example::

    python convert_anchor_to_spec.py \
        --pkl  /path/to/anchors_run9/anchor_..._mae_0.0215_ratio_0.7495.pkl \
        --raw  opencompass/data/longbenchv2/data.json \
        --dataset-key LongBenchv2_0shot \
        --name tiny \
        --base-loader opencompass.datasets.LongBenchv2Dataset \
        --breakdown-fields difficulty,length \
        --out-dir opencompass/zipbench/subsets/LongBenchv2_0shot
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import re

import numpy as np

WEIGHTING_FORMULA = (
    'weighted_acc = sum(w_i * correct_i) * 100   '
    '# weights are normalised so that sum(w_i) == 1'
)

# Per ``--units-from`` profile: how the anchor ``index`` maps to a selection
# unit, and which zipbench loader/evaluator the manifest should point at.
# ``rows`` is the original behaviour (index == dataset row, exact-match
# scoring); ``scicode`` selects sub-steps and scores via code execution.
UNIT_PROFILES = {
    'rows': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.evaluator.WeightedAccuracyEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    'scicode': dict(
        dataset_type='zipbench.adapters.scicode.ZipSciCodeDataset',
        evaluator_type='zipbench.adapters.scicode.WeightedSciCodeEvaluator',
        selection_unit='subproblem',
        merge_base_kwargs=True,
        inject_subset_spec=True,
    ),
    # SimpleQA selects rows (generic ZipSubsetDataset) but is scored by an LLM
    # judge through the subjective pipeline, so the "evaluator" here is a
    # weighted *dict_postprocessor* plugged into LMEvaluator (which receives
    # subset_spec as a kwarg -> inject_subset_spec=True). See
    # ``zipbench/adapters/simpleqa.py``.
    'simpleqa': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.simpleqa.simpleqa_zip_postprocess',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=True,
    ),
    # ArenaHard selects rows (generic ZipSubsetDataset) but is scored by an
    # LLM judge through the subjective pipeline (LMEvaluator, m2n mode with
    # two judged games per question), so the "evaluator" here is a weighted
    # *dict_postprocessor* plugged into LMEvaluator (which receives
    # subset_spec as a kwarg -> inject_subset_spec=True). See
    # ``zipbench/adapters/arenahard.py``.
    'arenahard': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.arenahard.arenahard_zip_postprocess',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=True,
    ),
    # C3 selects flattened *question* rows (C3Dataset_V2 expands each raw
    # dialogue record into one row per question), so the generic 'rows' id
    # enumeration over the raw json (one id per dialogue) would be wrong.
    # Same wiring as 'rows'; only the id enumerator differs
    # (``_load_c3_ids``).
    'c3': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.evaluator.WeightedAccuracyEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # TheoremQA selects rows (generic ZipSubsetDataset) but is scored by the
    # numeric-comparison logic of TheoremQAEvaluatorV3, so the evaluator is a
    # weighted adapter. The anchor item order is NOT natural: rows are
    # sorted lexicographically by 'TheoremQA_<row>' abbr
    # (see _load_theoremqa_ids), so anchor->natural remapping applies.
    'theoremqa': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.theoremqa.WeightedTheoremQAEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # TruthfulQA selects rows (generic ZipSubsetDataset) but is scored by the
    # generation-mode bleu/rouge logic of TruthfulQAEvaluator (multi sub-metric
    # max/diff/acc), so the evaluator is a weighted adapter.
    # merge_base_kwargs=True keeps the base eval_cfg's metrics=('bleu','rouge')
    # so the weighted evaluator scores the same metrics (and does NOT load any
    # judge model). Item order is NON-natural: anchor indices are in the HF
    # `multiple_choice` order, which differs from the `generation` order the
    # runtime loader uses (815/817 rows differ), so an anchor->natural remap
    # applies; see _load_truthfulqa_ids.
    'truthfulqa': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.truthfulqa.WeightedTruthfulQAEvaluator',
        selection_unit='row',
        merge_base_kwargs=True,
        inject_subset_spec=False,
    ),
    # SciBench selects rows of the *concatenated* 10-subject dataset
    # (ScibenchZipDataset; the upstream config splits subjects into separate
    # datasets) and is scored by numeric tolerance, so the evaluator is a
    # weighted adapter using last-number numeric-tolerance judging. The
    # anchor item order is natural concatenation order MINUS the one
    # empty-answer question (natural index 309, fund[24]) -- see
    # _load_scibench_ids, which returns the 582-item anchor_order for
    # remapping.
    'scibench': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.scibench.WeightedScibenchEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # HumanEval+ selects rows (generic ZipSubsetDataset) but is scored by
    # evalplus code execution, so the evaluator is a weighted adapter around
    # HumanEvalPlusEvaluator. The anchor item order is NOT natural: rows are
    # sorted lexicographically by 'humaneval_plus_<row>'
    # abbr (see _load_humaneval_plus_ids), so anchor->natural remapping
    # applies.
    'humaneval_plus': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type=(
            'zipbench.adapters.humaneval_plus.WeightedHumanEvalPlusEvaluator'),
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # MBPP+ selects rows (generic ZipSubsetDataset) but is scored by evalplus
    # code execution, so the evaluator is a weighted adapter around
    # MBPPEvaluator(metric='MBPPPlus'). The anchor item order is natural
    # (official MbppPlus jsonl line order == task_id numeric order), so no
    # remapping applies.
    'mbpp_plus': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type=(
            'zipbench.adapters.mbpp_plus.WeightedMbppPlusEvaluator'),
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # ARC-Challenge selects rows (generic ZipSubsetDataset) scored by plain
    # exact-match accuracy (WeightedAccuracyEvaluator), same wiring as 'rows'.
    # The anchor item order is NOT natural: it is the HF
    # open-llm-leaderboard-old details parquet order, a non-natural permutation
    # of ARC-Challenge-Test.jsonl. That permutation cannot be reproduced by a
    # sort, so it is stored as data (arc_challenge_anchor_order.json) and read
    # by _load_arc_challenge_ids for anchor->natural remapping.
    'arc_challenge': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.evaluator.WeightedAccuracyEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # GSM8K selects rows (generic ZipSubsetDataset) but is scored by the
    # numeric-tolerance judging of Gsm8kEvaluator (float equality within 1e-6,
    # not string exact-match), so the evaluator is a weighted adapter. The
    # anchor item order is NOT natural: it is the HF open-llm-leaderboard-old
    # details parquet order (lm-eval 5-shot), which
    # is random.Random(42).shuffle of the natural test.jsonl line order. Like
    # ARC that permutation is stored as data (gsm8k_anchor_order.json) and read
    # by _load_gsm8k_ids for anchor->natural remapping.
    'gsm8k': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.gsm8k.WeightedGsm8kEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # MATH-Hard (MATH Level-5, Open LLM Leaderboard v2 setting; 1324 items, NOT
    # the 5000-item OpenCompass math.json) selects rows (generic
    # ZipSubsetDataset over base_loader='opencompass.datasets.MATHDataset' with
    # file_name='math_hard.json') scored by the upstream MATHEvaluator
    # mathematical-equivalence judging (is_equiv, version v2), so the evaluator
    # is a weighted adapter (plain string-exact would under-count equivalent
    # LaTeX forms). Order is NATURAL: the anchor `indices` index the 1324
    # items in alphabetical subject-block order -- exactly the order of
    # data/math_hard/math_hard.json -- so no anchor->natural remap is needed
    # (_load_math_ids returns a 2-tuple, anchor_order stays None).
    'math': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.math.WeightedMATHEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # GPQA selects rows of the concatenated diamond+extended+main dataset
    # (GPQAZipDataset; the upstream config evaluates one csv at a time) scored by
    # plain exact-match on the extracted "ANSWER: LETTER"
    # (WeightedAccuracyEvaluator), same wiring as 'rows'. Order is NATURAL: the
    # anchor `indices` index the 1192-row concatenated space
    # (diamond 0-197 -> extended 198-743 ->
    # main 744-1191 by doc_id), which is the order GPQAZipDataset produces -- so
    # no anchor->natural remap is needed (_load_gpqa_ids returns a 2-tuple,
    # anchor_order stays None). Stable ids are subset-qualified positional
    # ('<subset>_<idx>') because the 1192-row space has only 546 unique questions.
    'gpqa': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.evaluator.WeightedAccuracyEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # BBH selects rows of the *concatenated* 24 leaderboard tasks
    # (BBHAllDataset; the upstream config is one dataset per task) and is scored
    # by the two OpenCompass BBH post-processors + a MACRO (per-task, then equal
    # weight over tasks) aggregation, so the evaluator is a weighted adapter.
    # Order is NATURAL: the anchor `indices` index the 5761 rows in
    # sorted()-task-block order,
    # each task in data/BBH/data/<task>.json order -- exactly what BBHAllDataset
    # produces -- so no
    # anchor->natural remap is needed (_load_bbh_ids returns a 2-tuple,
    # anchor_order stays None).
    'bbh': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.bbh.WeightedBBHEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # HellaSwag selects rows (generic ZipSubsetDataset) scored by plain
    # exact-match accuracy on the extracted A/B/C/D letter
    # (WeightedAccuracyEvaluator), same wiring as 'rows' / 'arc_challenge'. The
    # anchor item order is NOT natural: it is the HF open-llm-leaderboard-old
    # details parquet order (lm-eval
    # 10-shot, correctness = metrics['acc'] NOT acc_norm), a non-natural
    # permutation of hellaswag.jsonl line order. That permutation is stored as
    # data (hellaswag_anchor_order.json) and read by
    # _load_hellaswag_ids for anchor->natural remapping.
    'hellaswag': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.evaluator.WeightedAccuracyEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # Winogrande selects rows (generic ZipSubsetDataset) scored by plain
    # exact-match accuracy on the extracted A/B letter (WeightedAccuracyEvaluator),
    # same wiring as 'rows' / 'arc_challenge' / 'hellaswag'. The upstream loader
    # WinograndeDatasetV3 returns a DatasetDict{train_xs, dev}; the manifest sets
    # test_split='dev', so ZipSubsetDataset subsets only 'dev' and keeps the full
    # 'train_xs' for the 5-shot FixKRetriever. The anchor item order is NOT
    # natural: it is the HF open-llm-leaderboard-old details
    # parquet order (lm-eval 5-shot, correctness = metrics['acc']), a non-natural
    # permutation of dev.jsonl line order. That permutation is stored as data
    # (winogrande_anchor_order.json) and read by
    # _load_winogrande_ids for anchor->natural remapping.
    'winogrande': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.evaluator.WeightedAccuracyEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # MMLU selects rows of the *concatenated* 57 subjects (MMLUAllDataset; the
    # upstream config is one dataset per subject) scored by plain **micro**
    # exact-match on the extracted "ANSWER: LETTER" (WeightedAccuracyEvaluator),
    # same wiring as 'rows' / 'gpqa'. Order is NATURAL: the anchor indices
    # index the 14042 rows in sorted()-subject-block order, each subject in
    # its <subject>_test.csv row order -- exactly what MMLUAllDataset produces --
    # so no anchor->natural remap is needed (_load_mmlu_ids returns a 2-tuple,
    # anchor_order stays None). Scoring is micro (flat item-level weighted acc);
    # do not switch to a BBH-style within-subject macro -- MMLU subject sizes
    # are unbalanced.
    'mmlu': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.evaluator.WeightedAccuracyEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # MMLU-Pro selects rows of the 14 categories flattened into the native HF
    # test-split order (MMLUProAllDataset; the upstream config is one dataset per
    # category), scored by plain **micro** exact-match on the extracted
    # "ANSWER: LETTER" (WeightedAccuracyEvaluator), same wiring as 'rows' /
    # 'gpqa' / 'mmlu'. Order is NATURAL: the anchor indices index the
    # 12032 rows in
    # native parquet == Open-LLM-Leaderboard leaderboard_mmlu_pro doc_id order --
    # exactly what MMLUProAllDataset produces -- so no
    # anchor->natural remap is needed (_load_mmlu_pro_ids returns a 2-tuple,
    # anchor_order stays None). Scoring is micro; do not switch to a
    # within-category macro -- MMLU-Pro category sizes are unbalanced
    # (381..1351).
    'mmlu_pro': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.evaluator.WeightedAccuracyEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # MuSR selects rows of the 3 scenarios concatenated into one flat 756-row
    # index space (MusrAllDataset; the upstream config is 3 separate datasets,
    # one name= per scenario), scored by a custom weighted **macro** evaluator
    # (WeightedMusrEvaluator: last-"ANSWER: <int>" line == gold choice number,
    # per-scenario weighted acc then an equal-weight mean over the 3 scenarios --
    # the official MuSR / musr_average metric, same shape as the 'bbh' profile).
    # Order is NATURAL: the anchor indices index the 756-row space
    # ([murder_mysteries 0:250, object_placements 250:506, team_allocation
    # 506:756]) in that scenario-block order, each scenario in natural (story,
    # question) order -- exactly what MusrAllDataset produces -- so no
    # anchor->natural remap is needed (_load_musr_ids returns a 2-tuple,
    # anchor_order stays None). The reported metric is macro (the
    # official one); micro is still reported as a secondary metric.
    'musr': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.musr.WeightedMusrEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
    # IFEval selects rows (generic ZipSubsetDataset over
    # base_loader='opencompass.datasets.IFEvalDataset') but is scored by
    # Google's official instruction_following_eval checkers, which report four
    # numbers (prompt- and inst-level x strict and loose), so the evaluator is a
    # weighted adapter. merge_base_kwargs=True keeps the base eval_cfg
    # evaluator's `pred_postprocessor` (extract-non-reasoning-content) on the
    # swapped-in evaluator: IFEval checks the surface form of the answer, so a
    # chain of thought has to be stripped before the checkers run.
    # Order is NATURAL, i.e. data/ifeval/input_data.jsonl line order, verified
    # decisively against the Open LLM Leaderboard v2 per-sample details of two
    # of the correctness matrix's models (541/541 doc keys and prompt texts in
    # both, and 541/541 exact agreement on prompt_level_strict_acc), so no
    # anchor->natural remap is needed and _load_ifeval_ids returns a 2-tuple.
    # `by_sub` in the anchor pkls is a single degenerate bucket, hence
    # breakdown_fields=[].
    'ifeval': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.ifeval.WeightedIFEvalEvaluator',
        selection_unit='row',
        merge_base_kwargs=True,
        inject_subset_spec=False,
    ),
    # LiveCodeBench (codegeneration) selects rows (generic ZipSubsetDataset
    # over base_loader='opencompass.datasets.LCBOfficialCodeGenerationDataset',
    # release_v6, 1055 problems) but is scored by the official lcb_runner
    # codegen harness (n samples per problem grouped into pass@k), so the
    # evaluator is a weighted adapter around LCBOfficialCodeGenerationEvaluator.
    # merge_base_kwargs=True keeps the base eval_cfg evaluator's
    # release_version / lm_style / timeout kwargs on the swapped-in evaluator.
    # Order is NATURAL: the loader sorts problems by question_id (matching the
    # upstream runner order), and the anchor `indices` index that same sorted
    # order, so no anchor->natural remap is needed (_load_livecodebench_ids
    # returns a 2-tuple, anchor_order stays None).
    'livecodebench': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type=(
            'zipbench.adapters.livecodebench.WeightedLCBCodeGenerationEvaluator'),
        selection_unit='row',
        merge_base_kwargs=True,
        inject_subset_spec=False,
    ),
    # LongBench v1 selects rows of the 21 tasks concatenated into one flat
    # 4750-row index space (LongbenchAllDataset; the upstream config is 21
    # separate datasets with heterogeneous prompts / metrics / max_out_len),
    # scored by a custom weighted evaluator (WeightedLongbenchEvaluator:
    # per-item official LongBench scores -- F1 / ROUGE-L / fuzz / count /
    # retrieval / classification, dispatched on the task column -- aggregated
    # as the six-category equal-weight macro; micro and the official EN/ZH
    # leaderboard tables are secondary). Per-row max_out_len (32/64/128/512)
    # is honoured by zipbench.adapters.longbench.LongBenchGenInferencer.
    # Order is the CATEGORY-GROUPED concatenation (single_doc_qa ->
    # multi_doc_qa -> summarization -> few_shot -> synthetic -> code, official
    # task order inside each category == longbench_tasks.CATEGORY_MAP, natural
    # jsonl line order inside each task). The anchor `indices` index this
    # concatenation directly (_load_longbench_ids returns a 2-tuple,
    # anchor_order stays None).
    'longbench': dict(
        dataset_type='zipbench.dataset.ZipSubsetDataset',
        evaluator_type='zipbench.adapters.longbench.WeightedLongbenchEvaluator',
        selection_unit='row',
        merge_base_kwargs=False,
        inject_subset_spec=False,
    ),
}


def _load_scicode_ids(raw_path):
    """Enumerate SciCode sub-step ids (``'<problem>-<step>'``) in dataset order.

    SciCode's selection unit is the sub-step: the *generated* steps of every
    problem, in natural file order, skipping the pre-provided steps (which the
    model never generates). The id is the sub-step id used in the evaluator's
    ``sub_results_detail.json`` (e.g. ``'13-6'``), so a spec row is validated
    against the exact sub-step it scores.

    Returns ``(ids, anchor_order, fingerprint)``:

      * ``ids`` -- sub-step ids in *dataset natural order*; a spec ``index`` is
        a position in this list (the order the zipbench loader/evaluator see).
      * ``anchor_order`` -- the same ids in the *anchor item order*: the 288
        sub-steps sorted lexicographically by sub_id, so an anchor
        ``indices[i]`` denotes
        ``sorted(ids)[indices[i]]`` -- NOT ``ids[indices[i]]``. The converter
        translates anchor positions into natural-order spec indices.
    """
    from zipbench.adapters.scicode import enumerate_sub_ids
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    ids = enumerate_sub_ids(test_data)
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, sorted(ids), fingerprint


def _load_simpleqa_ids(raw_path):
    """Enumerate SimpleQA row ids in natural CSV order.

    The CSV (``simple_qa_test_set.csv``) has no stable id column, so the id is
    a content hash of ``problem + answer`` (the pair is unique across all 4326
    rows). The anchor ``indices`` are positions in this same natural CSV
    order, so no anchor-order remapping is needed (``anchor_order is None``).
    """
    import csv
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8', newline='') as f:
        for rec in csv.DictReader(f):
            basis = (str(rec.get('problem', '')) + '\x1f' +
                     str(rec.get('answer', '')))
            ids.append(hashlib.sha1(basis.encode('utf-8')).hexdigest()[:16])
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate problem+answer rows in {raw_path}; '
                         'content-hash ids would be ambiguous')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_arenahard_ids(raw_path):
    """Enumerate ArenaHard question ids in natural jsonl order.

    ``arenahard.jsonl`` carries a globally-unique ``question_id`` per line;
    the anchor pkls' ``indices`` are positions in this same natural order, so
    no anchor-order remapping is needed (``anchor_order is None``).
    """
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)['question_id']))
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate question_id in {raw_path}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_truthfulqa_ids(raw_path):
    """Enumerate TruthfulQA ids; natural = HF ``generation`` validation order.

    ``raw_path`` is the HF dataset id (``'truthful_qa'``). The runtime loader
    ``opencompass.datasets.TruthfulQADataset`` reads the ``generation`` config's
    ``validation`` split, so the NATURAL (spec) order is that config's order.
    Rows carry no stable id, so the id is a content hash ``sha1(question)[:16]``
    (the 817 questions are unique).

    Returns the 3-tuple ``(ids, anchor_order, fingerprint)``. The anchor
    ``indices`` are NOT positions in ``generation`` order: they follow the HF
    ``multiple_choice`` config order. That order differs from
    ``generation`` (815/817 rows differ), so ``anchor_order`` lists the natural
    (generation) ids in ``multiple_choice`` order and the converter remaps anchor
    positions into ``generation``-order spec indices. The two configs share the
    same 817 questions (one
    question differs only by a trailing space), so ``sha1`` of the
    whitespace-normalised question joins them.
    """
    from datasets import load_dataset

    def _qid(q):
        # Normalise whitespace so the one question that differs only by a
        # trailing space between the two configs still joins.
        return hashlib.sha1(
            re.sub(r'\s+', ' ', str(q).strip()).encode('utf-8')).hexdigest()[:16]

    gen = load_dataset(raw_path, 'generation', split='validation')
    mc = load_dataset(raw_path, 'multiple_choice', split='validation')
    ids = [_qid(q) for q in gen['question']]
    anchor_order = [_qid(q) for q in mc['question']]
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate generation question rows in {raw_path}; '
                         'content-hash ids would be ambiguous')
    if set(anchor_order) != set(ids) or len(anchor_order) != len(ids):
        raise ValueError(
            'truthful_qa generation vs multiple_choice question sets differ; '
            'cannot remap anchor (MC2) order onto generation order')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, anchor_order, fingerprint


def _load_c3_ids(raw_path):
    """Enumerate C3 question ids in C3Dataset_V2's flattened natural order.

    The raw file (``data/CLUE/C3/dev_0.json``) is a JSON array of dialogue
    records ``[content, questions, dialogue_id]`` (e.g. 1628 dialogues for
    dev_0). ``opencompass.datasets.C3Dataset_V2`` flattens it into one row per
    question (1825 rows), and that flattened order is the anchor index
    space, so no anchor-order remapping is needed (``anchor_order is None``).

    Each question's stable id is ``<dialogue_id>-<question_idx>`` (e.g.
    ``47-275-0``); falls back to a content hash if dialogue ids are missing
    or would collide.
    """
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        raw = json.load(f)
    ids = []
    for line in raw:
        content = ''.join([''.join(paragraph) for paragraph in line[0]])
        dlg_id = line[2] if len(line) > 2 and line[2] else None
        for qidx, question in enumerate(line[1]):
            if dlg_id is not None:
                ids.append(f'{dlg_id}-{qidx}')
            else:
                basis = content + '\x1f' + str(question.get('question', ''))
                ids.append(hashlib.sha1(basis.encode('utf-8')).hexdigest()[:16])
    if not ids:
        raise ValueError(f'No questions parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate question ids in {raw_path}; '
                         'dialogue ids are not unique')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_theoremqa_ids(raw_path):
    """Enumerate TheoremQA row ids in natural JSON order.

    The raw file (``data/TheoremQA/theoremqa_test.json``) is a JSON array of
    800 records with a unique ``id`` field (e.g.
    ``jianyu_xu/Lah_number_6.json``), used as the stable spec id.

    Returns ``(ids, anchor_order, fingerprint)``. The anchor ``indices``
    are NOT positions in natural order: items are sorted by their opencompass
    ``example_abbr`` string (``TheoremQA_0, TheoremQA_1, TheoremQA_10,
    TheoremQA_100, ...``). ``anchor_order`` therefore lists the natural-order
    ids re-sorted by that lexicographic abbr order, and the converter remaps
    anchor positions into natural-order spec indices.
    """
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        raw = json.load(f)
    ids = [str(rec['id']) for rec in raw]
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate id values in {raw_path}')
    lex = sorted(range(len(ids)), key=lambda i: f'TheoremQA_{i}')
    anchor_order = [ids[i] for i in lex]
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, anchor_order, fingerprint


def _load_humaneval_plus_ids(raw_path):
    """Enumerate HumanEval+ row ids in natural jsonl order.

    The raw file (``data/humaneval/human-eval-v2-20210705.jsonl``) is JSON
    Lines with 164 records carrying a unique ``task_id`` (``HumanEval/0`` ...
    ``HumanEval/163``), used as the stable spec id.

    Returns ``(ids, anchor_order, fingerprint)``. The anchor ``indices``
    are NOT positions in natural order: items are sorted by their opencompass
    ``example_abbr`` string
    (``humaneval_plus_0, humaneval_plus_1, humaneval_plus_10,
    humaneval_plus_100, ...``). ``anchor_order`` therefore lists the
    natural-order ids re-sorted by that lexicographic abbr order, and the
    converter remaps anchor positions into natural-order spec indices.
    """
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)['task_id']))
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate task_id values in {raw_path}')
    lex = sorted(range(len(ids)), key=lambda i: f'humaneval_plus_{i}')
    anchor_order = [ids[i] for i in lex]
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, anchor_order, fingerprint


def _load_mbpp_plus_ids(raw_path):
    """Enumerate MBPP+ row ids in natural jsonl order.

    The raw file (``data/mbpp_plus/MbppPlus-v0.1.0.jsonl``) is JSON Lines
    with 399 records carrying a unique ``task_id`` (``Mbpp/2`` ...
    ``Mbpp/809``), used as the stable spec id. The anchor ``indices`` are
    positions in this natural order, so no remapping applies.
    """
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)['task_id']))
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate task_id values in {raw_path}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_arc_challenge_ids(raw_path):
    """Enumerate ARC-Challenge row ids in natural jsonl order.

    ``raw_path`` is ``data/ARC/ARC-c/ARC-Challenge-Test.jsonl`` (1172 records,
    each with a unique ``id`` such as ``Mercury_7175875``), matching the ``id``
    column produced by ``opencompass.datasets.ARCDatasetAllChoices`` (which
    keeps every question, unlike ``ARCDataset`` that drops the 7 non-4-choice
    ones). The id is the stable spec id.

    Returns ``(ids, anchor_order, fingerprint)``. The anchor ``indices``
    are NOT positions in natural order: they follow the
    HF ``open-llm-leaderboard-old`` details parquet order (config
    ``harness_arc_challenge_25``, lm-eval 25-shot), a non-natural permutation of
    the jsonl line order. Unlike TheoremQA / HumanEval+ that order is not a
    sort, so it is stored as data in ``arc_challenge_anchor_order.json``.
    ``anchor_order``
    lists the natural ids in that parquet order; the converter remaps anchor
    positions into natural-order spec indices.
    """
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)['id']))
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate id values in {raw_path}')
    order_path = osp.join(osp.dirname(__file__), 'adapters',
                          'arc_challenge_anchor_order.json')
    with open(order_path, 'r', encoding='utf-8') as f:
        anchor_order = json.load(f)['order']
    if set(anchor_order) != set(ids) or len(anchor_order) != len(ids):
        raise ValueError(
            f'{order_path} order does not match the ids in {raw_path}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, anchor_order, fingerprint


def _load_gsm8k_ids(raw_path):
    """Enumerate GSM8K row ids in natural ``test.jsonl`` order.

    ``raw_path`` is ``data/gsm8k/test.jsonl`` (1319 records, each
    ``{question, answer}``). GSM8K rows carry no stable id, so the id is a
    content hash ``sha1(question + '\\x1f' + answer)[:16]`` -- the same scheme
    as ``_rec_id`` / the ``rows`` profile, and unique across all 1319 rows.
    This matches the natural order of ``opencompass.datasets.GSM8KDataset``
    (which reads ``test.jsonl`` line by line).

    Returns ``(ids, anchor_order, fingerprint)``. The anchor ``indices``
    are NOT positions in natural order: they follow the
    HF ``open-llm-leaderboard-old`` details parquet order (lm-eval
    5-shot), a non-natural permutation of the jsonl line order equal to
    ``random.Random(42).shuffle(range(1319))`` (the harness shuffles test docs
    before evaluation). That order is stored as data in
    ``gsm8k_anchor_order.json``. ``anchor_order`` lists the natural ids
    in that parquet order; the converter remaps anchor positions into
    natural-order spec indices.
    """
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                basis = (str(rec.get('question', '')) + '\x1f' +
                         str(rec.get('answer', '')))
                ids.append(hashlib.sha1(basis.encode('utf-8')).hexdigest()[:16])
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate question+answer rows in {raw_path}; '
                         'content-hash ids would be ambiguous')
    order_path = osp.join(osp.dirname(__file__), 'adapters',
                          'gsm8k_anchor_order.json')
    with open(order_path, 'r', encoding='utf-8') as f:
        anchor_order = json.load(f)['order']
    if set(anchor_order) != set(ids) or len(anchor_order) != len(ids):
        raise ValueError(
            f'{order_path} order does not match the ids in {raw_path}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, anchor_order, fingerprint


def _load_math_ids(raw_path):
    """Enumerate MATH-Hard row ids in the correctness-matrix row order.

    ``raw_path`` is ``data/math_hard/math_hard.json`` -- a dict
    ``{key: {problem, level, type, solution}}`` whose *insertion order* is the
    1324-item matrix row order (the alphabetical concatenation of the 7
    MATH-Hard subject blocks; see the offline builder).
    ``opencompass.datasets.MATHDataset`` iterates the same ``data.keys()`` in
    the same order, so ``ids[i]`` is the row selected by anchor index ``i``.
    MATH rows carry no stable id, so the id is a content hash
    ``sha1(problem + '\\x1f' + solution)[:16]`` -- the same scheme as
    ``_rec_id`` -- unique across all 1324 rows (provenance only; the runtime
    drift check is skipped as the loaded rows have no ``_id`` column).

    Returns the 2-tuple ``(ids, fingerprint)``. Order is NATURAL (== the
    anchor index space by construction), so no anchor->natural remap is needed
    (``anchor_order`` stays None).
    """
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        data = json.load(f)
    ids = []
    for k in data.keys():
        rec = data[k]
        basis = (str(rec.get('problem', '')) + '\x1f' +
                 str(rec.get('solution', '')))
        ids.append(hashlib.sha1(basis.encode('utf-8')).hexdigest()[:16])
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate problem+solution rows in {raw_path}; '
                         'content-hash ids would be ambiguous')
    if len(ids) != 1324:
        raise ValueError(f'expected 1324 MATH-Hard rows, got {len(ids)}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_hellaswag_ids(raw_path):
    """Enumerate HellaSwag row ids in natural ``hellaswag.jsonl`` order.

    ``raw_path`` is ``data/hellaswag/hellaswag.jsonl`` (10042 validation records
    ``{query, choices, gold}``). Rows carry no stable id, so the id is a content
    hash ``sha1(query + '\\x1f' + '\\x1f'.join(choices))[:16]`` -- unique across
    all 10042 rows (query alone is not: 22 activity contexts repeat with
    different endings). This matches the natural order of
    ``opencompass.datasets.HellaswagDataset_V2`` (which reads ``hellaswag.jsonl``
    line by line).

    Returns ``(ids, anchor_order, fingerprint)``. The anchor ``indices``
    are NOT positions in natural order: they follow the
    HF ``open-llm-leaderboard-old``
    details parquet order (lm-eval 10-shot, correctness = ``metrics['acc']`` NOT
    acc_norm), a non-natural permutation of the jsonl line order. Like ARC /
    GSM8K that order is stored as data in ``hellaswag_anchor_order.json``.
    ``anchor_order`` lists the natural ids in that parquet
    order; the converter remaps anchor positions into natural-order spec
    indices.
    """
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                basis = (str(rec.get('query', '')) + '\x1f' +
                         '\x1f'.join(rec.get('choices', [])))
                ids.append(hashlib.sha1(basis.encode('utf-8')).hexdigest()[:16])
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate query+choices rows in {raw_path}; '
                         'content-hash ids would be ambiguous')
    order_path = osp.join(osp.dirname(__file__), 'adapters',
                          'hellaswag_anchor_order.json')
    with open(order_path, 'r', encoding='utf-8') as f:
        anchor_order = json.load(f)['order']
    if set(anchor_order) != set(ids) or len(anchor_order) != len(ids):
        raise ValueError(
            f'{order_path} order does not match the ids in {raw_path}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, anchor_order, fingerprint


def _load_winogrande_ids(raw_path):
    """Enumerate Winogrande row ids in natural ``dev.jsonl`` order.

    ``raw_path`` is ``data/winogrande/dev.jsonl`` (1267 records, each with a
    unique ``qID`` such as ``3FCO4VKOZ4BJQ6IFC0VAIBK4KTWE7U-2``). The qID is the
    stable spec id (both qID and sentence are 1267-unique). This matches the
    natural order of ``opencompass.datasets.WinograndeDatasetV3`` (which reads
    ``dev.jsonl`` line by line into the ``dev`` split).

    Returns ``(ids, anchor_order, fingerprint)``. The anchor ``indices``
    are NOT positions in natural order: they follow the
    HF ``open-llm-leaderboard-old``
    details parquet order (lm-eval 5-shot, correctness = ``metrics['acc']``), a
    non-natural permutation of the jsonl line order (1266/1267 rows differ). Like
    ARC / GSM8K / HellaSwag that permutation is stored as data in
    ``winogrande_anchor_order.json``.
    ``anchor_order`` lists the natural qIDs in that parquet order; the
    converter remaps anchor positions into natural-order spec indices.
    """
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)['qID']))
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate qID values in {raw_path}')
    order_path = osp.join(osp.dirname(__file__), 'adapters',
                          'winogrande_anchor_order.json')
    with open(order_path, 'r', encoding='utf-8') as f:
        anchor_order = json.load(f)['order']
    if set(anchor_order) != set(ids) or len(anchor_order) != len(ids):
        raise ValueError(
            f'{order_path} order does not match the ids in {raw_path}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, anchor_order, fingerprint


def _load_gpqa_ids(raw_path):
    """Enumerate GPQA row ids over the concatenated diamond+extended+main csvs.

    ``raw_path`` is the ``data/gpqa`` *directory* holding ``gpqa_diamond.csv``,
    ``gpqa_extended.csv`` and ``gpqa_main.csv``. Ids are subset-qualified
    positional (``'<subset>_<local_idx>'``, e.g. ``diamond_0``) because the
    1192-row space has only 546 unique questions (diamond subset of main subset
    of extended), so content-hash or Record-ID ids would collide. They match the
    ``gpqa_uid`` column of ``zipbench.adapters.gpqa.GPQAZipDataset`` (which
    imports the same ``GPQA_SPLITS`` constant used here) and the natural
    concatenation order of the anchor index space, so no anchor->natural remap
    is needed. Returns the 2-tuple ``(ids, fingerprint)`` (natural order;
    ``anchor_order`` stays None).
    """
    import csv

    from zipbench.adapters.gpqa import GPQA_SPLITS
    base = osp.expanduser(raw_path)
    ids = []
    for subset, csv_name in GPQA_SPLITS:
        n = 0
        with open(osp.join(base, csv_name), 'r', encoding='utf-8') as f:
            for row in csv.reader(f, delimiter=','):
                # Skip the header row exactly as GPQADataset.load does.
                if not row or row[7] == 'Question':
                    continue
                ids.append(f'{subset}_{n}')
                n += 1
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate gpqa ids in {raw_path}')
    if len(ids) != 1192:
        raise ValueError(
            f'expected 1192 GPQA rows (198+546+448), got {len(ids)}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_scibench_ids(raw_path):
    """Enumerate SciBench question ids over the concatenated 10 subjects.

    ``raw_path`` is the ``data/scibench`` *directory* holding one
    ``<subject>.json`` per subject. Ids are positional
    (``'<subject>-<local_idx>'``) because content hashes collide (583
    questions, only 577 unique question+answer pairs); they match the
    ``_id`` column of ``zipbench.adapters.scibench.ScibenchZipDataset``.

    Returns ``(ids, anchor_order, fingerprint)``. The anchor indices
    are rows of a 582-item index space: natural
    concatenation order with the single empty-``answer_number`` question
    (fund[24], natural index 309) removed (items whose reference yields no
    number are excluded). ``anchor_order`` is therefore ``ids`` minus that
    one question, and the converter remaps anchor positions into
    natural-order spec indices.
    """
    from zipbench.adapters.scibench import SCIBENCH_SUBJECTS
    ids, empty_answer_ids = [], []
    for subject in SCIBENCH_SUBJECTS:
        with open(osp.join(osp.expanduser(raw_path), f'{subject}.json'),
                  'r', encoding='utf-8') as f:
            raw = json.load(f)
        for local_idx, entry in enumerate(raw):
            qid = f'{subject}-{local_idx}'
            ids.append(qid)
            if not str(entry.get('answer_number', '')).strip():
                empty_answer_ids.append(qid)
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if empty_answer_ids != ['fund-24']:
        raise ValueError(
            'expected exactly one empty-answer question (fund-24, the item '
            f'the anchor index space drops), found: {empty_answer_ids}')
    anchor_order = [qid for qid in ids if qid not in set(empty_answer_ids)]
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, anchor_order, fingerprint


def _load_bbh_ids(raw_path):
    """Enumerate BBH question ids over the concatenated 24 leaderboard tasks.

    ``raw_path`` is the ``data/BBH/data`` *directory* holding one
    ``<task>.json`` per task (each ``{"examples": [{"input", "target"}, ...]}``).
    Ids are positional (``'<task>-<local_idx>'``), matching the ``_id`` column of
    :class:`zipbench.adapters.bbh.BBHAllDataset` (content hashes could collide --
    e.g. boolean_expressions repeats short expressions). Enumeration order is
    ``sorted()`` of the 24 tasks, each task in its json ``examples`` order,
    giving the 5761-row space (21x250 + causal_judgement 187 +
    penguins_in_a_table 146 + snarks 178).

    Returns the 2-tuple ``(ids, fingerprint)``. Order is NATURAL: the anchor
    ``indices`` index the 5761-row space in exactly this order, so no
    anchor->natural remap is needed (``anchor_order`` stays None).
    """
    from zipbench.adapters.bbh_tasks import LEADERBOARD_24_TASKS
    base = osp.expanduser(raw_path)
    ids = []
    for task in LEADERBOARD_24_TASKS:
        with open(osp.join(base, f'{task}.json'), 'r', encoding='utf-8') as f:
            examples = json.load(f)['examples']
        for local_idx in range(len(examples)):
            ids.append(f'{task}-{local_idx}')
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate bbh ids in {raw_path}')
    if len(ids) != 5761:
        raise ValueError(f'expected 5761 BBH rows, got {len(ids)}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_mmlu_ids(raw_path):
    """Enumerate MMLU question ids over the concatenated 57 subjects.

    ``raw_path`` is the ``data/mmlu`` *directory*; the per-subject test CSVs are
    at ``{raw}/test/{subject}_test.csv`` (no header, six columns
    input/A/B/C/D/target, with embedded newlines inside quoted fields -- a csv
    parser is mandatory). Ids are positional (``'<subject>-<local_idx>'``)
    because MMLU rows carry no id column; they match the ``_id`` column of
    :class:`zipbench.adapters.mmlu.MMLUAllDataset` (which imports the same
    ``MMLU_SUBJECTS`` constant used here). Enumeration order is ``sorted()`` of
    the 57 subjects, each subject in its CSV row order, giving the 14042-row
    space.

    Returns the 2-tuple ``(ids, fingerprint)``. Order is NATURAL: the anchor
    ``indices`` index the 14042-row space in exactly this order, so no
    anchor->natural remap is needed (``anchor_order`` stays None).
    """
    import csv

    from zipbench.adapters.mmlu_tasks import MMLU_SUBJECTS
    base = osp.expanduser(raw_path)
    ids = []
    for subject in MMLU_SUBJECTS:
        fn = osp.join(base, 'test', f'{subject}_test.csv')
        with open(fn, encoding='utf-8') as f:
            for local_idx, row in enumerate(csv.reader(f)):
                # Match MMLUDataset.load's parse exactly (six columns per row).
                if len(row) != 6:
                    raise ValueError(
                        f'{fn}:{local_idx}: expected 6 columns, got {len(row)}')
                ids.append(f'{subject}-{local_idx}')
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate mmlu ids in {raw_path}')
    if len(ids) != 14042:
        raise ValueError(f'expected 14042 MMLU rows, got {len(ids)}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_musr_ids(raw_path):
    """Enumerate MuSR question ids over the concatenated 3 scenarios.

    ``raw_path`` is the ``data/musr`` *directory*; each scenario file is
    ``{raw}/{scenario}.json`` (a list of stories, each ``{'context',
    'questions': [...]}``). Ids are positional (``'<scenario>-<local_idx>'``,
    example-major then question-minor) because MuSR questions carry no id
    column; they match the ``_id`` column of
    :class:`zipbench.adapters.musr.MusrAllDataset` (which imports the same
    ``MUSR_SCENARIOS`` constant used here). Enumeration order is
    ``MUSR_SCENARIOS`` = [murder_mysteries, object_placements, team_allocation],
    each scenario in its natural (story, question) order (one row per question;
    the cot+ flatten keeps all questions), giving the 756-row space
    (250 + 256 + 250; object_placements is 64 stories x 4 questions).

    Returns the 2-tuple ``(ids, fingerprint)``. Order is NATURAL: the anchor
    ``indices`` index the 756-row space in exactly this order, so no
    anchor->natural remap is needed (``anchor_order`` stays None).
    """
    from zipbench.adapters.musr_tasks import MUSR_SCENARIOS
    base = osp.expanduser(raw_path)
    ids = []
    for scenario in MUSR_SCENARIOS:
        with open(osp.join(base, f'{scenario}.json'), 'r',
                  encoding='utf-8') as f:
            stories = json.load(f)
        local_idx = 0
        for story in stories:
            for _question in story['questions']:
                ids.append(f'{scenario}-{local_idx}')
                local_idx += 1
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate musr ids in {raw_path}')
    if len(ids) != 756:
        raise ValueError(
            f'expected 756 MuSR rows (250+256+250), got {len(ids)}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_longbench_ids(raw_path):
    """Enumerate LongBench v1 item ids over the 21 concatenated tasks.

    ``raw_path`` is the ``data/Longbench/data`` *directory* holding one
    ``<task>.jsonl`` per task (the non-``_e`` files, 4750 rows total). Ids are
    LongBench's native per-item ``_id`` (a unique hex string in every jsonl
    row; the upstream per-task loaders drop it, the merged
    :class:`zipbench.adapters.longbench.LongbenchAllDataset` keeps it).
    Enumeration order is ``longbench_tasks.TASK_ORDER`` -- the six
    categories (single_doc_qa, multi_doc_qa, summarization, few_shot,
    synthetic, code) each in official task order, natural jsonl line order
    inside each task -- the category-grouped concatenation the anchor pkls'
    ``indices`` index directly (see the 'longbench' UNIT_PROFILES entry for
    the verification evidence), so no anchor->natural remap is needed
    (``anchor_order`` stays None).
    """
    from zipbench.adapters.longbench_tasks import DATASET_SIZE, TASK_ORDER
    base = osp.expanduser(raw_path)
    ids = []
    for task in TASK_ORDER:
        with open(osp.join(base, f'{task}.jsonl'), 'r',
                  encoding='utf-8') as f:
            for line in f:
                ids.append(str(json.loads(line)['_id']))
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate longbench _id in {raw_path}')
    if len(ids) != DATASET_SIZE:
        raise ValueError(
            f'expected {DATASET_SIZE} LongBench rows, got {len(ids)}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_livecodebench_ids(raw_path):
    """Enumerate LiveCodeBench question_ids in the runner's sorted order.

    ``raw_path`` is the release version tag (e.g. ``release_v6``), not a file:
    the dataset is resolved through the upstream lcb_runner loader (HF
    ``livecodebench/code_generation_lite``), exactly as
    ``opencompass.datasets.LCBOfficialCodeGenerationDataset`` does at eval
    time. The stable id is the native ``question_id`` (globally unique).

    Returns the 2-tuple ``(ids, fingerprint)``. Order is NATURAL: the loader
    sorts problems by question_id (the upstream runner order), and the anchor
    ``indices`` index that same sorted order, so no anchor->natural remap is
    needed (``anchor_order`` stays None).
    """
    from opencompass.datasets.livecodebench.official import \
        LCBOfficialCodeGenerationDataset
    release = raw_path if raw_path.startswith('release') else 'release_v6'
    ds = LCBOfficialCodeGenerationDataset.load(release_version=release)
    ids = [str(q) for q in ds['test']['question_id']]
    if not ids:
        raise ValueError(f'No problems loaded for {release}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate question_id values in {release}')
    if ids != sorted(ids):
        raise ValueError('LiveCodeBench loader order is not sorted by '
                         'question_id; anchor index space would not match')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_mmlu_pro_ids(raw_path):
    """Enumerate MMLU-Pro row ids over the native HF test split (12032 rows).

    ``raw_path`` is the ``mmlu_pro`` data *directory* (holding the HF
    ``test-*.parquet``); we ``load_dataset`` it and take the ``test`` split in
    native order (no per-category filter/reorder), exactly matching
    :class:`zipbench.adapters.mmlu_pro.MMLUProAllDataset`. The stable id is the
    native ``question_id`` (globally unique and monotonic across the split), so
    it matches the loader's ``id_field='question_id'`` from a single source of
    truth.

    Returns the 2-tuple ``(ids, fingerprint)``. Order is NATURAL: the anchor
    ``indices`` index the 12032-row space in this exact order --
    native parquet == Open-LLM-Leaderboard ``leaderboard_mmlu_pro`` doc_id
    order -- so no anchor->natural remap is needed
    (``anchor_order`` stays None).
    """
    from datasets import load_dataset

    test = load_dataset(osp.expanduser(raw_path))['test']
    ids = [str(q) for q in test['question_id']]
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate mmlu_pro question_id in {raw_path}')
    if len(ids) != 12032:
        raise ValueError(
            f'expected 12032 MMLU-Pro test rows, got {len(ids)}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _load_ifeval_ids(raw_path):
    """Enumerate IFEval row ids in natural ``input_data.jsonl`` order.

    ``raw_path`` is ``data/ifeval/input_data.jsonl`` (541 records, each
    ``{key, prompt, instruction_id_list, kwargs}``). The official ``key`` is
    unique and is used stringified as the stable spec id.
    ``opencompass.datasets.IFEvalDataset`` reads the same file line by line, so
    ``ids[i]`` names the row an anchor index ``i`` selects. The keys are not
    monotonic in file order, so they must be read positionally and never
    sorted. The generic ``_load_raw_ids`` cannot be used here: it hashes
    ``question`` + ``answer``, neither of which exists in these records, so
    every row would collapse onto the same content-hash id.

    Returns the 2-tuple ``(ids, fingerprint)`` -- the anchor pkls index this
    natural order directly (see the ``ifeval`` UNIT_PROFILES entry), so no
    ``anchor_order`` remap is emitted.
    """
    ids = []
    with open(osp.expanduser(raw_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)['key']))
    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'duplicate IFEval key values in {raw_path}')
    if len(ids) != 541:
        raise ValueError(f'expected 541 IFEval rows, got {len(ids)}')
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _rec_id(rec):
    """Extract a stable id from one record, falling back to a content hash."""
    rid = None
    if isinstance(rec, dict):
        rid = rec.get('_id', rec.get('id'))
    if rid is None:
        basis = (str(rec.get('question', '')) + '\x1f' +
                 str(rec.get('answer', ''))) if isinstance(rec, dict) \
            else repr(rec)
        rid = hashlib.sha1(basis.encode('utf-8')).hexdigest()[:16]
    return str(rid)


def _load_raw_ids(raw_path, chunk_size=1 << 22, progress_every=50):
    """Return the list of stable ids (and a fingerprint) for the *full*
    dataset, in natural file order so that ``ids[i]`` corresponds to index
    ``i`` (0-based).

    The raw file is a JSON *array* whose ``context`` fields can be enormous
    (LongBench v2 documents). Rather than ``json.load`` the whole thing -- which
    holds every record in memory at once and blocks in a single, uninterruptible
    read syscall (this can wedge a slow networked filesystem) -- we decode the
    array **incrementally**, one record at a time, using only the standard
    library:

      * memory stays bounded to roughly one record + one chunk;
      * the process yields between records and reads in ``chunk_size`` blocks,
        so it stays responsive to signals / ``timeout``;
      * each record is fully parsed, so id extraction is exact (no regex
        false-positives from ``"_id":`` appearing inside a context string).
    """
    dec = json.JSONDecoder()
    ids = []
    # JSON Lines (one object per line, e.g. commonsenseqa's
    # dev_rand_split.jsonl): the first non-whitespace character is '{', not
    # the '[' of a top-level array.
    with open(raw_path, 'r', encoding='utf-8') as f:
        head = f.read(64)
        first = head.lstrip()[:1]
    if first == '{':
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids.append(_rec_id(json.loads(line)))
                if progress_every and len(ids) % progress_every == 0:
                    print(f'  ... parsed {len(ids)} records', flush=True)
        if not ids:
            raise ValueError(f'No records parsed from {raw_path}')
        fingerprint = hashlib.sha256(
            '\n'.join(ids).encode('utf-8')).hexdigest()
        return ids, fingerprint

    with open(raw_path, 'r', encoding='utf-8') as f:
        buf = ''
        # Advance to the opening bracket of the top-level array.
        while '[' not in buf:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f'No JSON array found in {raw_path}')
            buf += chunk
        buf = buf[buf.index('[') + 1:]

        while True:
            buf = buf.lstrip()
            while buf[:1] == ',':
                buf = buf[1:].lstrip()
            if buf[:1] == ']':
                break
            if buf == '':
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buf += chunk
                continue
            try:
                obj, end = dec.raw_decode(buf)
            except json.JSONDecodeError:
                # Current record is not fully buffered yet -- read more.
                chunk = f.read(chunk_size)
                if not chunk:
                    raise
                buf += chunk
                continue
            ids.append(_rec_id(obj))
            buf = buf[end:]
            if progress_every and len(ids) % progress_every == 0:
                print(f'  ... parsed {len(ids)} records', flush=True)

    if not ids:
        raise ValueError(f'No records parsed from {raw_path}')

    # Fingerprint captures both identity and ordering of the full dataset
    # without hashing the (very large) contexts.
    fingerprint = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    return ids, fingerprint


def _parse_ratio(pkl_path):
    """Best-effort extraction of ``ratio`` from the anchor filename
    e.g. ``anchor_..._ratio_0.7495.pkl``."""
    name = osp.basename(pkl_path)
    ratio = None
    # Match a decimal number explicitly so the trailing '.' in '0.7495.pkl' is
    # not swallowed into the captured group.
    m = re.search(r'ratio_(\d+\.\d+)', name)
    if m:
        ratio = float(m.group(1))
    return ratio


def _load_anchor(pkl_path, dataset_key):
    with open(pkl_path, 'rb') as f:
        anchor = pickle.load(f)
    if not isinstance(anchor, dict):
        raise ValueError(
            f'Anchor pkl must be a dict, got {type(anchor).__name__}')
    if dataset_key not in anchor:
        raise KeyError(
            f'Dataset key {dataset_key!r} not found in anchor pkl. '
            f'Available keys: {list(anchor.keys())}')
    entry = anchor[dataset_key]
    indices = np.asarray(entry['indices']).astype(int).ravel()
    weights = np.asarray(entry['weights']).astype(float).ravel()
    return indices, weights


def _validate(indices, weights, n_total):
    if len(indices) != len(weights):
        raise ValueError(
            f'indices ({len(indices)}) and weights ({len(weights)}) '
            'have different lengths')
    if len(indices) == 0:
        raise ValueError('subset is empty')
    if len(set(indices.tolist())) != len(indices):
        raise ValueError('indices contain duplicates')
    lo, hi = int(indices.min()), int(indices.max())
    if lo < 0 or hi >= n_total:
        raise ValueError(
            f'indices out of range: [{lo}, {hi}] not within [0, {n_total})')
    if (weights < 0).any():
        raise ValueError('weights contain negative values')
    w_sum = float(weights.sum())
    return w_sum


def _write_jsonl(out_path, ids, indices, weights):
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, w in zip(indices.tolist(), weights.tolist()):
            rec = {'id': ids[idx], 'index': int(idx), 'weight': float(w)}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _update_manifest(manifest_path, dataset_key, fingerprint, n_total,
                     subset_name, subset_file, n_subset, ratio, w_sum,
                     breakdown_fields, selection_unit, dataset_type,
                     loader_kwargs, evaluator_type, merge_base_kwargs,
                     inject_subset_spec, dropped_ids=None):
    import yaml  # local import; PyYAML ships with opencompass/mmengine

    if osp.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f) or {}
    else:
        manifest = {}

    manifest['dataset_key'] = dataset_key
    manifest['dataset_size'] = int(n_total)
    manifest['dataset_fingerprint'] = fingerprint
    # ``selection_unit`` documents what one anchor ``index`` selects: 'row' for
    # exact-match benchmarks, or a benchmark-specific unit (e.g. 'subproblem').
    manifest['selection_unit'] = selection_unit
    manifest['breakdown_fields'] = list(breakdown_fields)
    # How to rewire the dataset / evaluator onto a subset at eval time; read by
    # ``zipbench.apply.resolve_wiring`` and the inline config block. See
    # ``resolve_wiring`` for the schema + defaults.
    manifest['dataset'] = {
        'type': dataset_type,
        'loader_kwargs': dict(loader_kwargs),
    }
    manifest['evaluator'] = {
        'type': evaluator_type,
        'merge_base_kwargs': bool(merge_base_kwargs),
        'inject_subset_spec': bool(inject_subset_spec),
    }
    manifest['weighting'] = WEIGHTING_FORMULA
    subsets = manifest.setdefault('subsets', {})
    subsets[subset_name] = {
        'file': subset_file,
        'n': int(n_subset),
        'ratio': round(float(ratio), 4) if ratio is not None else None,
        'weight_sum': round(float(w_sum), 6),
    }
    if dropped_ids:
        # Items removed after the anchor was built (e.g. deleted upstream).
        subsets[subset_name]['dropped_ids'] = sorted(dropped_ids)

    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)

    # Also emit a JSON sibling so lazy-import OpenCompass configs can read the
    # manifest with the standard library only (no PyYAML, which a lazy config
    # cannot import without turning it into an uncallable LazyObject).
    json_path = osp.splitext(manifest_path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkl', required=True, help='anchor .pkl file')
    parser.add_argument('--raw', required=True,
                        help='full dataset json (rows mode: list of records '
                        'with _id; scicode mode: a SciCode_datasets*.json)')
    parser.add_argument('--dataset-key', default='LongBenchv2_0shot',
                        help='key inside the anchor pkl / spec namespace')
    parser.add_argument('--name', required=True,
                        help="subset version name, e.g. 'tiny' or 'small'")
    parser.add_argument('--out-dir', required=True,
                        help='output directory for the jsonl + manifest')
    parser.add_argument(
        '--units-from', default='rows', choices=sorted(UNIT_PROFILES),
        help="how an anchor 'index' maps to a selection unit / stable id: "
        "'rows' (one per dataset row, exact-match) or 'scicode' (one per "
        'generated sub-step). Also picks the default loader/evaluator profile.')
    parser.add_argument(
        '--base-loader', default='opencompass.datasets.LongBenchv2Dataset',
        help='(rows mode) fully-qualified loader used to rebuild the full '
        'dataset at eval time (e.g. opencompass.datasets.LongBenchv2Dataset)')
    parser.add_argument(
        '--id-field', default='_id',
        help="(rows mode) the dataset's stable per-question id field")
    parser.add_argument(
        '--breakdown-fields', default='difficulty,length',
        help='comma-separated metadata fields for per-group weighted accuracy '
        "(empty string for none)")
    # Optional overrides of the ``--units-from`` profile (default None -> use
    # the profile value); these populate the manifest's dataset/evaluator
    # wiring sections consumed by ``zipbench.apply.resolve_wiring``.
    parser.add_argument('--dataset-type', default=None,
                        help='override the zipbench loader type string')
    parser.add_argument('--evaluator-type', default=None,
                        help='override the zipbench evaluator type string')
    parser.add_argument('--selection-unit', default=None,
                        help="override the documented selection unit label")
    parser.add_argument('--loader-kwargs', default=None,
                        help='JSON dict of extra loader ctor kwargs '
                        '(rows mode defaults to {base_loader, id_field})')
    parser.add_argument('--merge-base-kwargs', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='merge the evaluator onto the original eval_cfg '
                        'evaluator dict instead of replacing it')
    parser.add_argument('--inject-subset-spec', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='pass subset_spec= to the evaluator too')
    parser.add_argument('--renormalize', action='store_true',
                        help='renormalize weights to sum to 1 if they do not')
    parser.add_argument('--ratio', type=float, default=None,
                        help='override ratio in the manifest when the pkl '
                        'filename lacks it. ratio is the COMPRESSION rate '
                        '1 - n_subset/n_total (the fraction removed), matching '
                        'the convention encoded in the '
                        'ratio_X.XXXX filenames -- NOT the retained fraction '
                        'n_subset/n_total; default parses it from the filename')
    parser.add_argument(
        '--drop-ids', default='',
        help='comma-separated stable ids to remove from the subset after '
        'anchor->natural mapping (e.g. items deleted from the upstream '
        'dataset since the anchor was built); remaining weights are '
        'renormalized to sum to 1')
    args = parser.parse_args()

    profile = UNIT_PROFILES[args.units_from]
    dataset_type = args.dataset_type or profile['dataset_type']
    evaluator_type = args.evaluator_type or profile['evaluator_type']
    selection_unit = args.selection_unit or profile['selection_unit']
    merge_base_kwargs = (profile['merge_base_kwargs']
                         if args.merge_base_kwargs is None
                         else args.merge_base_kwargs)
    inject_subset_spec = (profile['inject_subset_spec']
                          if args.inject_subset_spec is None
                          else args.inject_subset_spec)
    if args.loader_kwargs is not None:
        loader_kwargs = json.loads(args.loader_kwargs)
    elif args.units_from == 'rows':
        loader_kwargs = {'base_loader': args.base_loader,
                         'id_field': args.id_field}
    elif args.units_from == 'simpleqa':
        # No id column in the live dataset -> content-hash ids in the spec are
        # for offline provenance only; validate_and_select selects by index.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.SimpleQADataset'}
    elif args.units_from == 'c3':
        # C3Dataset_V2 rows carry no id column either -> spec ids are
        # provenance only; selection is by (flattened) row index.
        loader_kwargs = {'base_loader': 'opencompass.datasets.C3Dataset_V2'}
    elif args.units_from == 'theoremqa':
        # The dataset keeps the raw 'id' column, so drift checking is on.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.TheoremQADatasetV3',
                         'id_field': 'id'}
    elif args.units_from == 'scibench':
        # The concatenating loader synthesises a unique positional '_id'
        # column ('<subject>-<local_idx>'), so drift checking is on.
        loader_kwargs = {'base_loader':
                         'zipbench.adapters.scibench.ScibenchZipDataset',
                         'id_field': '_id'}
    elif args.units_from == 'humaneval_plus':
        # The dataset keeps the raw 'task_id' column, so drift checking is on.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.HumanevalDataset',
                         'id_field': 'task_id'}
    elif args.units_from == 'mbpp_plus':
        # MBPPPlusDataset returns a flat Dataset (no separate 'train'
        # split), so ZipSubsetDataset selects by index and the stable-id drift
        # check is skipped at runtime; 'task_id' is kept for offline
        # provenance.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.MBPPPlusDataset',
                         'id_field': 'task_id'}
    elif args.units_from == 'arc_challenge':
        # ARCDatasetAllChoices returns a flat Dataset (no separate 'train'
        # split), so ZipSubsetDataset selects by index and the stable-id drift
        # check is skipped at runtime; 'id' is kept for offline provenance.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.ARCDatasetAllChoices',
                         'id_field': 'id'}
    elif args.units_from == 'gsm8k':
        # GSM8KDataset returns a DatasetDict{train,test} whose rows have no id
        # column, so ZipSubsetDataset selects the test split by index and the
        # stable-id drift check is skipped at runtime; '_id' is inert here and
        # the content-hash spec ids are for offline provenance only.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.GSM8KDataset',
                         'id_field': '_id'}
    elif args.units_from == 'math':
        # MATHDataset returns a DatasetDict{train,test} whose rows have no id
        # column, so ZipSubsetDataset selects the test split by index and the
        # stable-id drift check is skipped at runtime; '_id' is inert here and
        # the content-hash spec ids are for offline provenance only. The full
        # 1324-item MATH-Hard set is loaded from data/math_hard/math_hard.json
        # (path/file_name are set on the dataset dict in the eval config).
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.MATHDataset',
                         'id_field': '_id'}
    elif args.units_from == 'hellaswag':
        # HellaswagDataset_V2 returns a flat Dataset (no separate 'train'
        # split), so ZipSubsetDataset selects by index and the stable-id drift
        # check is skipped at runtime; '_id' is inert here and the content-hash
        # spec ids are for offline provenance only.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.HellaswagDataset_V2',
                         'id_field': '_id'}
    elif args.units_from == 'winogrande':
        # WinograndeDatasetV3 returns a DatasetDict{train_xs, dev}; the inference
        # split is 'dev' (not 'test'), so record test_split='dev' -> the
        # ZipSubsetDataset Branch-B path subsets only 'dev' and keeps 'train_xs'
        # full for the 5-shot FixKRetriever. That path builds a work dict with no
        # 'train' key, so the stable-id drift check is skipped at runtime; the
        # qID spec ids are for offline provenance only.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.WinograndeDatasetV3',
                         'test_split': 'dev'}
    elif args.units_from == 'truthfulqa':
        # TruthfulQADataset returns a DatasetDict with only the 'validation'
        # split (no 'train'), so record test_split='validation' -> the
        # ZipSubsetDataset Branch-B path subsets only 'validation'. That path
        # builds a work dict with no 'train' key, so the stable-id drift check
        # is skipped at runtime; the content-hash spec ids are provenance only.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.TruthfulQADataset',
                         'test_split': 'validation'}
    elif args.units_from == 'gpqa':
        # GPQAZipDataset concatenates the 3 csvs and synthesises a unique
        # positional 'gpqa_uid' column ('<subset>_<local_idx>'), returning a
        # DatasetDict{train,test} both carrying it, so ZipSubsetDataset's
        # stable-id drift check is ACTIVE at runtime.
        loader_kwargs = {'base_loader':
                         'zipbench.adapters.gpqa.GPQAZipDataset',
                         'id_field': 'gpqa_uid'}
    elif args.units_from == 'bbh':
        # BBHAllDataset returns a flat Dataset (no separate 'train' split), so
        # ZipSubsetDataset selects by index and the stable-id drift check is
        # skipped at runtime; '_id' is kept for offline provenance.
        loader_kwargs = {'base_loader':
                         'zipbench.adapters.bbh.BBHAllDataset',
                         'id_field': '_id'}
    elif args.units_from == 'mmlu':
        # MMLUAllDataset concatenates the 57 subjects and synthesises a unique
        # positional '_id' column ('<subject>-<local_idx>'), returning a
        # DatasetDict{train,test} both carrying it, so ZipSubsetDataset's
        # stable-id drift check is ACTIVE at runtime.
        loader_kwargs = {'base_loader':
                         'zipbench.adapters.mmlu.MMLUAllDataset',
                         'id_field': '_id'}
    elif args.units_from == 'mmlu_pro':
        # MMLUProAllDataset loads the test split in native HF/parquet order (no
        # per-category filter) and returns a DatasetDict{train,test} both
        # carrying the native, globally-unique 'question_id', so
        # ZipSubsetDataset's stable-id drift check is ACTIVE at runtime.
        loader_kwargs = {'base_loader':
                         'zipbench.adapters.mmlu_pro.MMLUProAllDataset',
                         'id_field': 'question_id'}
    elif args.units_from == 'musr':
        # MusrAllDataset concatenates the 3 scenarios and synthesises a unique
        # positional '_id' column ('<scenario>-<local_idx>'), returning a
        # DatasetDict{train,test} both carrying it, so ZipSubsetDataset's
        # stable-id drift check is ACTIVE at runtime.
        loader_kwargs = {'base_loader':
                         'zipbench.adapters.musr.MusrAllDataset',
                         'id_field': '_id'}
    elif args.units_from == 'longbench':
        # LongbenchAllDataset concatenates the 21 tasks (category-grouped
        # anchor order) keeping LongBench's native per-item '_id' (a unique
        # hex id in the raw jsonl, dropped by the upstream per-task loaders),
        # returning a DatasetDict{train,test} both carrying it, so
        # ZipSubsetDataset's stable-id drift check is ACTIVE at runtime.
        loader_kwargs = {'base_loader':
                         'zipbench.adapters.longbench.LongbenchAllDataset',
                         'id_field': '_id'}
    elif args.units_from == 'livecodebench':
        # LCBOfficialCodeGenerationDataset returns a DatasetDict{test,train}
        # (same Dataset twice) carrying the native, globally-unique
        # 'question_id', so ZipSubsetDataset's stable-id drift check is ACTIVE
        # at runtime. release_version pins the same 1055-problem set the
        # anchor index space was built over.
        loader_kwargs = {
            'base_loader':
            'opencompass.datasets.LCBOfficialCodeGenerationDataset',
            'id_field': 'question_id',
            'release_version': 'release_v6'}
    elif args.units_from == 'ifeval':
        # IFEvalDataset returns a flat Dataset (no separate 'train' split) whose
        # only columns are 'prompt' and the nested 'reference' dict -- the
        # official 'key' lives inside that dict, not as a top-level column -- so
        # ZipSubsetDataset selects by index and the stable-id drift check is
        # SKIPPED at runtime. The str(key) spec ids are offline provenance,
        # covered by the manifest's dataset_fingerprint. No id_field is recorded
        # (as for the 'simpleqa' / 'c3' profiles).
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.IFEvalDataset'}
    elif args.units_from == 'arenahard':
        # ArenaHardDataset returns a flat Dataset whose 'question_id' lives
        # only inside the nested 'judge' dict (no top-level id column), so
        # ZipSubsetDataset selects by index and the stable-id drift check is
        # SKIPPED at runtime. The question_id spec ids are offline provenance
        # and the join key the weighted dict_postprocessor uses to group the
        # two judged games per question.
        loader_kwargs = {'base_loader':
                         'opencompass.datasets.ArenaHardDataset'}
    else:
        loader_kwargs = {}

    breakdown_fields = [f.strip() for f in args.breakdown_fields.split(',')
                        if f.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    anchor_order = None  # anchor pkl item order, when it differs from `ids`
    if args.units_from == 'scicode':
        ids, anchor_order, fingerprint = _load_scicode_ids(args.raw)
    elif args.units_from == 'simpleqa':
        ids, fingerprint = _load_simpleqa_ids(args.raw)
    elif args.units_from == 'c3':
        ids, fingerprint = _load_c3_ids(args.raw)
    elif args.units_from == 'theoremqa':
        ids, anchor_order, fingerprint = _load_theoremqa_ids(args.raw)
    elif args.units_from == 'scibench':
        ids, anchor_order, fingerprint = _load_scibench_ids(args.raw)
    elif args.units_from == 'humaneval_plus':
        ids, anchor_order, fingerprint = _load_humaneval_plus_ids(args.raw)
    elif args.units_from == 'mbpp_plus':
        ids, fingerprint = _load_mbpp_plus_ids(args.raw)
    elif args.units_from == 'arc_challenge':
        ids, anchor_order, fingerprint = _load_arc_challenge_ids(args.raw)
    elif args.units_from == 'gsm8k':
        ids, anchor_order, fingerprint = _load_gsm8k_ids(args.raw)
    elif args.units_from == 'math':
        ids, fingerprint = _load_math_ids(args.raw)
    elif args.units_from == 'hellaswag':
        ids, anchor_order, fingerprint = _load_hellaswag_ids(args.raw)
    elif args.units_from == 'winogrande':
        ids, anchor_order, fingerprint = _load_winogrande_ids(args.raw)
    elif args.units_from == 'truthfulqa':
        ids, anchor_order, fingerprint = _load_truthfulqa_ids(args.raw)
    elif args.units_from == 'gpqa':
        ids, fingerprint = _load_gpqa_ids(args.raw)
    elif args.units_from == 'bbh':
        ids, fingerprint = _load_bbh_ids(args.raw)
    elif args.units_from == 'mmlu':
        ids, fingerprint = _load_mmlu_ids(args.raw)
    elif args.units_from == 'mmlu_pro':
        ids, fingerprint = _load_mmlu_pro_ids(args.raw)
    elif args.units_from == 'musr':
        ids, fingerprint = _load_musr_ids(args.raw)
    elif args.units_from == 'livecodebench':
        ids, fingerprint = _load_livecodebench_ids(args.raw)
    elif args.units_from == 'ifeval':
        ids, fingerprint = _load_ifeval_ids(args.raw)
    elif args.units_from == 'longbench':
        ids, fingerprint = _load_longbench_ids(args.raw)
    elif args.units_from == 'arenahard':
        ids, fingerprint = _load_arenahard_ids(args.raw)
    else:
        ids, fingerprint = _load_raw_ids(args.raw)
    n_total = len(ids)
    indices, weights = _load_anchor(args.pkl, args.dataset_key)
    if len(np.unique(indices)) != len(indices):
        # Some anchor pkls list the same item more than once (the miner can
        # re-pick an anchor). Selecting a row twice is equivalent to selecting
        # it once with the weights summed, so merge -- the weighted estimator
        # is unchanged.
        n_before = len(indices)
        order = np.argsort(indices, kind='stable')
        uniq, inverse = np.unique(indices[order], return_inverse=True)
        merged_w = np.zeros(len(uniq), dtype=float)
        np.add.at(merged_w, inverse, weights[order])
        # Keep the original anchor ordering of first occurrences.
        first_pos = {int(i): p for p, i in
                     reversed(list(enumerate(indices.tolist())))}
        keep = np.argsort([first_pos[int(i)] for i in uniq], kind='stable')
        indices, weights = uniq[keep], merged_w[keep]
        print(f'[warn] anchor pkl contains duplicate indices; merged '
              f'{n_before} -> {len(indices)} entries (weights summed)')
    w_sum = _validate(indices, weights, n_total)
    if anchor_order is not None:
        # The anchor's `indices` are positions in the anchor item
        # order (`anchor_order`), not in dataset natural order; translate them
        # into natural-order spec indices (see _load_scicode_ids).
        nat_pos = {sid: i for i, sid in enumerate(ids)}
        indices = np.array(
            [nat_pos[anchor_order[i]] for i in indices.tolist()])

    drop_ids = {s.strip() for s in args.drop_ids.split(',') if s.strip()}
    if drop_ids:
        unknown = drop_ids - {ids[i] for i in indices.tolist()}
        if unknown:
            raise ValueError(
                f'--drop-ids not present in the subset: {sorted(unknown)}')
        keep = np.array([ids[i] not in drop_ids for i in indices.tolist()])
        n_before = len(indices)
        indices, weights = indices[keep], weights[keep]
        weights = weights / weights.sum()
        w_sum = float(weights.sum())
        print(f'[warn] dropped {n_before - len(indices)} item(s) '
              f'({sorted(drop_ids)}); weights renormalized')

    if abs(w_sum - 1.0) > 1e-6:
        if args.renormalize:
            weights = weights / w_sum
            print(f'[warn] weights summed to {w_sum:.6f}; renormalized to 1.0')
            w_sum = 1.0
        else:
            print(f'[warn] weights sum to {w_sum:.6f} (not 1.0); '
                  'pass --renormalize to fix')

    jsonl_name = f'{args.name}.jsonl'
    jsonl_path = osp.join(args.out_dir, jsonl_name)
    _write_jsonl(jsonl_path, ids, indices, weights)

    ratio = _parse_ratio(args.pkl)
    if args.ratio is not None:
        ratio = args.ratio
    manifest_path = osp.join(args.out_dir, 'manifest.yaml')
    _update_manifest(manifest_path, args.dataset_key, fingerprint, n_total,
                     args.name, jsonl_name, len(indices), ratio, w_sum,
                     breakdown_fields, selection_unit, dataset_type,
                     loader_kwargs, evaluator_type, merge_base_kwargs,
                     inject_subset_spec, dropped_ids=drop_ids)

    print(f'[ok] wrote {jsonl_path}  (n={len(indices)}, '
          f'index range=[{int(indices.min())},{int(indices.max())}], '
          f'weight_sum={w_sum:.6f})')
    print(f'[ok] updated {manifest_path}  '
          f'(dataset_size={n_total}, fingerprint={fingerprint[:12]}...)')


if __name__ == '__main__':
    main()
