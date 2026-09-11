"""ZipBench adapter for ArenaHard (LLM-judge pairwise scoring, weighted).

ArenaHard selects plain rows, so the dataset side reuses the generic
:class:`zipbench.dataset.ZipSubsetDataset` with
``base_loader='opencompass.datasets.ArenaHardDataset'``. Scoring runs through
the subjective pipeline: every question is judged twice against the
gpt4-0314 baseline (``infer_order='double'``: once with the baseline as
assistant A, once swapped), the judge emits a ``[[A>>B]]``-style verdict, and
a ``dict_postprocessor`` turns the judge outputs into metrics.

This weighted postprocessor scores each question with a soft win rate for the
evaluated model:

  * verdict -> score for assistant A: ``A>>B``=1, ``A>B``=0.75, ``A=B``=0.5,
    ``B>A``=0.25, ``B>>A``=0; mirrored (``1 - s``) when the evaluated model is
    assistant B (``gold['answer1']`` names assistant A);
  * per-question score = mean over the games whose verdict parsed; a question
    with no parseable verdict scores 0;
  * ``score = sum(w_q * score_q) / sum(w_q) * 100``.

Games are grouped by ``gold['question_id']`` (never by position: the judge
output interleaves the two game orders), and question ids are joined against
the subset spec's ``id`` column.

Note this per-question soft win rate is NOT the ArenaHard leaderboard number
(a Bradley-Terry/ELO fit over all battles); no ELO fit is run here.
"""
import re

from ..result import zip_result
from ..spec import load_subset_spec

# Score for *assistant A*; the first [[...]] match wins, mirroring the
# upstream ``post_process_arenahard`` extraction.
_VERDICT_RE = re.compile(r'\[\[([AB<>=]+)\]\]')
_SCORE_FOR_A = {'A>>B': 1.0, 'A>B': 0.75, 'A=B': 0.5, 'B>A': 0.25,
                'B>>A': 0.0}


def arenahard_zip_postprocess(output: dict, output_path: str,
                              subset_spec: str) -> dict:
    """Weighted ArenaHard soft win rate from LM-judge outputs."""
    spec = load_subset_spec(subset_spec)
    if len(output) != 2 * len(spec):
        raise ValueError(
            f'judge output has {len(output)} samples but subset spec '
            f'{subset_spec} has {len(spec)} questions (expected 2 games '
            'each); the evaluated dataset does not match the spec')

    per_q = {r['id']: [] for r in spec}
    n_unparsed = 0
    for key, sample in output.items():
        gold = sample['gold']
        qid = str(gold['question_id'])
        if qid not in per_q:
            raise ValueError(f'judged question_id {qid!r} not in subset spec '
                             f'{subset_spec}')
        match = _VERDICT_RE.search(sample['prediction'])
        if match is None or match.group(1) not in _SCORE_FOR_A:
            sample['game_score'] = None
            n_unparsed += 1
            continue
        s = _SCORE_FOR_A[match.group(1)]
        base_models = gold.get('base_models') or []
        model_is_a = gold['answer1'] not in base_models
        game_score = s if model_is_a else 1.0 - s
        sample['game_score'] = game_score
        per_q[qid].append(game_score)

    bad = [qid for qid, games in per_q.items() if len(games) > 2]
    if bad:
        raise ValueError(f'more than 2 parsed games for question(s) {bad[:3]}')

    w_sum = w_score = u_score = 0.0
    for row in spec:
        games = per_q[row['id']]
        q_score = sum(games) / len(games) if games else 0.0
        w = row['weight']
        w_sum += w
        w_score += w * q_score
        u_score += q_score

    score = w_score / w_sum * 100
    return zip_result(
        score,
        'weighted_soft_win_rate — subset-weighted mean of the per-question '
        'judge soft win rate vs the gpt4-0314 baseline (two games per '
        'question, A>>B=1 .. B>>A=0 from the evaluated model side; '
        'unparseable verdicts are skipped, a question with none scores 0); '
        'NOT the leaderboard Bradley-Terry/ELO win rate',
        {
            'weighted_soft_win_rate': score,
            'soft_win_rate_unweighted': u_score / len(spec) * 100,
            'num_questions': len(spec),
            'num_unparsed_games': n_unparsed,
        },
        output,
    )
