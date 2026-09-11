"""BBH task-name constants, importable without the heavy ``opencompass`` chain.

``zipbench.adapters.bbh`` needs ``opencompass`` at import time (base classes +
registry decorators), and importing that package is very slow on this cluster
(~200s cold). The offline converter (``convert_anchor_to_spec._load_bbh_ids``)
and verify scripts only need the task *names*, so those live here in a pure
standard-library module and ``bbh.py`` re-exports them -- single source of truth,
no opencompass import for offline id enumeration.

The 24 tasks are the Open-LLM-Leaderboard-v2 BBH set: the official 27 minus
``dyck_languages`` / ``multistep_arithmetic_two`` / ``word_sorting`` (the three
free-form tasks lm-eval could not cast to multiple-choice). Which tasks are
rendered as multiple-choice vs free-form matches ``bbh_leaderboard24_gen`` and
decides which OpenCompass scorer each row uses.
"""

MCQ_TASKS = [
    'temporal_sequences', 'disambiguation_qa', 'date_understanding',
    'tracking_shuffled_objects_three_objects', 'penguins_in_a_table',
    'geometric_shapes', 'snarks', 'ruin_names',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects', 'logical_deduction_three_objects',
    'hyperbaton', 'logical_deduction_five_objects',
    'logical_deduction_seven_objects', 'movie_recommendation',
    'salient_translation_error_detection', 'reasoning_about_colored_objects',
]
FREEFORM_TASKS = [
    'navigate', 'sports_understanding', 'boolean_expressions',
    'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies',
]
# Fixed enumeration order of the flat 5761-row index space = sorted() of all 24
# tasks, each task in its json ``examples`` order.
LEADERBOARD_24_TASKS = sorted(MCQ_TASKS + FREEFORM_TASKS)

# eval_type per task: which of the two OpenCompass BBH scorers a row uses.
EVAL_TYPE = {t: 'mcq' for t in MCQ_TASKS}
EVAL_TYPE.update({t: 'freeform' for t in FREEFORM_TASKS})
