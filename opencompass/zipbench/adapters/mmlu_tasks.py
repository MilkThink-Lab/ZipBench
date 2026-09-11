"""MMLU subject-name constants, importable without the heavy ``opencompass`` chain.

``zipbench.adapters.mmlu`` needs ``opencompass`` at import time (base class +
registry decorator + ``MMLUDataset``), and importing that package is very slow on
this cluster (~200s cold). The offline converter
(``convert_anchor_to_spec._load_mmlu_ids``) and verify scripts only need the
subject *names*, so those live here in a pure standard-library module and
``mmlu.py`` re-exports them -- single source of truth, no opencompass import for
offline id enumeration.

The 57 subjects are the full MMLU set (copied verbatim from
``opencompass/configs/datasets/mmlu/mmlu_all_sets.py``); ``sorted()`` gives the
fixed enumeration order of the flat 14042-row index space that the anchor
indices refer to.
"""

# Copied verbatim from opencompass/configs/datasets/mmlu/mmlu_all_sets.py (in
# that file's order for easy diffing); sorted() below fixes the enumeration.
_MMLU_SUBJECTS_RAW = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]

# Fixed enumeration order of the flat 14042-row index space = sorted() of the 57
# subjects, each subject in its ``<subject>_test.csv`` natural row order.
MMLU_SUBJECTS = sorted(_MMLU_SUBJECTS_RAW)

assert len(MMLU_SUBJECTS) == 57, \
    f'expected 57 MMLU subjects, got {len(MMLU_SUBJECTS)}'
assert len(set(MMLU_SUBJECTS)) == 57, 'duplicate MMLU subject names'
