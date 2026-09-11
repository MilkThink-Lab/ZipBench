"""MuSR scenario-name constant, importable without the heavy ``opencompass`` chain.

``zipbench.adapters.musr`` needs ``opencompass`` at import time (base class +
registry decorator + ``MusrDataset``), and importing that package is very slow on
this cluster (~200s cold). The offline converter
(``convert_anchor_to_spec._load_musr_ids``) only needs the scenario *names*, so
those live here in a pure standard-library module and ``musr.py`` re-exports them
-- single source of truth, no opencompass import for offline id enumeration.

The three scenarios are the full MuSR set. Their order is the fixed enumeration
order of the flat 756-row index space that the anchor indices refer to:
``[murder_mysteries, object_placements, team_allocation]``. It also matches
opencompass ``DATASET_CONFIGS`` insertion order, so a natural per-scenario
concat reproduces the anchor item order with no anchor->natural remap.
"""

# Fixed enumeration order of the flat 756-row index space: the three scenarios
# concatenated as [murder_mysteries 0:250, object_placements 250:506,
# team_allocation 506:756], each scenario in its natural (story, question) order.
MUSR_SCENARIOS = ['murder_mysteries', 'object_placements', 'team_allocation']

assert len(MUSR_SCENARIOS) == 3, \
    f'expected 3 MuSR scenarios, got {len(MUSR_SCENARIOS)}'
assert len(set(MUSR_SCENARIOS)) == 3, 'duplicate MuSR scenario names'
