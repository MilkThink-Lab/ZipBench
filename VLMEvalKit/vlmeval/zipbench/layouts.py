"""Anchor position space -> TSV ``index`` resolvers (ZipBench layouts).

An anchor pkl stores *positions* into the correctness matrix used at mining
time. For most datasets that position space is simply the TSV row order
(``RowLayout``). OCRBench_v2's mining matrices are stored per task group, so
its position space is the concatenation "group order x within-group TSV
order".

A layout answers three questions:

* ``positions(data)`` -- the TSV ``index`` value living at every anchor
  position, in position order;
* ``groups(data)``    -- the task-group name of every position (``None`` for
  flat layouts);
* ``aggregate(scores, weights, groups)`` -- how the benchmark officially
  folds per-item scores into one number (micro vs group-macro).

``stack_record(data)`` additionally re-assembles a mining record pkl's
``data`` dict into one ``(n_items, n_models)`` matrix in this layout's
position order, for ``validate --record``.
"""

#: category -> aggregation-group cache, filled by probing the upstream code.
_CAT2GROUP = {}


def ocrbench_cat2group(categories):
    """``{category: group}`` probed from the upstream aggregation if-chain.

    Feeding ``ocrbench_v2_aggregate_accuracy`` a single scored item lights up
    exactly one group key, so the mapping is *derived* from upstream rather
    than duplicated here -- if upstream regroups, we follow automatically.
    """
    todo = [c for c in dict.fromkeys(categories) if c not in _CAT2GROUP]
    if todo:
        from ..dataset.utils.ocrbrnch_v2_eval import ocrbench_v2_aggregate_accuracy
        for cat in todo:
            en, cn = ocrbench_v2_aggregate_accuracy([{'type': cat, 'score': 1.0}])
            lit = {**en, **cn}
            if len(lit) != 1:
                raise ValueError(
                    f'category {cat!r} lit {len(lit)} aggregation groups ({sorted(lit)}); '
                    'cannot derive an unambiguous category -> group mapping')
            _CAT2GROUP[cat] = next(iter(lit))
    return {c: _CAT2GROUP[c] for c in dict.fromkeys(categories)}


class RowLayout:
    """Position p == TSV row p. The default, equivalent to prior behaviour."""

    name = 'row'
    #: short tag used as the key in per-layout manifest dicts; None == flat.
    label = None
    GROUPS = None

    def positions(self, data):
        return list(data['index'])

    def groups(self, data):
        return None

    def aggregate(self, scores, weights, groups=None):
        """Micro: ``sum(w*s)/sum(w)``."""
        import numpy as np

        scores = np.asarray(scores, dtype=float)
        weights = np.asarray(weights, dtype=float)
        return float((scores * weights).sum() / weights.sum())

    def stack_record(self, data):
        if len(data) != 1:
            raise ValueError(
                f"record pkl holds {sorted(data)} -- a multi-key 'data' almost certainly "
                'means a concatenated position space; use a dedicated layout')
        import numpy as np

        return np.asarray(next(iter(data.values()))['correctness'], dtype=float)


class IdSortedLayout(RowLayout):
    """Position p == the p-th TSV row when rows are sorted lexicographically by
    the ``id`` column.

    MMMU_Pro's mining matrices are stored in this order, NOT in TSV row order.
    The ``id`` column must be unique for the mapping to be a bijection.
    """

    name = 'id_sorted'

    def positions(self, data):
        if 'id' not in data:
            raise ValueError('id_sorted layout needs an `id` column in the TSV')
        ids = [str(x) for x in data['id']]
        if len(set(ids)) != len(ids):
            raise ValueError('`id` column is not unique -- id_sorted layout is ambiguous')
        indices = list(data['index'])
        return [idx for _, idx in sorted(zip(ids, indices))]


class IdListLayout(RowLayout):
    """Position p == the p-th id in an explicit, spec-shipped id file.

    For benchmarks whose mining matrices are keyed by an id list that exists
    nowhere in a VLMEvalKit TSV (OmniDocBench v1.5: page basenames in the
    record pkl's ``question_ids`` order), the id list itself is the layout.
    The file is a JSON array of ids shipped next to the subset specs.

    ``positions(data)`` ignores ``data`` entirely, so this layout also works
    for datasets with no TSV at all (convert with ``--no-dataset``).
    """

    name = 'id_list'

    def __init__(self, ids_path):
        import json
        import os.path as osp

        self.ids_path = osp.expanduser(ids_path)
        self.name = f'id_list:{ids_path}'
        with open(self.ids_path, 'r', encoding='utf-8') as f:
            self.ids = json.load(f)
        if not isinstance(self.ids, list) or not self.ids:
            raise ValueError(f'{ids_path}: expected a non-empty JSON array of ids')
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f'{ids_path}: ids are not unique')

    def positions(self, data):
        return list(self.ids)


class OCRBenchV2Layout(RowLayout):
    """Positions = task groups concatenated (record-pkl key order, i.e. the
    alphabetical group order), rows within a group in TSV order.

    Group order and sizes are written out explicitly on purpose -- they must
    match the mined position space, so a silent upstream change should
    fail the assertions below rather than re-derive a different layout.
    """

    EN_GROUPS = ('en_element_parsing', 'en_knowledge_reasoning',
                 'en_mathematical_calculation', 'en_relationship_extraction',
                 'en_text_detection', 'en_text_recognition',
                 'en_text_spotting', 'en_visual_text_understanding')
    EN_SIZES = (1600, 1400, 500, 700, 500, 1200, 200, 1300)
    CN_GROUPS = ('cn_element_parsing', 'cn_knowledge_reasoning',
                 'cn_relationship_extraction', 'cn_text_recognition',
                 'cn_visual_text_understanding')
    CN_SIZES = (800, 800, 600, 200, 200)

    def __init__(self, lang):
        assert lang in ('en', 'cn')
        self.lang = lang
        self.label = lang
        self.name = f'ocrbench_v2_{lang}'
        self.GROUPS = self.EN_GROUPS if lang == 'en' else self.CN_GROUPS
        self.SIZES = self.EN_SIZES if lang == 'en' else self.CN_SIZES

    def _by_group(self, data):
        """TSV ``index`` values of each group, in TSV row order."""
        cat2group = ocrbench_cat2group(data['category'])
        by_group = {g: [] for g in self.GROUPS}
        for idx, cat in zip(data['index'], data['category']):
            group = cat2group[cat]
            if group in by_group:
                by_group[group].append(idx)
        for group, size in zip(self.GROUPS, self.SIZES):
            if len(by_group[group]) != size:
                raise ValueError(
                    f'{self.name}: group {group!r} has {len(by_group[group])} rows in the TSV, '
                    f'expected {size} -- the layout no longer matches the mining records')
        return by_group

    def positions(self, data):
        by_group = self._by_group(data)
        return [idx for group in self.GROUPS for idx in by_group[group]]

    def groups(self, data):
        return [g for g, size in zip(self.GROUPS, self.SIZES) for _ in range(size)]

    def aggregate(self, scores, weights, groups):
        """Within-group weighted mean, then macro average over the groups."""
        import numpy as np

        scores = np.asarray(scores, dtype=float)
        weights = np.asarray(weights, dtype=float)
        groups = np.asarray(groups)
        missing = [g for g in self.GROUPS if g not in set(groups.tolist())]
        if missing:
            # A missing group silently changes the macro-average denominator.
            raise ValueError(f'{self.name}: no items for group(s) {missing}')
        per_group = []
        for group in self.GROUPS:
            m = groups == group
            per_group.append(float((scores[m] * weights[m]).sum() / weights[m].sum()))
        return float(np.mean(per_group))

    def stack_record(self, data):
        import numpy as np

        missing = [g for g in self.GROUPS if g not in data]
        if missing:
            raise ValueError(f'{self.name}: record pkl is missing group(s) {missing}')
        blocks = []
        for group, size in zip(self.GROUPS, self.SIZES):
            block = np.asarray(data[group]['correctness'], dtype=float)
            if block.shape[0] != size:
                raise ValueError(
                    f'{self.name}: record group {group!r} has {block.shape[0]} items, expected {size}')
            blocks.append(block)
        return np.concatenate(blocks, axis=0)


LAYOUTS = {
    'row': RowLayout(),
    'id_sorted': IdSortedLayout(),
    'ocrbench_v2_en': OCRBenchV2Layout('en'),
    'ocrbench_v2_cn': OCRBenchV2Layout('cn'),
}


def get_layout(name):
    if name.startswith('id_list:'):
        return IdListLayout(name.split(':', 1)[1])
    if name not in LAYOUTS:
        raise ValueError(
            f'unknown layout {name!r}; available: {sorted(LAYOUTS)} or id_list:<ids.json>')
    return LAYOUTS[name]
