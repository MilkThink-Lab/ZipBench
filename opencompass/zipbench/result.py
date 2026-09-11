"""The standard ZipBench result shape: one headline metric, the rest nested.

Every ZipBench evaluator returns::

    {
        'score':     <float>,   # the benchmark's OFFICIAL metric, and the only
                                # top-level number -> the only row the summary
                                # table shows
        'metric':    <str>,     # which metric 'score' is, and how it
                                # aggregates
        'secondary': <dict>,    # every other number (counters and
                                # per-group breakdowns included) -- kept
                                # out of the table, written to the json
        'details':   <list>,    # per-item records (unchanged OpenCompass API)
    }

**Why the nesting.** ``DefaultSummarizer`` emits ONE TABLE ROW PER TOP-LEVEL
NUMERIC KEY: ``_pick_up_results`` keeps every value passing
``isinstance(score, (int, float))`` and ``_format_table`` renders one row for
each (``opencompass/summarizers/default.py``; ``default_subjective.py`` is
identical). An evaluator returning six numbers therefore produced six
undifferentiated rows -- including bare counters like ``num_samples`` -- with
nothing marking which one is the benchmark's official metric. Values that are
not ``int``/``float`` fail that filter, so nesting the secondary numbers keeps
them out of the table while ``mmengine.dump`` still writes them, in full, to
``results/<model>/<dataset>.json``. ``metric`` is a ``str`` for the same
reason: invisible in the table, impossible to miss in the json.

**Why the key is literally ``score``.** It is ``METRIC_WHITELIST[0]`` in both
summarizers, and ``_dm`` is sorted by that whitelist. So ``score`` always sorts
first and stays the table's default metric even if OpenCompass later injects a
whitelisted key of its own -- e.g. ``extract_rate`` (whitelist index 11), which
``openicl_eval.py`` adds whenever ``cal_extract_rate`` is on. Relying on "the
evaluator returns the primary metric first" would silently lose to that.

The official metric of each benchmark is *not* a free choice: see the
"Official metric per benchmark" table in ``zipbench/README.md``.

One sanctioned exception: when a benchmark's source officially reports a
*pair* of numbers side by side (SimpleQA: f1 + accuracy_given_attempted), the
second one may also be set as a top-level numeric key next to ``score`` so it
gets its own table row; ``score`` (whitelist index 0) still sorts first and
stays the headline.
"""

__all__ = ['zip_result', 'ZipResultShapeMixin']


def zip_result(score, metric, secondary=None, details=None):
    """Build the standard ZipBench result dict (see the module docstring).

    Args:
        score (float): the headline number -- the benchmark's official metric.
        metric (str): ``'<metric name> — <one-line description>'``. Name the
            aggregation (micro/macro) and, in one clause, what the benchmark's
            official metric is. Keep it to what a reader needs to read the
            number.
        secondary (dict): every other number, including counters and per-group
            breakdowns. Repeat ``score`` here under its real name so the json
            is self-describing and scripts can select it by name.
        details (list): per-item records, passed through untouched. Omitted
            from the result when ``None`` (SciCode writes its own detail file).
    """
    result = {'score': float(score), 'metric': metric}
    result['secondary'] = dict(secondary or {})
    if details is not None:
        result['details'] = details
    return result


class ZipResultShapeMixin:
    """Keep ``metric`` / ``secondary`` unwrapped in the dumped results json.

    ``BaseEvaluator.evaluate`` collects each of the ``n`` replicas' results
    into a list per key, then unwraps only ``int``/``float`` values when
    ``n == 1`` (``opencompass/openicl/icl_evaluator/icl_base_evaluator.py``).
    A ``str`` or ``dict`` value would therefore land in the json as
    ``["..."]`` / ``[{...}]`` -- which is how ``per_task`` had been showing
    up in BBH results. Unwrap those two keys for the ``n == 1`` case; with
    ``n > 1`` the per-replica list is the correct representation and is left
    alone.

    Mix in FIRST so it precedes the upstream evaluator in the MRO::

        class WeightedFooEvaluator(ZipResultShapeMixin, FooEvaluator):
    """

    def evaluate(self, k, n, original_dataset, **score_kwargs):
        result = super().evaluate(k, n, original_dataset, **score_kwargs)
        if n == 1:
            for key in ('metric', 'secondary'):
                value = result.get(key)
                if isinstance(value, list) and len(value) == 1:
                    result[key] = value[0]
        return result
