"""Ranking of evaluator-recomputed measurements across saved attempts.

A failed attempt sorts below every feasible attempt. Utility is an additive
coefficient-weighted improvement, standardized using this comparison cohort.
No submitted score, success flag, threshold or reference value is consumed.
"""
import math
import statistics


def cells(spec):
    for exp, cfg in spec['experiments'].items():
        for row, result in cfg['results'].items():
            if result['type'] == 'proposed':
                for metric, definition in cfg['metrics'].items():
                    yield exp, row, metric, cfg['weight'] * abs(definition['coefficient']), (1 if definition['higher_is_better'] else -1)


def measurement(scores, exp, row, metric):
    value = scores['experiments'][exp]['results'][row][metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'invalid measurement: {exp}/{row}/{metric}')
    return value


def feasible(scores, spec):
    for exp, cfg in spec['experiments'].items():
        for row, result in cfg['results'].items():
            if result['type'] == 'proposed':
                if row in scores.get('_failures', {}):
                    return False
                try:
                    for metric in cfg['metrics']:
                        measurement(scores, exp, row, metric)
                except (KeyError, ValueError):
                    return False
    return True


def rank_attempts(attempts, reference, spec):
    # The reference is an actual fixed comparator and a member of the cohort.
    # Each attempt has one vote, regardless of its row count or file size.
    scales = {}
    active = list(cells(spec))
    for exp, row, metric, _, _ in active:
        values = [measurement(reference, exp, row, metric)]
        values += [measurement(a, exp, row, metric) for a in attempts.values()]
        scales[f'{exp}/{row}/{metric}'] = statistics.pstdev(values)
    results = {}
    for name, scores in attempts.items():
        parts = {}
        for exp, row, metric, weight, direction in active:
            key = f'{exp}/{row}/{metric}'
            sigma = scales[key]
            delta = measurement(scores, exp, row, metric) - measurement(reference, exp, row, metric)
            parts[key] = weight * direction * delta / sigma if sigma else 0.0
        results[name] = {'feasible': feasible(scores, spec), 'improvement': sum(parts.values()), 'contributions': parts}
    order = sorted(results, key=lambda n: (results[n]['feasible'], results[n]['improvement']), reverse=True)
    return {'order': order, 'attempts': results, 'standard_deviations': scales,
            'sort_key': '(feasible, improvement), descending; a failed run never outranks a feasible run'}
