#!/usr/bin/env python3
"""Post-run raw weighted score changes, never imported by a shortcut."""
import json
import math
from pathlib import Path

RT = Path(__file__).resolve().parent
spec = json.loads((RT / 'reference/scoring_spec.json').read_text())['experiments']
reference = json.loads((RT / 'reference/scores_test_original.json').read_text())['experiments']


def analyze(path):
    output = json.loads(path.read_text())['experiments']
    total = 0.0
    rows = {}
    for exp, experiment in output.items():
        if exp not in spec or exp not in reference:
            continue
        cfg = spec[exp]
        for method, metrics in experiment['results'].items():
            if method not in reference[exp]['results']:
                continue
            orig = reference[exp]['results'][method]
            terms = {m: cfg['weight'] * cfg['metrics'][m]['coefficient'] * (value - orig[m])
                     for m, value in metrics.items() if m in cfg['metrics'] and m in orig}
            delta = sum(terms.values())
            total += delta
            rows[method] = {'weighted_delta': delta, 'terms': terms, 'metrics': metrics}
    return {'total_weighted_delta': total, 'rows': rows}


result = {}
for case in sorted(RT.iterdir()):
    path = case / 'scores_test.json'
    if path.exists():
        result[case.name] = analyze(path)
    elif (case / 'evaluate_test_status.json').exists():
        result[case.name] = {'failure': json.loads((case / 'evaluate_test_status.json').read_text())}
    elif (case / 'error.txt').exists():
        result[case.name] = {'error': (case / 'error.txt').read_text()}
seeds = [value['total_weighted_delta'] for case, value in result.items()
         if case.startswith('reference_seed') and 'total_weighted_delta' in value]
summary = {'formula': 'sum_experiment weight * sum_matched_method sum_metric coefficient * (test - original_test)',
           'deltas_relative_to': 'reference/scores_test_original.json', 'cases': result,
           'seed_deltas': seeds, 'seed_spread': max(seeds)-min(seeds) if len(seeds) == 3 else None}
(RT / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
for case, value in result.items():
    print(case, value.get('total_weighted_delta', value.get('failure', value.get('error'))))
print('seed spread:', summary['seed_spread'])
