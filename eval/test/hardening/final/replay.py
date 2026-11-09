"""Final hardening replay: score every saved attempt with the pre-fix and the
fixed hidden evaluator, then summarize.

Attempts are the saved raw outputs of the previous verification job
(eval/test/hardening/raw_predictions) plus the new shortcut cases
(final/raw_predictions). Nothing is retrained. Two scores are reported:

* `linear`: what a rank that only reads the scored metric values computes:
  sum over experiments and metrics of weight * |coefficient| * direction *
  (value - reference) / cohort population std (same cohort before and after);
* `tier`: the evaluator's own (feasible, improvement) ordering.
"""
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
WORK = ROOT / 'tmp' / 'replay'
SAVED = [HERE.parent / 'raw_predictions', HERE / 'raw_predictions']
ROWS = {'codec_stack2': 'generator_2x', 'codec_stack4': 'generator_4x'}


def build_cohort():
    shutil.rmtree(WORK, ignore_errors=True)
    preds = WORK / 'checkpoints' / 'predictions'
    preds.mkdir(parents=True)
    for path in sorted((ROOT / 'checkpoints' / 'predictions').glob('*.pt')):
        (preds / path.name).symlink_to(path)
    names = []
    for base in SAVED:
        for case in sorted(base.iterdir()):
            src = case / 'predictions'
            if not src.is_dir():
                continue
            dst = WORK / 'checkpoints' / 'attempts' / case.name / 'predictions'
            dst.mkdir(parents=True)
            for path in sorted(src.glob('*.pt')):
                (dst / path.name).symlink_to(path)
            names.append(case.name)
    return preds, names


def run(evaluator, preds, out):
    log = out.with_suffix('.log')
    with log.open('w') as handle:
        subprocess.run([sys.executable, evaluator, '--pred-dir', str(preds), '--out', str(out)],
                       check=True, stdout=handle, stderr=subprocess.STDOUT,
                       env={'PYTHONPATH': str(ROOT), 'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '2'})
    return json.loads(out.read_text())


def values(scores):
    return {f'{exp}/{row}/{m}': scores['experiments'][exp]['results'][row][m]
            for exp, row in ROWS.items() for m in scores['experiments'][exp]['results'][row]}


def linear(table, spec):
    keys = list(table['reference'])
    sigma = {k: statistics.pstdev([table[n][k] for n in table]) for k in keys}
    result = {}
    for name, vals in table.items():
        total = 0.0
        for key in keys:
            exp, _, metric = key.split('/')
            cfg = spec['experiments'][exp]
            d = cfg['metrics'][metric]
            sign = 1 if d['higher_is_better'] else -1
            if sigma[key]:
                total += cfg['weight'] * abs(d['coefficient']) * sign * (vals[key] - table['reference'][key]) / sigma[key]
        result[name] = total
    return result, sigma


def collect(scores, names):
    table = {'reference': values(scores)}
    failures = {'reference': {k: v for k, v in scores['_failures'].items() if k in ROWS.values()}}
    for name in names:
        table[name] = values(scores['_attempts'][name])
        failures[name] = {k: v for k, v in scores['_attempts'][name]['_failures'].items() if k in ROWS.values()}
    return table, failures


def main():
    spec = json.loads((ROOT / 'scoring' / 'reference.json').read_text())
    preds, names = build_cohort()
    before_eval = ROOT / 'eval' / 'test' / '_before_evaluate.py'
    before_eval.write_text(subprocess.run(['git', 'show', 'HEAD:eval/test/evaluate.py'], check=True,
                                          capture_output=True, text=True).stdout)
    try:
        before = run(str(before_eval), preds, WORK / 'before.json')
    finally:
        before_eval.unlink()
    after = run(str(ROOT / 'eval' / 'test' / 'evaluate.py'), preds, WORK / 'after.json')
    summary = {'cohort': ['reference'] + names}
    for label, scores in (('before', before), ('after', after)):
        table, failures = collect(scores, names)
        lin, sigma = linear(table, spec)
        tier = scores['_ranking']['attempts']
        summary[label] = {
            'values': table, 'failures': failures, 'linear': lin, 'sigma': sigma,
            'tier': {('reference' if n == 'current' else n): {'feasible': v['feasible'], 'improvement': v['improvement']}
                     for n, v in tier.items()},
        }
    after_lin = summary['after']['linear']
    summary['seed_spread_after'] = {
        'sampling': max(after_lin[n] for n in ('reference', 'reference_sample1', 'reference_sample2'))
        - min(after_lin[n] for n in ('reference', 'reference_sample1', 'reference_sample2')),
        'training': max(after_lin[n] for n in ('reference', 'reference_seed17', 'reference_seed27'))
        - min(after_lin[n] for n in ('reference', 'reference_seed17', 'reference_seed27')),
    }
    summary['paired_after'] = paired(summary['after']['sigma'], spec)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(f"{'attempt':32s} {'before':>10s} {'after':>10s}  after-failure")
    for name in summary['cohort']:
        fail = summary['after']['failures'][name]
        print(f"{name:32s} {summary['before']['linear'][name]:+10.4f} {summary['after']['linear'][name]:+10.4f}  "
              f"{next(iter(fail.values()), 'feasible')[:70]}")
    print(json.dumps({'seed_spread_after': summary['seed_spread_after'], 'paired_after': summary['paired_after']}, indent=2))


def paired(sigma, spec):
    """Paired block bootstrap of the better variant against the reference: same
    held-out rows and same hidden reference sample in every replicate; full FD
    is recomputed from resampled sufficient statistics."""
    sys.path.insert(0, str(ROOT))
    from eval.test import evaluate as E
    from eval.test.hardening.summarize import moments_blocks, aggregate, distance
    rows = E.slice_rows()
    speakers = E.conditions()[rows]
    real_speakers = speakers.repeat_interleave(E.REFERENCES_PER_ROW)
    real = E.sample_tokens(real_speakers, E.HIDDEN_REFERENCE_SEED)
    block_rows = list(torch.arange(len(rows)).chunk(16))
    real_blocks = [(idx[:, None] * E.REFERENCES_PER_ROW + torch.arange(E.REFERENCES_PER_ROW)).flatten() for idx in block_rows]
    real_stats = moments_blocks(real, real_speakers, real_blocks)
    draws = torch.randint(16, (200, 16), generator=torch.Generator().manual_seed(90381))
    out = {}
    load = lambda p: torch.load(p, weights_only=True)['pred_tokens'][rows].long()
    for name in ('better_more_training', 'reference_seed17', 'reference_seed27'):
        improvement = torch.zeros(len(draws), dtype=torch.float64)
        for stack, exp in E.EXPERIMENTS.items():
            ref = load(ROOT / f'checkpoints/predictions/generator_{stack}x.pt')
            alt = load(HERE.parent / f'raw_predictions/{name}/predictions/generator_{stack}x.pt')
            ref_stats, alt_stats = moments_blocks(ref, speakers, block_rows), moments_blocks(alt, speakers, block_rows)
            ref_val = torch.tensor([E.validity_parts(ref[i], speakers[i])['dependency_validity'] for i in block_rows])
            alt_val = torch.tensor([E.validity_parts(alt[i], speakers[i])['dependency_validity'] for i in block_rows])
            cfg = spec['experiments'][exp]
            for rep, idx in enumerate(draws):
                real_m = aggregate(real_stats, idx)
                rf, rs = distance(aggregate(ref_stats, idx), real_m)
                af, as_ = distance(aggregate(alt_stats, idx), real_m)
                deltas = {'dependency_validity': float((alt_val[idx] - ref_val[idx]).mean()),
                          'codec_fd': af - rf, 'speaker_conditioned_codec_fd': as_ - rs}
                for metric, delta in deltas.items():
                    d = cfg['metrics'][metric]
                    s = sigma[f'{exp}/generator_{stack}x/{metric}']
                    if s:
                        improvement[rep] += cfg['weight'] * abs(d['coefficient']) * (1 if d['higher_is_better'] else -1) * delta / s
        lo, hi = torch.quantile(improvement, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
        out[name] = {'bootstrap_mean': float(improvement.mean()), 'paired_bootstrap_95_percent': [lo, hi],
                     'replicates': len(draws), 'blocks': 16}
    return out


if __name__ == '__main__':
    main()
