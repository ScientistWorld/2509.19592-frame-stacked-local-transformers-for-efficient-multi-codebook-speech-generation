"""Summarize cohort ranking and paired block bootstrap of actual measurements."""
import json
from pathlib import Path
import torch
from eval.test import evaluate as E
from eval.test.ranking import rank_attempts

HERE = Path(__file__).resolve().parent


def moments_blocks(tokens, speakers, blocks):
    stats = []
    for idx in blocks:
        mu, cov = E.feature_moments(tokens[idx], speakers[idx])
        n = torch.tensor([int((speakers[idx] == s).sum()) * 23 for s in range(16)] + [len(idx) * 23], dtype=torch.float64)
        total = mu * n[:, None]
        products = cov * (n - 1)[:, None, None] + mu[:, :, None] * mu[:, None, :] * n[:, None, None]
        stats.append((n, total, products))
    return tuple(torch.stack([s[k] for s in stats]) for k in range(3))


def aggregate(stats, indices):
    n, total, products = (part[indices].sum(0) for part in stats)
    mu = total / n[:, None]
    cov = (products - total[:, :, None] * total[:, None, :] / n[:, None, None]) / (n - 1)[:, None, None]
    return mu, cov


def distance(a, b):
    am, ac = a; bm, bc = b
    d = [E.frechet_moments(am[i], ac[i], bm[i], bc[i]) for i in range(17)]
    return d[-1], sum(d[:-1]) / 16


def main():
    cohort = json.loads((HERE / 'cohort_test.json').read_text())
    train = json.loads((HERE / 'cohort_train.json').read_text())
    ranking = cohort['_ranking']
    spec = json.loads(Path('scoring/reference.json').read_text())
    summary = {'ranking': ranking, 'reference': cohort['experiments'],
               'reference_diagnostics': cohort['_raw'], 'failures': cohort['_failures'],
               'attempts': {name: {'metrics': x['experiments'], 'failures': x['_failures']} for name, x in cohort['_attempts'].items()},
               'train_ranking': train['_ranking'], 'paired': {}}
    # Fixed seeds eliminate training randomness from future comparisons. This
    # still reports both sampled-output variability and old training-seed controls.
    def spread(names):
        vals = [ranking['attempts'][n]['improvement'] for n in names]
        return max(vals) - min(vals)
    summary['sample_seed_spread'] = spread(['current', 'reference_sample1', 'reference_sample2'])
    summary['training_seed_spread'] = spread(['current', 'reference_seed17', 'reference_seed27'])
    # Same held-out utterance indices and same reference sample in every pair.
    # Bootstrap sufficient statistics, so each replicate recomputes full FD,
    # including covariance, instead of averaging individual distances.
    rows = E.slice_rows()
    speakers = E.conditions()[rows]
    real_speakers = speakers.repeat_interleave(E.REFERENCES_PER_ROW)
    real = E.sample_tokens(real_speakers, E.HIDDEN_REFERENCE_SEED)
    block_rows = list(torch.arange(len(rows)).chunk(16))
    real_blocks = [(idx[:, None] * E.REFERENCES_PER_ROW + torch.arange(E.REFERENCES_PER_ROW)).flatten() for idx in block_rows]
    real_stats = moments_blocks(real, real_speakers, real_blocks)
    draws = torch.randint(16, (100, 16), generator=torch.Generator().manual_seed(90381))
    reference_stats, reference_validity = {}, {}
    for stack in (2, 4):
        pred = torch.load(Path(f'checkpoints/predictions/generator_{stack}x.pt'), weights_only=True)['pred_tokens'][rows].long()
        reference_stats[stack] = moments_blocks(pred, speakers, block_rows)
        reference_validity[stack] = torch.tensor([E.validity_parts(pred[idx], speakers[idx])['dependency_validity'] for idx in block_rows])
    for name in ('better_more_training', 'better_bigger', 'weaker_training'):
        improvement = torch.zeros(100, dtype=torch.float64)
        for stack, exp in E.EXPERIMENTS.items():
            path = HERE / f'raw_predictions/{name}/predictions/generator_{stack}x.pt'
            tokens = torch.load(path, weights_only=True)['pred_tokens'][rows].long()
            alt_stats = moments_blocks(tokens, speakers, block_rows)
            alt_validity = torch.tensor([E.validity_parts(tokens[idx], speakers[idx])['dependency_validity'] for idx in block_rows])
            for rep, indices in enumerate(draws):
                real_moments = aggregate(real_stats, indices)
                ref_fd, ref_sfd = distance(aggregate(reference_stats[stack], indices), real_moments)
                alt_fd, alt_sfd = distance(aggregate(alt_stats, indices), real_moments)
                deltas = {'dependency_validity': float((alt_validity[indices] - reference_validity[stack][indices]).mean()),
                          'codec_fd': alt_fd - ref_fd, 'speaker_conditioned_codec_fd': alt_sfd - ref_sfd}
                cfg = spec['experiments'][exp]
                for metric, delta in deltas.items():
                    definition = cfg['metrics'][metric]
                    sigma = ranking['standard_deviations'][f'{exp}/generator_{stack}x/{metric}']
                    sign = 1 if definition['higher_is_better'] else -1
                    if sigma:
                        improvement[rep] += cfg['weight'] * abs(definition['coefficient']) * sign * delta / sigma
        lo, hi = torch.quantile(improvement, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
        summary['paired'][name] = {'normalized_improvement': ranking['attempts'][name]['improvement'],
                                   'feasible': ranking['attempts'][name]['feasible'],
                                   'paired_bootstrap_95_percent': [lo, hi], 'replicates': 100, 'blocks': 16,
                                   'note': 'Unconditional scientific metric difference; feasibility reported separately. Standard deviations fixed to the ranked cohort.'}
    truth = json.loads((HERE / 'shortcut_replace_truth.json').read_text())
    summary['truth_replacement'] = {'failures': truth['_failures'], 'ranking': truth['_ranking']['attempts']['current']}
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for name in ranking['order']:
        value = ranking['attempts'][name]
        print(f"{name}: feasible={value['feasible']} improvement={value['improvement']:+.6f}")
    print(json.dumps(summary['paired'], indent=2))

if __name__ == '__main__':
    main()
