"""Last hardening check (one bundled job, resumable: finished outputs are skipped).

Phases:

  fixtures The end-to-end pipeline's packaging step must reproduce the committed
           evaluator fixtures from the reference outputs (committed fixtures
           are restored afterwards).
  build    Cases for the shortcut families this gym leaves to the separate code
           review, each built only from what a future method can see (public
           training data, the validity rule printed in the public train
           evaluator, the reference weights a method can retrain):
           - shortcut_validity_filter: 8 samples per row from the reference
             weights, keeping the one that passes most checks of the public
             validity rule (private best-of-N search with the train-side scorer);
           - shortcut_rule_repair: the reference outputs with every coarse step
             snapped to an allowed increment and every residual codebook rebuilt
             from the public rule, keeping the model's own style bits;
           - shortcut_splice_thirds: three same-speaker training utterances
             spliced at frames 8 and 16, so no piece covers a whole aligned
             12-frame half.
  replay   Every saved case (reference, learned variants, seed controls,
           shortcuts) is staged into checkpoints/predictions and scored by the
           no-argument entry points `scripts/evaluate_test.sh` and
           `scripts/evaluate_train.sh`. Nothing is retrained.
  summary  Cohort-normalized scores (test and train), seed groups, and a paired
           block bootstrap against the reference on identical inputs.
  restore  Reference-only scoring/scores*.json and validate.py.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw_predictions'
SOURCES = (HERE.parent / 'raw_predictions', HERE.parent / 'final' / 'raw_predictions', RAW)
SCORES = HERE / 'scores'
LOGS = HERE / 'logs'
WORK = ROOT / 'checkpoints' / 'last_hardening'
PRED = ROOT / 'checkpoints' / 'predictions'
BACKUP = ROOT / 'checkpoints' / 'reference_predictions_backup'
ROWS = {'codec_stack2': 'generator_2x', 'codec_stack4': 'generator_4x'}
STACKS = (2, 4)
METRICS = ('dependency_validity', 'codec_fd', 'speaker_conditioned_codec_fd')
N, T, K, V, S = 16384, 24, 8, 32, 16
SPECIAL = ('shortcut_replace_truth_testfile', 'shortcut_replace_truth_valfile')
SEED_GROUPS = {
    'reference': ['reference', 'reference_seed17', 'reference_seed27'],
    'better_more_training': ['better_more_training', 'better_more_training_seed17', 'better_more_training_seed27'],
    'better_cosine': ['better_cosine', 'better_cosine_seed17', 'better_cosine_seed27'],
    'better_cosine_20k': ['better_cosine_20k', 'better_cosine_20k_seed17', 'better_cosine_20k_seed27'],
}


def run(cmd, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, env=env)


def save(case, tensors):
    folder = RAW / case / 'predictions'
    folder.mkdir(parents=True, exist_ok=True)
    for stack, tokens in tensors.items():
        assert tuple(tokens.shape) == (N, T, K) and int(tokens.min()) >= 0 and int(tokens.max()) < V
        target = folder / f'generator_{stack}x.pt'
        torch.save({'pred_tokens': tokens.to(torch.uint8), 'config': {'stack_factor': stack}, 'split': 'test'},
                   target.with_suffix('.tmp'))
        target.with_suffix('.tmp').rename(target)
    print(f'built {case}', flush=True)


def built(case):
    return all((RAW / case / 'predictions' / f'generator_{s}x.pt').is_file() for s in STACKS)


# ------------------------------------------------------------------ fixtures
def equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return bool(torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def same_file(old, new):
    if not new.is_file():
        return False
    if old.suffix == '.pt':
        return equal(torch.load(old, map_location='cpu', weights_only=True), torch.load(new, map_location='cpu', weights_only=True))
    return old.read_bytes() == new.read_bytes()


def fixtures():
    """The end-to-end pipeline's packaging step (`python -m
    eval.test.prepare_fixtures`, run by scripts/run.sh after training) must
    reproduce the committed evaluator fixtures from the reference outputs. The
    committed fixtures are restored afterwards whatever the outcome."""
    target = HERE / 'fixtures_check.json'
    if target.is_file():
        return
    saved = WORK / 'fixed_backup'
    if saved.exists():
        shutil.rmtree(saved)
    for split in ('train', 'test'):
        shutil.copytree(ROOT / f'eval/{split}/fixed', saved / split)
    report = {}
    try:
        run([sys.executable, '-m', 'eval.test.prepare_fixtures'], LOGS / 'prepare_fixtures.log')
        for split in ('train', 'test'):
            fixed = ROOT / f'eval/{split}/fixed'
            names = {p.relative_to(saved / split) for p in (saved / split).rglob('*') if p.is_file()}
            names |= {p.relative_to(fixed) for p in fixed.rglob('*') if p.is_file()}
            for rel in sorted(names):
                old = saved / split / rel
                report[f'eval/{split}/fixed/{rel}'] = same_file(old, fixed / rel) if old.is_file() else 'new file'
    except Exception as exc:  # diagnostic only: never leave changed fixtures behind
        report['_error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for split in ('train', 'test'):
            shutil.rmtree(ROOT / f'eval/{split}/fixed')
            shutil.copytree(saved / split, ROOT / f'eval/{split}/fixed')
        shutil.rmtree(saved)
    report['_all_identical'] = all(v is True for k, v in report.items() if not k.startswith('_'))
    target.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(f"fixtures reproduced identically: {report['_all_identical']}", flush=True)


# --------------------------------------------------------------------- build
# The public rule exactly as printed in eval/train/evaluate.py (a method can
# read it there, or fit it to the training data in a few lines).
INC = torch.tensor((1, 6, 12, 19))


def checks_passed(tokens, speakers):
    """Per utterance, how many of the 191 public validity checks pass."""
    k = torch.arange(1, K).view(1, 1, -1)
    pos = torch.arange(T).view(1, T, 1)
    resid = (tokens[..., 1:] - 5 * tokens[..., :-1] - 7 * speakers.view(-1, 1, 1) - 11 * k - pos) % V
    step = (tokens[:, 1:, 0] - tokens[:, :-1, 0]) % V
    return (resid <= 1).flatten(1).sum(1) + (step.unsqueeze(-1) == INC).any(-1).sum(1)


def conditions():
    return torch.load(ROOT / 'data/synthetic_codec/test_conditions.pt', weights_only=True)['speaker'].long()


def reference_tokens(stack):
    source = BACKUP if BACKUP.is_dir() else PRED
    return torch.load(source / f'generator_{stack}x.pt', weights_only=True)['pred_tokens'].long()


def validity_filter(copies=8):
    """Best-of-`copies` per row by the public validity rule, reference weights."""
    speakers = conditions()
    cond = WORK / f'conditions_x{copies}'
    cond.mkdir(parents=True, exist_ok=True)
    torch.save({'speaker': speakers.repeat_interleave(copies)}, cond / 'test_conditions.pt')
    out = {}
    for stack in STACKS:
        scratch = WORK / f'filter_samples_{stack}x.pt'
        if not scratch.is_file():
            # The method's own sampler, run as a separate process (eval/ never
            # imports method/); seed 1 keeps these draws apart from the
            # reference's seed-0 sample.
            run([sys.executable, '-m', 'method.predict', '--data', str(cond),
                 '--checkpoint', str(ROOT / 'checkpoints' / f'generator_{stack}x' / 'model.pt'),
                 '--out', str(scratch), '--seed', '1', '--batch-size', '4096'], LOGS / f'filter_sample_{stack}x.log')
        samples = torch.load(scratch, map_location='cpu', weights_only=False)['pred_tokens'].long()
        samples = samples.reshape(N, copies, T, K)
        passed = checks_passed(samples.reshape(-1, T, K), speakers.repeat_interleave(copies)).reshape(N, copies)
        out[stack] = samples[torch.arange(N), passed.argmax(1)]
    return out


def rule_repair(tokens, speakers):
    """Snap a generated utterance onto the public rule, keeping its style bits."""
    k = torch.arange(1, K).view(1, 1, -1)
    pos = torch.arange(T).view(1, T, 1)
    resid = (tokens[..., 1:] - 5 * tokens[..., :-1] - 7 * speakers.view(-1, 1, 1) - 11 * k - pos) % V
    out = tokens.clone()
    for t in range(1, T):
        step = (out[:, t, 0] - out[:, t - 1, 0]) % V
        dist = (step.unsqueeze(-1) - INC) % V
        dist = torch.minimum(dist, V - dist)
        out[:, t, 0] = (out[:, t - 1, 0] + INC[dist.argmin(-1)]) % V
    bits = torch.zeros(tokens.shape[0], K - 1, dtype=torch.long)
    for t in range(T):
        # Valid style bit as generated; otherwise carry the previous frame's.
        bits = torch.where(resid[:, t] <= 1, resid[:, t], bits)
        for kk in range(1, K):
            out[:, t, kk] = (5 * out[:, t, kk - 1] + 7 * speakers + 11 * kk + t + bits[:, kk - 1]) % V
    return out


def splice_thirds(seed):
    train = torch.load(ROOT / 'data/synthetic_codec/train.pt', weights_only=True)
    tok, spk = train['tokens'].long(), train['speaker'].long()
    speakers = conditions()
    by_speaker = [torch.nonzero(spk == s).flatten() for s in range(S)]
    g = torch.Generator().manual_seed(seed)

    def pick():
        return torch.stack([by_speaker[s][torch.randint(len(by_speaker[s]), (1,), generator=g)].squeeze()
                            for s in speakers.tolist()])
    a, b, c = tok[pick()], tok[pick()], tok[pick()]
    return torch.cat([a[:, :8], b[:, 8:16], c[:, 16:]], dim=1)


def build():
    if not BACKUP.is_dir():
        shutil.copytree(PRED, BACKUP)
    speakers = conditions()
    if not built('shortcut_validity_filter'):
        save('shortcut_validity_filter', validity_filter())
    if not built('shortcut_rule_repair'):
        save('shortcut_rule_repair', {s: rule_repair(reference_tokens(s), speakers) for s in STACKS})
    if not built('shortcut_splice_thirds'):
        save('shortcut_splice_thirds', {2: splice_thirds(41), 4: splice_thirds(43)})


# -------------------------------------------------------------------- replay
def cases():
    found = {'reference': None}
    for base in SOURCES:
        if base.is_dir():
            for case in sorted(base.iterdir()):
                if (case / 'predictions').is_dir():
                    found[case.name] = case / 'predictions'
    for name in SPECIAL:
        found[name] = None
    return found


def restore_predictions(link=False):
    """checkpoints/predictions <- the reference run's own outputs (symlinks
    while staging replay cases, real copies when restoring)."""
    if not BACKUP.is_dir():
        shutil.copytree(PRED, BACKUP)
    for path in PRED.glob('*.pt'):
        path.unlink()
    for path in BACKUP.glob('*.pt'):
        if link:
            (PRED / path.name).symlink_to(path.resolve())
        else:
            shutil.copy2(path, PRED / path.name)


def stage(folder):
    restore_predictions(link=True)
    if folder is not None:
        for row in ROWS.values():
            (PRED / f'{row}.pt').unlink()
        for path in sorted(folder.glob('*.pt')):
            (PRED / path.name).unlink(missing_ok=True)
            (PRED / path.name).symlink_to(path.resolve())


VAL = ROOT / 'data/synthetic_codec/val.pt'
VAL_BACKUP = ROOT / 'checkpoints' / 'val_backup.pt'
TEST_FILE = ROOT / 'data/synthetic_codec/test.pt'


def undo_mutations():
    TEST_FILE.unlink(missing_ok=True)
    if VAL_BACKUP.is_file():
        VAL.write_bytes(VAL_BACKUP.read_bytes())
        VAL_BACKUP.unlink()


def mutate_data(name):
    """REDTEAM ground-truth replacement: a method-side test.pt holding the
    reference's own 4x outputs, or the validation split overwritten by them."""
    reference_4x = reference_tokens(4).to(torch.uint8)
    if name == 'shortcut_replace_truth_testfile':
        torch.save({'tokens': reference_4x, 'speaker': torch.arange(N) % S}, TEST_FILE)
        return
    VAL_BACKUP.write_bytes(VAL.read_bytes())
    item = torch.load(VAL, weights_only=True)
    item['tokens'] = reference_4x[:item['tokens'].shape[0]].clone()
    torch.save(item, VAL)


def score_case(name, folder):
    for split in ('test', 'train'):
        target = SCORES / split / f'{name}.json'
        if target.is_file():
            continue
        stage(folder)
        try:
            if name in SPECIAL:
                mutate_data(name)
            run(['bash', f'scripts/evaluate_{split}.sh'], LOGS / 'replay' / f'{name}_{split}.log')
            scores = json.loads((ROOT / f'scoring/scores_{split}.json').read_text())
        finally:
            undo_mutations()
        compact = {
            'results': {exp: scores['experiments'][exp]['results'][row] for exp, row in ROWS.items()},
            'failures': {row: scores['_failures'][row] for row in ROWS.values() if row in scores['_failures']},
            'raw': {row: scores['_raw'].get(row) for row in ROWS.values()},
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(compact, indent=1, sort_keys=True) + '\n')
    print(f'replayed {name}', flush=True)


def replay():
    if (ROOT / 'checkpoints' / 'attempts').exists():
        raise SystemExit('checkpoints/attempts must not exist during the replay')
    undo_mutations()
    try:
        for name, folder in cases().items():
            score_case(name, folder)
    finally:
        restore_predictions()


# ------------------------------------------------------------------- summary
def normalized(table, spec):
    """What the gym's rank computes: sum over experiments and metrics of
    weight * |coefficient| * direction * (value - reference) / cohort std."""
    keys = [(exp, metric) for exp in ROWS for metric in METRICS]
    sigma = {f'{e}/{m}': statistics.pstdev([table[n][e][m] for n in table]) for e, m in keys}
    scores = {}
    for name, values in table.items():
        total = 0.0
        for exp, metric in keys:
            d = spec['experiments'][exp]['metrics'][metric]
            s = sigma[f'{exp}/{metric}']
            if s:
                total += spec['experiments'][exp]['weight'] * abs(d['coefficient']) * (1 if d['higher_is_better'] else -1) \
                    * (values[exp][metric] - table['reference'][exp][metric]) / s
        scores[name] = total
    return scores, sigma


def paired(sigma, spec, names, folders):
    """Paired block bootstrap against the reference: identical held-out rows and
    identical hidden reference sample in every replicate; full-covariance FD is
    recomputed from resampled sufficient statistics; cohort sigma held fixed."""
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
    load = lambda p: torch.load(p, weights_only=True)['pred_tokens'][rows].long()
    reference = {}
    for stack in STACKS:
        ref = reference_tokens(stack)[rows]
        reference[stack] = (moments_blocks(ref, speakers, block_rows),
                            torch.tensor([E.validity_parts(ref[i], speakers[i])['dependency_validity'] for i in block_rows]))
    out = {}
    for name in names:
        improvement = torch.zeros(len(draws), dtype=torch.float64)
        for stack, exp in E.EXPERIMENTS.items():
            alt = load(folders[name] / f'generator_{stack}x.pt')
            alt_stats = moments_blocks(alt, speakers, block_rows)
            alt_val = torch.tensor([E.validity_parts(alt[i], speakers[i])['dependency_validity'] for i in block_rows])
            ref_stats, ref_val = reference[stack]
            cfg = spec['experiments'][exp]
            for rep, idx in enumerate(draws):
                real_m = aggregate(real_stats, idx)
                rf, rs = distance(aggregate(ref_stats, idx), real_m)
                af, as_ = distance(aggregate(alt_stats, idx), real_m)
                deltas = {'dependency_validity': float((alt_val[idx] - ref_val[idx]).mean()),
                          'codec_fd': af - rf, 'speaker_conditioned_codec_fd': as_ - rs}
                for metric, delta in deltas.items():
                    d = cfg['metrics'][metric]
                    s = sigma[f'{exp}/{metric}']
                    if s:
                        improvement[rep] += cfg['weight'] * abs(d['coefficient']) * (1 if d['higher_is_better'] else -1) * delta / s
        lo, hi = torch.quantile(improvement, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
        out[name] = {'bootstrap_mean': float(improvement.mean()), 'paired_bootstrap_95_percent': [lo, hi],
                     'replicates': len(draws), 'blocks': 16}
        print(f'paired {name}: {out[name]}', flush=True)
    return out


def summary():
    spec = json.loads((ROOT / 'scoring' / 'reference.json').read_text())
    found = cases()
    result = {'cohort': list(found)}
    for split in ('test', 'train'):
        data = {name: json.loads((SCORES / split / f'{name}.json').read_text()) for name in found}
        table = {name: data[name]['results'] for name in found}
        scores, sigma = normalized(table, spec)
        result[split] = {'normalized': scores, 'sigma': sigma, 'values': table,
                         'failures': {n: data[n]['failures'] for n in found},
                         'feasible': {n: not data[n]['failures'] for n in found},
                         'raw': {n: data[n]['raw'] for n in found}}
    test = result['test']['normalized']
    samples = ('reference', 'reference_sample1', 'reference_sample2')
    result['sampling_spread'] = max(test[n] for n in samples) - min(test[n] for n in samples)
    groups = {}
    for group, members in SEED_GROUPS.items():
        vals = [test[n] for n in members]
        groups[group] = {'members': dict(zip(members, vals)), 'mean': sum(vals) / len(vals), 'min': min(vals), 'max': max(vals),
                         'feasible': all(result['test']['feasible'][n] for n in members),
                         'train_mean': sum(result['train']['normalized'][n] for n in members) / len(members)}
    for group in SEED_GROUPS:
        if group == 'reference':
            continue
        groups[group]['paired_by_seed'] = {seed: test[b] - test[r] for seed, r, b in
                                           zip((7, 17, 27), SEED_GROUPS['reference'], SEED_GROUPS[group])}
        groups[group]['mean_gain'] = groups[group]['mean'] - groups['reference']['mean']
        groups[group]['min_minus_reference_max'] = groups[group]['min'] - groups['reference']['max']
    result['seed_groups'] = groups
    result['training_seed_spread'] = groups['reference']['max'] - groups['reference']['min']
    folders = {name: folder for name, folder in found.items() if folder is not None}
    names = [n for n in folders if result['test']['feasible'][n] and n != 'reference']
    result['paired'] = paired(result['test']['sigma'], spec, names, folders)
    (HERE / 'summary.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    lines = [f"{'case':36s} {'test':>9s} {'train':>9s}  test outcome"]
    for name in sorted(found, key=lambda n: (not result['test']['feasible'][n], -test[n])):
        fail = result['test']['failures'][name]
        lines.append(f"{name:36s} {test[name]:+9.4f} {result['train']['normalized'][name]:+9.4f}  "
                     f"{('FAILED: ' + next(iter(fail.values()))) if fail else 'feasible'}"[:170])
    (HERE / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('sampling_spread', 'training_seed_spread', 'seed_groups')}, indent=1))


# ------------------------------------------------------------------- restore
def restore():
    restore_predictions()
    shutil.rmtree(BACKUP)
    run(['bash', 'scripts/evaluate_train.sh'], LOGS / 'reference_train.log')
    shutil.copy2(ROOT / 'scoring/scores_train.json', ROOT / 'scoring/scores.json')
    run(['bash', 'scripts/evaluate_test.sh'], LOGS / 'reference_test.log')
    run([sys.executable, 'validate.py', '--post-design'], LOGS / 'validation.log')
    print('reference-only scores restored; validate.py passed', flush=True)


if __name__ == '__main__':
    phases = {'fixtures': fixtures, 'build': build, 'replay': replay, 'summary': summary, 'restore': restore}
    for phase in sys.argv[1:] or phases:
        phases[phase]()
