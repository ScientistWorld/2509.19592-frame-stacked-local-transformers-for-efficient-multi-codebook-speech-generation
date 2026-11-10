"""Final hardening job (one bundled run).

Phases, each resumable (finished outputs are skipped):

  train    GPU. The learned variants still missing from the saved evidence:
           the 20k-step variant at reference training seeds 17/27,
           cosine-learning-rate variants (same 10k budget, and 20k) at seeds
           7/17/27, two
           prediction-only variants from saved weights, and the
           train-on-formerly-public-evaluator-references shortcut.
  replay   Every saved case is staged into checkpoints/predictions and scored
           by the no-argument hidden-test entry point `scripts/evaluate_test.sh`
           ("after"), and by the hidden evaluator as it stood at the start of
           this iteration, commit 205a717 ("before"). Nothing is retrained.
  summary  Cohort-normalized scores, seed analysis and paired bootstrap.
  restore  Reference-only scoring/scores*.json and validate.py.

All predictions are sampled with seed 0, like the reference and the saved
seed controls. Raw outputs are stored as uint8 tokens with the fixed
operating-point declaration only.
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
OLD_RAW = HERE.parent / 'raw_predictions'
# Overridable only for a local dry run of the replay; the job uses the defaults.
SCORES = Path(os.environ.get('FINAL_SCORES_DIR', HERE / 'scores'))
LOGS = Path(os.environ.get('FINAL_LOGS_DIR', HERE / 'logs'))
SUMMARY_DIR = Path(os.environ.get('FINAL_SUMMARY_DIR', HERE))
CKPT = ROOT / 'checkpoints' / 'variants'
PRED = ROOT / 'checkpoints' / 'predictions'
BACKUP = ROOT / 'checkpoints' / 'reference_predictions_backup'
BEFORE_COMMIT = '205a717'
ROWS = {'codec_stack2': 'generator_2x', 'codec_stack4': 'generator_4x'}
STACKS = (2, 4)
METRICS = ('dependency_validity', 'codec_fd', 'speaker_conditioned_codec_fd')

TRAINED = {
    'better_more_training_seed17': {'steps': 20000, 'seed': 17},
    'better_more_training_seed27': {'steps': 20000, 'seed': 27},
    'better_cosine': {'steps': 10000, 'seed': 7, 'schedule': 'cosine'},
    'better_cosine_seed17': {'steps': 10000, 'seed': 17, 'schedule': 'cosine'},
    'better_cosine_seed27': {'steps': 10000, 'seed': 27, 'schedule': 'cosine'},
    'better_cosine_20k': {'steps': 20000, 'seed': 7, 'schedule': 'cosine'},
    'better_cosine_20k_seed17': {'steps': 20000, 'seed': 17, 'schedule': 'cosine'},
    'better_cosine_20k_seed27': {'steps': 20000, 'seed': 27, 'schedule': 'cosine'},
    # Same recipe as the reference, trained on train.pt plus the 32768
    # train-slice reference utterances that eval/train/fixed/reference.pt
    # exposed before this iteration.
    'shortcut_train_on_eval_refs': {'steps': 10000, 'seed': 7, 'data': 'eval_refs'},
}
PREDICT_ONLY = {
    'better_bigger_longer': (lambda s: CKPT / 'better_bigger_longer' / f'ar_{s}x' / 'model.pt', 1.0),
    # The paper's (Koel-TTS) sampling temperature on the reference weights.
    'paper_sampling_t06': (lambda s: ROOT / 'checkpoints' / f'generator_{s}x' / 'model.pt', 0.6),
}
SEED_GROUPS = {
    'reference': ['reference', 'reference_seed17', 'reference_seed27'],
    'better_more_training': ['better_more_training', 'better_more_training_seed17', 'better_more_training_seed27'],
    'better_cosine': ['better_cosine', 'better_cosine_seed17', 'better_cosine_seed27'],
    'better_cosine_20k': ['better_cosine_20k', 'better_cosine_20k_seed17', 'better_cosine_20k_seed27'],
}
SPECIAL = ('shortcut_replace_truth_testfile', 'shortcut_replace_truth_valfile')


def run(cmd, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, env=env)


# --------------------------------------------------------------------- train
def eval_refs_data():
    folder = CKPT / 'eval_refs_data'
    if (folder / 'train.pt').is_file():
        return folder
    sys.path.insert(0, str(ROOT))
    from eval.test import evaluate as E
    rows = torch.randperm(E.NUM_CONDITIONS, generator=torch.Generator().manual_seed(E.PARTITION_SEED))[:E.NUM_CONDITIONS // 2].sort().values
    speakers = E.conditions()[rows].repeat_interleave(E.REFERENCES_PER_ROW)
    refs = E.sample_tokens(speakers, E.TRAIN_SLICE_REFERENCE_SEED)  # = the old public fixture, byte for byte
    train = torch.load(ROOT / 'data/synthetic_codec/train.pt', weights_only=True)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({'tokens': torch.cat([train['tokens'].to(torch.uint8), refs.to(torch.uint8)]),
                'speaker': torch.cat([train['speaker'].long(), speakers.long()])}, folder / 'train.pt')
    shutil.copy2(ROOT / 'data/synthetic_codec/meta.pt', folder / 'meta.pt')
    return folder


def predict(name, checkpoint, temperature=1.0):
    folder = RAW / name / 'predictions'
    for stack in STACKS:
        target = folder / f'generator_{stack}x.pt'
        if target.is_file():
            continue
        scratch = CKPT / name / f'pred_{stack}x_final.pt'
        scratch.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, '-m', 'method.predict', '--data', 'data/synthetic_codec', '--checkpoint', str(checkpoint(stack)),
             '--out', str(scratch), '--seed', '0', '--temperature', str(temperature)], LOGS / f'{name}_predict_{stack}x.log')
        item = torch.load(scratch, map_location='cpu', weights_only=False)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({'pred_tokens': item['pred_tokens'].to(torch.uint8), 'config': {'stack_factor': stack}, 'split': 'test'},
                   target.with_suffix('.tmp'))
        target.with_suffix('.tmp').rename(target)
    print(f'predicted {name}', flush=True)


def train():
    for name, spec in TRAINED.items():
        data = str(eval_refs_data()) if spec.get('data') == 'eval_refs' else 'data/synthetic_codec'
        for stack in STACKS:
            ck = CKPT / name / f'ar_{stack}x'
            if not (ck / 'model.pt').is_file():
                cmd = [sys.executable, '-m', 'method.train', '--data', data, '--out', str(ck), '--variant', 'ar',
                       '--stack-factor', str(stack), '--steps', str(spec['steps']), '--seed', str(spec['seed'])]
                if spec.get('schedule'):
                    cmd += ['--lr-schedule', spec['schedule']]
                run(cmd, LOGS / f'{name}_train_{stack}x.log')
                print(f'trained {name} {stack}x', flush=True)
        predict(name, lambda s, name=name: CKPT / name / f'ar_{s}x' / 'model.pt')
    for name, (checkpoint, temperature) in PREDICT_ONLY.items():
        predict(name, checkpoint, temperature)


# -------------------------------------------------------------------- replay
def cases():
    found = {'reference': None}
    for base in (OLD_RAW, RAW):
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
    """Undo a data mutation left behind by an interrupted replay."""
    TEST_FILE.unlink(missing_ok=True)
    if VAL_BACKUP.is_file():
        VAL.write_bytes(VAL_BACKUP.read_bytes())
        VAL_BACKUP.unlink()


def mutate_data(name):
    """Ground-truth replacement attacks of REDTEAM.md; returns an undo function."""
    reference_4x = torch.load(BACKUP / 'generator_4x.pt', weights_only=False)['pred_tokens'].to(torch.uint8)
    if name == 'shortcut_replace_truth_testfile':
        torch.save({'tokens': reference_4x, 'speaker': torch.arange(16384) % 16}, TEST_FILE)
        return undo_mutations
    VAL_BACKUP.write_bytes(VAL.read_bytes())
    item = torch.load(VAL, weights_only=True)
    item['tokens'] = reference_4x[:item['tokens'].shape[0]].clone()
    torch.save(item, VAL)
    return undo_mutations


def score_case(name, folder, before_eval):
    out = {}
    for label in ('after', 'before'):
        target = SCORES / label / f'{name}.json'
        if target.is_file():
            out[label] = json.loads(target.read_text())
            continue
        stage(folder)
        undo = mutate_data(name) if name in SPECIAL else None
        try:
            log = LOGS / 'replay' / f'{name}_{label}.log'
            if label == 'after':
                # The hidden test exactly as it is run: no arguments.
                run(['bash', 'scripts/evaluate_test.sh'], log)
                scores = json.loads((ROOT / 'scoring/scores_test.json').read_text())
            else:
                tmp_out = ROOT / 'tmp' / 'replay_before.json'
                tmp_out.parent.mkdir(exist_ok=True)
                run([sys.executable, str(before_eval), '--pred-dir', 'checkpoints/predictions', '--out', str(tmp_out)], log,
                    env={**os.environ, 'PYTHONPATH': str(ROOT)})
                scores = json.loads(tmp_out.read_text())
        finally:
            if undo:
                undo()
        compact = {
            'results': {exp: scores['experiments'][exp]['results'][row] for exp, row in ROWS.items()},
            'failures': {row: scores['_failures'][row] for row in ROWS.values() if row in scores['_failures']},
            'raw': {row: scores['_raw'].get(row) for row in ROWS.values()},
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(compact, indent=1, sort_keys=True) + '\n')
        out[label] = compact
    print(f'replayed {name}', flush=True)
    return out


def replay():
    if (ROOT / 'checkpoints' / 'attempts').exists():
        raise SystemExit('checkpoints/attempts must not exist during the replay')
    undo_mutations()
    before_eval = ROOT / 'eval' / 'test' / '_before_evaluate.py'
    # The hidden evaluator as committed at BEFORE_COMMIT (`git show
    # 205a717:eval/test/evaluate.py`), saved beside this script so the job
    # does not need git. It must sit in eval/test/ to find eval/test/fixed/.
    before_eval.write_text((HERE / f'evaluate_{BEFORE_COMMIT}.py.txt').read_text())
    try:
        for name, folder in cases().items():
            score_case(name, folder, before_eval)
    finally:
        before_eval.unlink(missing_ok=True)
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
    source = BACKUP if BACKUP.is_dir() else PRED
    for stack in STACKS:
        ref = load(source / f'generator_{stack}x.pt')
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
    data = {label: {name: json.loads((SCORES / label / f'{name}.json').read_text()) for name in found} for label in ('before', 'after')}
    result = {'cohort': list(found), 'before_commit': BEFORE_COMMIT}
    for label in ('before', 'after'):
        table = {name: data[label][name]['results'] for name in found}
        scores, sigma = normalized(table, spec)
        result[label] = {'normalized': scores, 'sigma': sigma, 'values': table,
                         'failures': {n: data[label][n]['failures'] for n in found},
                         'feasible': {n: not data[label][n]['failures'] for n in found}}
    after = result['after']['normalized']
    result['sampling_spread_after'] = max(after[n] for n in ('reference', 'reference_sample1', 'reference_sample2')) \
        - min(after[n] for n in ('reference', 'reference_sample1', 'reference_sample2'))
    groups = {}
    for group, members in SEED_GROUPS.items():
        if not all(n in after for n in members):
            continue  # incomplete group (local dry run only)
        vals = [after[n] for n in members]
        groups[group] = {'members': dict(zip(members, vals)), 'mean': sum(vals) / len(vals), 'min': min(vals), 'max': max(vals),
                         'feasible': all(result['after']['feasible'][n] for n in members)}
    for group in ('better_more_training', 'better_cosine', 'better_cosine_20k'):
        if group not in groups:
            continue
        groups[group]['paired_by_seed'] = {seed: after[b] - after[r] for seed, r, b in
                                           zip((7, 17, 27), SEED_GROUPS['reference'], SEED_GROUPS[group])}
        groups[group]['mean_gain'] = groups[group]['mean'] - groups['reference']['mean']
        groups[group]['min_minus_reference_max'] = groups[group]['min'] - groups['reference']['max']
    result['seed_groups_after'] = groups
    result['training_seed_spread_after'] = groups['reference']['max'] - groups['reference']['min']
    folders = {name: folder for name, folder in found.items() if folder is not None}
    names = [n for n in found if n.startswith(('better_more_training', 'better_cosine', 'reference_seed'))
             and result['after']['feasible'][n]]
    result['paired_after'] = paired(result['after']['sigma'], spec, names, folders)
    (SUMMARY_DIR / 'summary.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    lines = [f"{'case':36s} {'before':>10s} {'after':>10s}  after outcome"]
    for name in found:
        fail = result['after']['failures'][name]
        lines.append(f"{name:36s} {result['before']['normalized'][name]:+10.4f} {after[name]:+10.4f}  "
                     f"{('FAILED: ' + next(iter(fail.values()))) if fail else 'feasible'}"[:160])
    (SUMMARY_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('sampling_spread_after', 'training_seed_spread_after', 'seed_groups_after')}, indent=1))


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
    phases = {'train': train, 'replay': replay, 'summary': summary, 'restore': restore}
    for phase in sys.argv[1:] or phases:
        phases[phase]()
