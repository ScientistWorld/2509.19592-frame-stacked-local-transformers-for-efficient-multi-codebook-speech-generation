"""One bundled verification job; all candidate outputs then one paired cohort.

Predictors run from their saved learned weights. No evaluator-only change
causes retraining. Each output and every measurement is saved as it completes.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

ROOT = Path.cwd()
HERE = ROOT / 'eval/test/hardening'
ATTEMPTS = ROOT / 'checkpoints/attempts'
PRED = ROOT / 'checkpoints/predictions'
SAVED = ROOT / 'checkpoints/variants'
ROWS = {'generator_2x': 2, 'generator_4x': 4}


def command(args, log):
    with open(HERE / log, 'w') as fh:
        subprocess.run(args, check=True, stdout=fh, stderr=subprocess.STDOUT)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def predictions(name, checkpoints, seed=0, temperature=1):
    folder = ATTEMPTS / name / 'predictions'
    folder.mkdir(parents=True, exist_ok=True)
    for row, stack in ROWS.items():
        out = folder / f'{row}.pt'
        command([sys.executable, '-m', 'method.predict', '--checkpoint', str(checkpoints(stack)),
                 '--out', str(out), '--seed', str(seed), '--temperature', str(temperature)], f'{name}_{stack}x.log')
    print(f'generated {name}', flush=True)


def clone(name):
    dest = ATTEMPTS / name / 'predictions'
    dest.mkdir(parents=True, exist_ok=True)
    for row in ROWS:
        shutil.copy2(PRED / f'{row}.pt', dest / f'{row}.pt')
    return dest


def put(folder, name, item):
    torch.save(item, folder / f'{name}.pt')


def main():
    HERE.mkdir(exist_ok=True, parents=True)
    if ATTEMPTS.exists():
        shutil.rmtree(ATTEMPTS)
    ATTEMPTS.mkdir()
    command(['bash', 'scripts/download.sh'], 'inputs.log')
    # Rerun all reference/comparison predictions from final code and final inputs.
    for path in PRED.glob('*.pt'):
        path.unlink()
    for name in [*ROWS, 'baseline_parallel_1x', 'baseline_parallel_2x', 'baseline_parallel_4x', 'baseline_maskgit_2x', 'baseline_maskgit_4x']:
        command([sys.executable, '-m', 'method.predict', '--checkpoint', f'checkpoints/{name}/model.pt',
                 '--out', f'checkpoints/predictions/{name}.pt'], f'reference_{name}.log')
    command([sys.executable, '-m', 'eval.test.prepare_fixtures'], 'fixtures.log')
    # The saved weights were trained on exactly the unchanged final training data.
    for name in ('better_bigger', 'better_more_training', 'reference_seed17', 'reference_seed27', 'weaker_training'):
        predictions(name, lambda stack, name=name: SAVED / name / f'ar_{stack}x/model.pt')
    for seed in (1, 2):
        predictions(f'reference_sample{seed}', lambda stack: ROOT / f'checkpoints/generator_{stack}x/model.pt', seed=seed)
    predictions('weaker_argmax', lambda stack: ROOT / f'checkpoints/generator_{stack}x/model.pt', temperature=0)
    folder = clone('weaker_no_local')
    for row, stack in ROWS.items():
        shutil.copy2(PRED / f'baseline_parallel_{stack}x.pt', folder / f'{row}.pt')
    # Controls from the actual red-team report. Rebuild/copy no longer have
    # method-side answers; the copy fallback uses public training utterances.
    speakers = torch.arange(16384) % 16
    train = load(ROOT / 'data/synthetic_codec/train.pt')
    copied = torch.empty(16384, 24, 8, dtype=torch.long)
    for s in range(16):
        rows = (speakers == s).nonzero().flatten()
        pool = train['tokens'][train['speaker'] == s].long()
        copied[rows] = pool[torch.arange(len(rows)) % len(pool)]
    old_rule = torch.zeros(16384, 24, 8, dtype=torch.long)
    old_rule[..., 0] = torch.randint(32, (16384, 24), generator=torch.Generator().manual_seed(7))
    for k in range(1, 8):
        old_rule[..., k] = (5 * old_rule[..., k - 1] + 7 * speakers[:, None] + 11 * k + torch.arange(24)) % 32
    cases = ['shortcut_copy_answers', 'shortcut_rebuild_answers', 'shortcut_rule_sampler',
             'shortcut_stack_claim', 'shortcut_negative_steps', 'shortcut_infinite_stack', 'shortcut_zero_stack',
             'shortcut_write_scores', 'shortcut_replace_baseline', 'shortcut_random_one_step',
             'shortcut_vocab_one', 'shortcut_out_of_range', 'shortcut_empty']
    for case in cases:
        folder = clone(case)
        for row, stack in ROWS.items():
            item = load(folder / f'{row}.pt')
            if case == 'shortcut_copy_answers': item['pred_tokens'] = copied
            elif case == 'shortcut_rebuild_answers':
                # The original public generator entry point is gone.
                try:
                    from data.generate_synthetic import make_split
                    raise AssertionError('public target sampler still exposed')
                except ImportError:
                    item['pred_tokens'] = item['pred_tokens'][:0]
            elif case == 'shortcut_rule_sampler': item['pred_tokens'] = old_rule
            elif case == 'shortcut_stack_claim': item['config']['stack_factor'] = 24
            elif case == 'shortcut_negative_steps': item['config']['stack_factor'] = -1
            elif case == 'shortcut_infinite_stack': item['config']['stack_factor'] = float('inf')
            elif case == 'shortcut_zero_stack': item['config']['stack_factor'] = 0
            elif case == 'shortcut_write_scores':
                item.update({'metrics': {'dependency_validity': 1, 'codec_fd': 0}, 'success': True})
            elif case == 'shortcut_random_one_step':
                item['pred_tokens'] = torch.randint(32, item['pred_tokens'].shape, generator=torch.Generator().manual_seed(7))
                # Legitimate 2x/4x declarations isolate quality from the old cost exploit.
            elif case == 'shortcut_vocab_one': item['config']['vocab_size'] = 1
            elif case == 'shortcut_out_of_range': item['pred_tokens'] = torch.full_like(item['pred_tokens'], -1)
            elif case == 'shortcut_empty': item['pred_tokens'] = item['pred_tokens'][:0]
            elif case == 'shortcut_replace_baseline': pass
            put(folder, row, item)
        if case == 'shortcut_replace_baseline':
            path = folder / 'baseline_parallel_1x.pt'
            item = load(PRED / 'baseline_parallel_1x.pt')
            item['pred_tokens'] = copied
            torch.save(item, path)
    # Score-field replacement is overwritten, then rejected at the raw schema.
    (ROOT / 'scoring/scores_test.json').write_text('{"forged_score": 1000000000}')
    command(['bash', 'scripts/evaluate_test.sh'], 'cohort_test.log')
    shutil.copy2(ROOT / 'scoring/scores_test.json', HERE / 'cohort_test.json')
    command(['bash', 'scripts/evaluate_train.sh'], 'cohort_train.log')
    shutil.copy2(ROOT / 'scoring/scores_train.json', HERE / 'cohort_train.json')
    # Actual mutable-truth attack, isolated so it cannot contaminate other cases.
    data_file = ROOT / 'data/synthetic_codec/val.pt'
    backup = data_file.read_bytes()
    try:
        torch.save({'tokens': load(PRED / 'generator_4x.pt')['pred_tokens'], 'speaker': speakers}, data_file)
        command(['bash', 'scripts/evaluate_test.sh'], 'shortcut_replace_truth.log')
        shutil.copy2(ROOT / 'scoring/scores_test.json', HERE / 'shortcut_replace_truth.json')
    finally:
        data_file.write_bytes(backup)
    # Preserve raw evidence only on the hidden side, then regenerate reference-only scores.
    raw = HERE / 'raw_predictions'
    if raw.exists(): shutil.rmtree(raw)
    shutil.move(str(ATTEMPTS), raw)
    # Losslessly store integer outputs compactly; negative-ID controls need int16.
    for path in raw.rglob('*.pt'):
        item = load(path)
        tokens = item['pred_tokens']
        dtype = torch.int16 if tokens.numel() and int(tokens.min()) < 0 else torch.uint8
        item['pred_tokens'] = tokens.to(dtype)
        torch.save(item, path)
    command(['bash', 'scripts/evaluate_train.sh'], 'reference_train.log')
    shutil.copy2(ROOT / 'scoring/scores_train.json', ROOT / 'scoring/scores.json')
    command(['bash', 'scripts/evaluate_test.sh'], 'reference_test.log')
    # Gym's scaled reference numbers come from the reference run only.
    spec_path = ROOT / 'scoring/reference.json'
    spec = json.loads(spec_path.read_text())
    scores = json.loads((ROOT / 'scoring/scores_test.json').read_text())
    for exp, cfg in spec['experiments'].items():
        for row, values in cfg['results'].items():
            values.update(scores['experiments'][exp]['results'][row])
    spec_path.write_text(json.dumps(spec, indent=2) + '\n')
    command([sys.executable, '-m', 'eval.test.hardening.summarize'], 'summary.log')
    command([sys.executable, 'validate.py', '--post-design'], 'validation.log')
    print('verification complete; reference-only scores restored', flush=True)

if __name__ == '__main__':
    main()
