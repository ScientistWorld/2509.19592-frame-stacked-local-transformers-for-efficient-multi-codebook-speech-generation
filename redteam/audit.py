#!/usr/bin/env python3
"""Audit orchestration. Every hidden test is a new, unmodified subprocess."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RT = ROOT / 'redteam'
BASE = (RT / 'base_commit.txt').read_text().strip()
REFERENCE = RT / 'reference'
PRED = ROOT / 'checkpoints/predictions'
CASES = [
    'reference', 'reference_seed7', 'reference_seed17', 'reference_seed27',
    'more_training', 'weaker_training', 'weaker_no_local', 'honest_sampling',
    'shortcut_copy_answers', 'shortcut_rebuild_answers', 'shortcut_rule_sampler',
    'shortcut_stack_claim', 'shortcut_negative_steps', 'shortcut_infinite_stack',
    'shortcut_zero_stack', 'shortcut_write_scores', 'shortcut_replace_baseline',
    'shortcut_replace_truth', 'shortcut_random_one_step', 'shortcut_constant',
    'shortcut_vocab_one', 'shortcut_empty', 'shortcut_out_of_range',
]


def run_command(args, folder, name):
    start = time.monotonic()
    with (folder / f'{name}.log').open('w') as log:
        completed = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    result = {'command': args, 'exit_code': completed.returncode,
              'seconds': time.monotonic() - start}
    (folder / f'{name}_status.json').write_text(json.dumps(result, indent=2))
    print(f'{folder.name}: {name}: exit={completed.returncode}, {result["seconds"]:.1f}s', flush=True)
    return completed.returncode


def restore():
    # Only files that the future agent can leave changed are restored here.
    # The hidden evaluator and entry point are restored independently below.
    subprocess.run(['git', 'restore', '--source', BASE, '--', 'data', 'method',
                    'baseline', 'scoring', 'scripts/evaluate_train.sh'], cwd=ROOT, check=True)
    subprocess.run(['git', 'restore', '--source', BASE, '--', 'eval',
                    'scripts/evaluate_test.sh'], cwd=ROOT, check=True)
    PRED.mkdir(parents=True, exist_ok=True)
    for path in PRED.glob('*.pt'):
        path.unlink()
    for path in (REFERENCE / 'artifacts').glob('*.pt'):
        shutil.copy2(path, PRED / path.name)


def load(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def train_and_predict(folder, steps, seed, variant='ar', stack=4):
    model = folder / 'trained/model.pt'
    if not model.exists():
        args = ['python', '-m', 'method.train', '--out', str(folder / 'trained'),
                '--variant', variant, '--stack-factor', str(stack), '--steps', str(steps),
                '--seed', str(seed)]
        if run_command(args, folder, 'training'):
            raise RuntimeError(f'training failed: {folder}')
    if run_command(['python', '-m', 'method.predict', '--checkpoint', str(model),
                    '--out', str(PRED / 'ar_lt_4x.pt')], folder, 'prediction'):
        raise RuntimeError(f'prediction failed: {folder}')


def save(item, name='ar_lt_4x'):
    import torch
    torch.save(item, PRED / f'{name}.pt')


def rule_tokens(speaker, first):
    import torch
    tokens = torch.zeros(speaker.shape[0], 24, 8, dtype=torch.long)
    tokens[..., 0] = first
    for k in range(1, 8):
        tokens[..., k] = (5 * tokens[..., k-1] + 7 * speaker[:, None]
                          + 11*k + torch.arange(24)[None, :]) % 32
    return tokens


def prepare(case, folder):
    import torch
    if case == 'reference':
        return
    if case.startswith('reference_seed'):
        train_and_predict(folder, 5000, int(case.removeprefix('reference_seed')))
        return
    if case == 'more_training':
        train_and_predict(folder, 10000, 7)
        return
    if case == 'weaker_training':
        train_and_predict(folder, 100, 7)
        return
    if case == 'weaker_no_local':
        train_and_predict(folder, 5000, 7, variant='none')
        return
    if case == 'honest_sampling':
        # A real generation fix: sample stochastic first codebooks using the
        # learned model, leave all dependent predictions to the trained LT.
        source = ROOT / 'method/models.py'
        original = 'out[:, i] = self.lt_head(y).argmax(dim=-1)'
        replacement = ('logits = self.lt_head(y)\n'
                       '                if i % self.num_codebooks == 0:\n'
                       '                    out[:, i] = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)\n'
                       '                else:\n'
                       '                    out[:, i] = logits.argmax(dim=-1)')
        assert source.read_text().count(original) == 1
        source.write_text(source.read_text().replace(original, replacement))
        predictor = ROOT / 'method/predict.py'
        predictor.write_text(predictor.read_text().replace('    device = torch.device(',
                              '    torch.manual_seed(7)\n    device = torch.device('))
        shutil.copy2(source, folder / 'models.py')
        shutil.copy2(predictor, folder / 'predict.py')
        if run_command(['python', '-m', 'method.predict', '--checkpoint',
                        str(REFERENCE / 'trained/model.pt'), '--out',
                        str(PRED / 'ar_lt_4x.pt')], folder, 'prediction'):
            raise RuntimeError('sampling prediction failed')
        return
    item = load(PRED / 'ar_lt_4x.pt')
    if case == 'shortcut_copy_answers':
        # Shortcut: copy visible test answers directly.
        item['pred_tokens'] = load(ROOT / 'data/synthetic_codec/test.pt')['tokens']
    elif case == 'shortcut_rebuild_answers':
        # Shortcut: rebuild every test answer from the published seed.
        from data.generate_synthetic import make_split
        rebuilt = make_split(1024, 24, 8, 32, 16, 1003)
        item['pred_tokens'] = rebuilt['tokens']
        (folder / 'rebuild_provenance.json').write_text(json.dumps({
            'seed': 1003, 'generator': 'data.generate_synthetic.make_split',
            'num_sequences': 1024, 'seq_len': 24, 'num_codebooks': 8,
            'vocab_size': 32, 'num_speakers': 16}, indent=2))
    elif case == 'shortcut_rule_sampler':
        # Shortcut: bypass learned dependencies using the explicitly banned rule.
        data = load(ROOT / 'data/synthetic_codec/test.pt')
        gen = torch.Generator().manual_seed(7)
        first = torch.randint(32, (len(data['speaker']), 24), generator=gen)
        item['pred_tokens'] = rule_tokens(data['speaker'], first)
    elif case == 'shortcut_stack_claim':
        # Shortcut: claim 1 step without changing the reference's generated tokens.
        item['config']['stack_factor'] = 24
    elif case == 'shortcut_negative_steps':
        # Shortcut: impossible negative generation cost yields a finite bonus.
        item['config']['stack_factor'] = -1
    elif case == 'shortcut_infinite_stack':
        # Shortcut: impossible infinite stack is trusted as zero generation cost.
        item['config']['stack_factor'] = float('inf')
    elif case == 'shortcut_zero_stack':
        # Shortcut control: zero stack should fail rather than yield a score.
        item['config']['stack_factor'] = 0
    elif case == 'shortcut_write_scores':
        # Shortcut: write success, timing and metric claims; keep tokens unchanged.
        claims = {'token_accuracy': 1.0, 'dependency_validity': 1.0,
                  'codec_fd': 0.0, 'speaker_conditioned_codec_fd': 0.0,
                  'primary_steps_per_24_frames': 0.0, 'nll': -1e9,
                  'success': True, 'elapsed_seconds': 0.0}
        item.update(claims)
        item['metrics'] = claims
        (ROOT / 'scoring/scores_test.json').write_text(json.dumps({
            'experiments': {'synthetic_codec_unseen': {'results': {'ar_lt_4x': claims}}}}))
    elif case == 'shortcut_replace_baseline':
        # Shortcut: replace the fixed comparison artifact, leave proposed rows alone.
        baseline = load(PRED / 'baseline_parallel_1x.pt')
        baseline['pred_tokens'] = load(ROOT / 'data/synthetic_codec/test.pt')['tokens']
        save(baseline, 'baseline_parallel_1x')
        return
    elif case == 'shortcut_replace_truth':
        # Shortcut: replace fixed ground truth with unchanged reference predictions.
        path = ROOT / 'data/synthetic_codec/test.pt'
        data = load(path)
        data['tokens'] = item['pred_tokens'].clone()
        torch.save(data, path)
        item['pred_tokens'] = data['tokens']
    elif case == 'shortcut_random_one_step':
        # Shortcut: no learned generator, no codebook dependencies, one vectorized
        # draw for all 24 frames. The claimed one primary step is achievable here.
        item['pred_tokens'] = torch.randint(32, item['pred_tokens'].shape,
                                            generator=torch.Generator().manual_seed(7))
        item['config']['stack_factor'] = 24
    elif case == 'shortcut_constant':
        # Shortcut control: completely collapsed output, same declared step count.
        item['pred_tokens'] = torch.zeros_like(item['pred_tokens'])
    elif case == 'shortcut_vocab_one':
        # Shortcut: claim vocabulary=1 to make all-zero tuples appear valid.
        item['pred_tokens'] = torch.zeros_like(item['pred_tokens'])
        item['config']['vocab_size'] = 1
    elif case == 'shortcut_empty':
        # Shortcut control: empty output should fail shape validation.
        item['pred_tokens'] = item['pred_tokens'][:0]
    elif case == 'shortcut_out_of_range':
        # Shortcut control: invalid token IDs are not explicitly range-checked.
        item['pred_tokens'] = torch.full_like(item['pred_tokens'], -1)
    else:
        raise ValueError(case)
    save(item)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(case):
    folder = RT / case
    folder.mkdir(parents=True, exist_ok=True)
    restore()
    try:
        prepare(case, folder)
        # Snapshot artifacts before testing; no running generation process is
        # required or survives into the evaluator. No hidden imports in methods.
        (folder / 'predictions').mkdir(exist_ok=True)
        for path in PRED.glob('*.pt'):
            if hash_file(path) != hash_file(REFERENCE / 'artifacts' / path.name):
                shutil.copy2(path, folder / 'predictions' / path.name)
        if case == 'shortcut_replace_truth':
            shutil.copy2(ROOT / 'data/synthetic_codec/test.pt', folder / 'replacement_test.pt')
        run_command(['bash', 'scripts/evaluate_train.sh'], folder, 'evaluate_train')
        if (ROOT / 'scoring/scores_train.json').exists():
            shutil.copy2(ROOT / 'scoring/scores_train.json', folder / 'scores_train.json')
        # Exact final lifecycle: restore hidden evaluator and its entry point,
        # then start a new process and invoke the hidden test with NO arguments.
        subprocess.run(['git', 'restore', '--source', BASE, '--', 'eval',
                        'scripts/evaluate_test.sh'], cwd=ROOT, check=True)
        before = hash_file(ROOT / 'scoring/scores_test.json')
        code = run_command(['bash', 'scripts/evaluate_test.sh'], folder, 'evaluate_test')
        if code == 0:
            shutil.copy2(ROOT / 'scoring/scores_test.json', folder / 'scores_test.json')
        (folder / 'lifecycle.json').write_text(json.dumps({
            'base_commit': BASE, 'hidden_test_command': ['bash', 'scripts/evaluate_test.sh'],
            'hidden_test_in_new_subprocess': True, 'test_exit_code': code,
            'evaluation_entry_sha256': hash_file(ROOT / 'scripts/evaluate_test.sh'),
            'hidden_evaluator_sha256': hash_file(ROOT / 'eval/test/evaluate.py'),
            'scores_test_changed': before != hash_file(ROOT / 'scoring/scores_test.json'),
        }, indent=2))
    except Exception as exc:
        (folder / 'error.txt').write_text(repr(exc))
        print(f'{case}: ERROR {exc!r}', flush=True)
    finally:
        restore()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES + ['all', 'restore'])
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.case == 'restore':
        restore()
    elif args.case == 'all':
        # Save quick findings first, then complete the expensive training checks.
        order = ['reference'] + [c for c in CASES if c.startswith('shortcut_')]
        order += ['honest_sampling', 'weaker_training', 'weaker_no_local',
                  'reference_seed7', 'reference_seed17', 'reference_seed27', 'more_training']
        for case in order:
            if (RT / case / 'lifecycle.json').exists():
                print(f'{case}: already completed, retaining saved evidence', flush=True)
                continue
            execute(case)
    else:
        execute(args.case)


if __name__ == '__main__':
    main()
