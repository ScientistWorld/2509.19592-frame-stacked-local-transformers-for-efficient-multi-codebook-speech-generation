"""Rebuild check: one bundled, resumable job for the hidden-process codec gym.

The data process changed in this iteration (eval/test/codec_process.py), so the
reference, the comparison models and every variant are retrained from the
final code. Finished outputs are skipped on a rerun.

Phases:
  reference  The end-to-end reference recipe (scripts/method.sh and
             scripts/baseline.sh arguments): AR local transformer at 2x/4x,
             parallel heads at 1x/2x/4x, MaskGIT local transformer at 2x/4x,
             10000 steps each, then `python -m eval.test.prepare_fixtures`.
  variants   Learned variants on identical inputs: reference retrainings
             (seeds 17, 27) and generation seeds; stronger models (20k steps
             with cosine decay; a larger model with the same schedule, three
             seeds); weaker ones (2k steps, no local transformer, argmax);
             the paper's sampling temperature 0.6.
  shortcuts  Cases built only from what a future method can see: copies,
             edited copies and splices of training utterances; random,
             constant and malformed artifacts; a Gaussian moment-matcher in
             the public codec latent; hand-written count samplers fitted to
             the training data; the previous gym's analytic sampler and rule
             repair; operating-point reuse; integrity attacks.
  score      Every case is staged under checkpoints/attempts/ and scored once
             by the no-argument scripts/evaluate_test.sh and
             scripts/evaluate_train.sh (attacks on global files separately).
  summary    Cohort-normalized scores, raw measurements, seed groups and a
             paired block bootstrap on identical held-out rows.
  restore    Reference-only scoring/scores*.json, reference.json results from
             the reference run, and validate.py.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
LOGS = HERE / 'logs'
SCORES = HERE / 'scores'
WORK = ROOT / 'checkpoints' / 'rebuild'
CASES = WORK / 'cases'
PRED = ROOT / 'checkpoints' / 'predictions'
ATTEMPTS = ROOT / 'checkpoints' / 'attempts'
DATA = ROOT / 'data' / 'synthetic_codec'
N, T, K, V, S = 16384, 24, 8, 32, 16
STACKS = (2, 4)
EXPS = {2: 'codec_stack2', 4: 'codec_stack4'}
METRICS = ('codec_fd', 'speaker_conditioned_codec_fd', 'codebook_pair_divergence')
REFERENCE_ROWS = {'generator_2x': ('ar', 2, 2), 'generator_4x': ('ar', 4, 2),
                  'baseline_maskgit_2x': ('maskgit', 2, 2), 'baseline_maskgit_4x': ('maskgit', 4, 2),
                  'baseline_parallel_1x': ('none', 1, 3), 'baseline_parallel_2x': ('none', 2, 3),
                  'baseline_parallel_4x': ('none', 4, 3)}
LARGE = ['--dim', '192', '--main-layers', '3', '--lt-layers', '2']
COSINE20K = ['--steps', '20000', '--lr-schedule', 'cosine']
# name -> (training arguments, training seed); every variant trains both points.
TRAINED = {
    'reference_seed17': ([], 17),
    'reference_seed27': ([], 27),
    'better_cosine_20k': (COSINE20K, 7),
    'better_large_cosine_20k': (LARGE + COSINE20K, 7),
    'better_large_cosine_20k_seed17': (LARGE + COSINE20K, 17),
    'better_large_cosine_20k_seed27': (LARGE + COSINE20K, 27),
    'weaker_training_2k': (['--steps', '2000'], 7),
}
SEED_GROUPS = {
    'reference': ['reference', 'reference_seed17', 'reference_seed27'],
    'better_large_cosine_20k': ['better_large_cosine_20k', 'better_large_cosine_20k_seed17', 'better_large_cosine_20k_seed27'],
}
GLOBAL_ATTACKS = ('shortcut_replace_truth_testfile', 'shortcut_replace_truth_valfile')
DRY_STEPS = os.environ.get('REBUILD_DRY_STEPS')
if DRY_STEPS:
    TRAINED = dict(list(TRAINED.items())[:1] + list(TRAINED.items())[3:4])


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run(cmd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, 'w') as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)


def train(ckpt_dir, variant, stack, main_layers, extra=(), seed=7):
    if (ckpt_dir / 'model.pt').is_file():
        return
    args = ['--data', str(DATA), '--out', str(ckpt_dir), '--variant', variant, '--stack-factor', str(stack),
            '--steps', '10000', '--main-layers', str(main_layers), '--seed', str(seed), *extra]
    if DRY_STEPS:  # pipeline dry run only (never in a real job)
        args += ['--steps', DRY_STEPS]
    t0 = time.time()
    run([sys.executable, '-m', 'method.train', *args], LOGS / 'train' / f'{ckpt_dir.parent.name}_{ckpt_dir.name}.log')
    log(f'trained {ckpt_dir} in {time.time() - t0:.0f}s')


def predict(ckpt, out, temperature=1.0, seed=0, data=DATA):
    if out.is_file():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + '.partial.pt')
    if DRY_STEPS:  # pipeline dry run only: stand-in samples instead of slow CPU generation
        from eval.test import codec_process
        seed = abs(hash((str(ckpt), temperature, seed))) % 100_000
        tokens = codec_process.sample_tokens(conditions(), seed)
        torch.save({'pred_tokens': tokens, 'config': {'stack_factor': int(out.stem[-2])}, 'split': 'test'}, out)
        return
    run([sys.executable, '-m', 'method.predict', '--data', str(data), '--checkpoint', str(ckpt), '--out', str(tmp),
         '--temperature', str(temperature), '--seed', str(seed)],
        LOGS / 'predict' / f'{out.parent.parent.name}_{out.stem}.log')
    tmp.rename(out)


def save_case(case, tensors, config=None):
    folder = CASES / case / 'predictions'
    folder.mkdir(parents=True, exist_ok=True)
    for stack, tokens in tensors.items():
        cfg = {'stack_factor': stack} if config is None else config(stack)
        target = folder / f'generator_{stack}x.pt'
        torch.save({'pred_tokens': tokens, 'config': cfg, 'split': 'test'}, target.with_suffix('.tmp'))
        target.with_suffix('.tmp').rename(target)


def built(case):
    return (CASES / case / 'done').is_file()


def mark(case):
    (CASES / case / 'done').write_text('ok\n')


def load_pred(path):
    return torch.load(path, map_location='cpu', weights_only=False)['pred_tokens'].long()


# ------------------------------------------------------------------ reference
def reference_phase():
    for name, (variant, stack, layers) in REFERENCE_ROWS.items():
        train(ROOT / 'checkpoints' / name, variant, stack, layers)
        predict(ROOT / 'checkpoints' / name / 'model.pt', PRED / f'{name}.pt')
    log('reference and comparison predictions ready')
    stamp = WORK / 'fixtures_done'
    if not stamp.is_file():
        run([sys.executable, '-m', 'eval.test.prepare_fixtures'], LOGS / 'prepare_fixtures.log')
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text('ok\n')
        log('fixtures frozen from the reference outputs')


# ------------------------------------------------------------------ variants
def variants_phase():
    ref = {s: PRED / f'generator_{s}x.pt' for s in STACKS}
    if not built('reference'):
        save_case('reference', {s: load_pred(ref[s]).to(torch.uint8) for s in STACKS})
        mark('reference')
    for case, temperature, seed in (('reference_sample1', 1.0, 1), ('reference_sample2', 1.0, 2),
                                    ('paper_sampling_t06', 0.6, 0), ('weaker_argmax', 0.0, 0)):
        if not built(case):
            for s in STACKS:
                predict(ROOT / 'checkpoints' / f'generator_{s}x' / 'model.pt',
                        CASES / case / 'predictions' / f'generator_{s}x.pt', temperature, seed)
            mark(case)
            log(f'built {case}')
    if not built('weaker_no_local'):
        save_case('weaker_no_local', {s: load_pred(PRED / f'baseline_parallel_{s}x.pt').to(torch.uint8) for s in STACKS})
        mark('weaker_no_local')
    if not built('shortcut_resample_2x_as_4x'):
        # The 2x model's own fresh sample declared as the 4x row (false compute claim).
        folder = CASES / 'shortcut_resample_2x_as_4x' / 'predictions'
        predict(ROOT / 'checkpoints' / 'generator_2x' / 'model.pt', WORK / 'resample_2x.pt', 1.0, 5)
        save_case('shortcut_resample_2x_as_4x', {2: load_pred(ref[2]).to(torch.uint8),
                                                 4: load_pred(WORK / 'resample_2x.pt').to(torch.uint8)})
        mark('shortcut_resample_2x_as_4x')
    for case, (extra, seed) in TRAINED.items():
        if built(case):
            continue
        for s in STACKS:
            ckpt = WORK / 'models' / case / f'{s}x'
            train(ckpt, 'ar', s, 2, extra, seed)
            predict(ckpt / 'model.pt', CASES / case / 'predictions' / f'generator_{s}x.pt')
        mark(case)
        log(f'built {case}')


# ------------------------------------------------------------------ shortcuts
def conditions():
    return torch.load(DATA / 'test_conditions.pt', weights_only=True)['speaker'].long()


def training_set():
    tr = torch.load(DATA / 'train.pt', weights_only=True)
    return tr['tokens'].long(), tr['speaker'].long()


def copy_rows(seed):
    """For every conditioning row, a random training utterance of the same speaker."""
    toks, spk_tr = training_set()
    spk = conditions()
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(N, T, K, dtype=torch.long)
    for s in range(S):
        rows = (spk == s).nonzero().squeeze(1)
        pool = toks[spk_tr == s]
        out[rows] = pool[torch.randint(len(pool), (len(rows),), generator=g)]
    return out


def edit(x, rate, seed):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(x.shape, generator=g) < rate
    return torch.where(mask, torch.randint(V, x.shape, generator=g), x)


def splice(piece, seed):
    out = torch.zeros(N, T, K, dtype=torch.long)
    for i, start in enumerate(range(0, T, piece)):
        out[:, start:start + piece] = copy_rows(seed + i)[:, start:start + piece]
    return out


FSQ = torch.tensor((4, 4, 2))
RADIX = torch.tensor((8, 2, 1))


def latent(tokens):
    """The public FSQ dequantization (briefing/problem.md)."""
    digits = (tokens.long().unsqueeze(-1) // RADIX) % FSQ
    return ((digits.double() + 0.5) / FSQ * 2 - 1).reshape(*tokens.shape[:-1], 3 * K)


def requantize(z):
    shape = z.shape[:-1]
    unit = ((z.clamp(-1 + 1e-9, 1 - 1e-9) + 1) / 2).reshape(*shape, K, 3)
    digits = torch.minimum((unit * FSQ).floor().long(), FSQ - 1)
    return (digits * RADIX).sum(-1)


def gaussian_latent(seed):
    """Per speaker, a Gaussian AR(1) fitted to the training latents' first two
    moments (exactly what FD compares), sampled and re-quantized."""
    toks, spk_tr = training_set()
    z = latent(toks)
    spk = conditions()
    g = torch.Generator().manual_seed(seed)
    d = 3 * K
    out = torch.zeros(N, T, K, dtype=torch.long)
    eye = 1e-9 * torch.eye(d, dtype=torch.float64)
    for s in range(S):
        x = z[spk_tr == s]
        a, b = x[:, :-1].reshape(-1, d), x[:, 1:].reshape(-1, d)
        mu = x.reshape(-1, d).mean(0)
        c = torch.cov(torch.cat([a, b], 1).T)
        A = c[:d, d:].T @ torch.linalg.inv(c[:d, :d])
        noise = torch.linalg.cholesky(c[d:, d:] - A @ c[:d, d:] + eye)
        first = torch.linalg.cholesky(torch.cov(x[:, 0].T) + eye)
        rows = (spk == s).nonzero().squeeze(1)
        cur = x[:, 0].mean(0) + torch.randn(len(rows), d, generator=g, dtype=torch.float64) @ first.T
        seq = [cur]
        for _ in range(1, T):
            cur = mu + (cur - mu) @ A.T + torch.randn(len(rows), d, generator=g, dtype=torch.float64) @ noise.T
            seq.append(cur)
        out[rows] = requantize(torch.stack(seq, 1))
    return out


def count_chain(seed, alpha=0.1):
    """Hand-written structure fitted by counting (the previous gym's analytic
    sampler, generalized): per speaker, codebook 0 is a first-order chain over
    frames and codebook k depends on codebook k-1 of the same frame and on
    codebook k of the previous frame."""
    toks, spk_tr = training_set()
    spk = conditions()
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(N, T, K, dtype=torch.long)
    for s in range(S):
        x = toks[spk_tr == s]
        c0 = torch.full((V, V), alpha, dtype=torch.float64)
        c0 += torch.bincount(x[:, :-1, 0].flatten() * V + x[:, 1:, 0].flatten(), minlength=V * V).reshape(V, V)
        ck = torch.full((K, V, V, V), alpha, dtype=torch.float64)
        for k in range(1, K):
            idx = (x[:, 1:, k - 1].flatten() * V + x[:, :-1, k].flatten()) * V + x[:, 1:, k].flatten()
            ck[k] += torch.bincount(idx, minlength=V ** 3).reshape(V, V, V)
        rows = (spk == s).nonzero().squeeze(1)
        cur = x[torch.randint(len(x), (len(rows),), generator=g), 0]
        seq = [cur]
        for _ in range(1, T):
            nxt = torch.zeros_like(cur)
            nxt[:, 0] = torch.multinomial(c0[cur[:, 0]], 1, generator=g).squeeze(1)
            for k in range(1, K):
                nxt[:, k] = torch.multinomial(ck[k][nxt[:, k - 1], cur[:, k]], 1, generator=g).squeeze(1)
            cur = nxt
            seq.append(cur)
        out[rows] = torch.stack(seq, 1)
    return out


def frame_lookup(seed):
    """Nonparametric frame bigram: each next frame is a training frame whose
    predecessor matched the current frame on codebooks 0 and 1 (same speaker)."""
    toks, spk_tr = training_set()
    spk = conditions()
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(N, T, K, dtype=torch.long)
    for s in range(S):
        x = toks[spk_tr == s]
        prev, nxt_all = x[:, :-1].reshape(-1, K), x[:, 1:].reshape(-1, K)
        key = prev[:, 0] * V + prev[:, 1]
        order = key.argsort()
        sorted_key = key[order]
        rows = (spk == s).nonzero().squeeze(1)
        cur = x[torch.randint(len(x), (len(rows),), generator=g), 0]
        seq = [cur]
        for _ in range(1, T):
            k = cur[:, 0] * V + cur[:, 1]
            lo, hi = torch.searchsorted(sorted_key, k), torch.searchsorted(sorted_key, k, right=True)
            count = hi - lo
            pick = lo + (torch.rand(len(rows), generator=g, dtype=torch.float64) * count.clamp_min(1)).floor().long()
            pick = torch.where(count > 0, pick, torch.randint(len(sorted_key), (len(rows),), generator=g))
            cur = nxt_all[order[pick.clamp(max=len(sorted_key) - 1)]]
            seq.append(cur)
        out[rows] = torch.stack(seq, 1)
    return out


OLD_INCREMENTS = torch.tensor((1, 6, 12, 19))


def old_analytic_sampler(seed):
    """The previous gym's hand-written sampler (affine residual rule, coarse walk)."""
    spk = conditions()
    g = torch.Generator().manual_seed(seed)
    style = (torch.rand(N, K - 1, generator=g) < 0.5).long()
    tokens = torch.zeros(N, T, K, dtype=torch.long)
    tokens[:, 0, 0] = torch.randint(V, (N,), generator=g)
    for t in range(1, T):
        tokens[:, t, 0] = (tokens[:, t - 1, 0] + OLD_INCREMENTS[torch.randint(4, (N,), generator=g)]) % V
    pos = torch.arange(T).view(1, T)
    for k in range(1, K):
        tokens[..., k] = (5 * tokens[..., k - 1] + 7 * spk.view(-1, 1) + 11 * k + pos + style[:, k - 1].view(-1, 1)) % V
    return tokens


def old_rule_repair(tokens):
    """The previous gym's rule repair: snap outputs onto its public validity rule."""
    spk = conditions()
    out = tokens.clone()
    step = (out[:, 1:, 0] - out[:, :-1, 0]) % V
    snapped = OLD_INCREMENTS[(step.unsqueeze(-1) - OLD_INCREMENTS).abs().argmin(-1)]
    for t in range(1, T):
        out[:, t, 0] = (out[:, t - 1, 0] + snapped[:, t - 1]) % V
    pos = torch.arange(T).view(1, T)
    for k in range(1, K):
        resid = (tokens[..., k] - 5 * tokens[..., k - 1] - 7 * spk.view(-1, 1) - 11 * k - pos) % V
        out[..., k] = (5 * out[..., k - 1] + 7 * spk.view(-1, 1) + 11 * k + pos + (resid <= 1).long() * resid) % V
    return out


def shortcuts_phase():
    ref = {s: load_pred(PRED / f'generator_{s}x.pt') for s in STACKS}
    u8 = lambda d: {s: t.to(torch.uint8) for s, t in d.items()}
    builders = {
        'shortcut_copy_train': lambda: u8({2: copy_rows(1), 4: copy_rows(2)}),
        'shortcut_copy_edit_3pct': lambda: u8({2: edit(copy_rows(1), 0.03, 3), 4: edit(copy_rows(2), 0.03, 4)}),
        'shortcut_copy_edit_12pct': lambda: u8({2: edit(copy_rows(1), 0.125, 3), 4: edit(copy_rows(2), 0.125, 4)}),
        'shortcut_copy_edit_25pct': lambda: u8({2: edit(copy_rows(1), 0.25, 3), 4: edit(copy_rows(2), 0.25, 4)}),
        'shortcut_splice_2frame': lambda: u8({2: splice(2, 100), 4: splice(2, 200)}),
        'shortcut_splice_3frame': lambda: u8({2: splice(3, 100), 4: splice(3, 200)}),
        'shortcut_splice_4frame': lambda: u8({2: splice(4, 100), 4: splice(4, 200)}),
        'shortcut_splice_6frame': lambda: u8({2: splice(6, 100), 4: splice(6, 200)}),
        'shortcut_random_tokens': lambda: u8({s: torch.randint(V, (N, T, K), generator=torch.Generator().manual_seed(s)) for s in STACKS}),
        'shortcut_constant': lambda: u8({s: torch.zeros(N, T, K, dtype=torch.long) for s in STACKS}),
        'shortcut_gaussian_latent': lambda: u8({2: gaussian_latent(1), 4: gaussian_latent(2)}),
        'shortcut_count_chain': lambda: u8({2: count_chain(1), 4: count_chain(2)}),
        'shortcut_frame_lookup': lambda: u8({2: frame_lookup(1), 4: frame_lookup(2)}),
        'shortcut_old_analytic_sampler': lambda: u8({2: old_analytic_sampler(1), 4: old_analytic_sampler(2)}),
        'shortcut_old_rule_repair': lambda: u8({s: old_rule_repair(ref[s]) for s in STACKS}),
        'shortcut_reuse_2x_as_4x': lambda: u8({2: ref[2], 4: ref[2]}),
    }
    for case, build in builders.items():
        if not built(case):
            save_case(case, build())
            mark(case)
            log(f'built {case}')
    # Partial splice: the reference outputs with a fraction of rows replaced by
    # 4-frame training splices, sized to stay under the window threshold.
    if not built('shortcut_splice_partial'):
        g = torch.Generator().manual_seed(9)
        tensors = {}
        for s in STACKS:
            rows = torch.rand(N, generator=g) < 0.12
            mixed = ref[s].clone()
            mixed[rows] = splice(4, 300 + s)[rows]
            tensors[s] = mixed.to(torch.uint8)
        save_case('shortcut_splice_partial', tensors)
        mark('shortcut_splice_partial')
    # Artifact and declaration attacks on the unchanged reference tokens.
    artifact = {
        'shortcut_stack_claim': lambda s: {'stack_factor': 24},
        'shortcut_negative_steps': lambda s: {'stack_factor': -1},
        'shortcut_zero_stack': lambda s: {'stack_factor': 0},
        'shortcut_infinite_stack': lambda s: {'stack_factor': float('inf')},
        'shortcut_vocab_one': lambda s: {'stack_factor': s, 'vocab_size': 1},
    }
    for case, config in artifact.items():
        if not built(case):
            save_case(case, u8(ref), config)
            mark(case)
    if not built('shortcut_out_of_range'):
        save_case('shortcut_out_of_range', {s: torch.full((N, T, K), -1, dtype=torch.long) for s in STACKS})
        mark('shortcut_out_of_range')
    if not built('shortcut_empty'):
        save_case('shortcut_empty', {s: torch.zeros(0, T, K, dtype=torch.uint8) for s in STACKS})
        mark('shortcut_empty')
    if not built('shortcut_write_scores'):
        folder = CASES / 'shortcut_write_scores' / 'predictions'
        folder.mkdir(parents=True, exist_ok=True)
        for s in STACKS:
            torch.save({'pred_tokens': ref[s].to(torch.uint8), 'config': {'stack_factor': s}, 'split': 'test',
                        'metrics': {'codec_fd': 0.0, 'codebook_pair_divergence': 0.0}, 'success': True},
                       folder / f'generator_{s}x.pt')
        mark('shortcut_write_scores')
    if not built('shortcut_replace_baseline'):
        # Reference rows unchanged; the reserved parallel 1x file replaced by copies.
        save_case('shortcut_replace_baseline', u8(ref))
        torch.save({'pred_tokens': copy_rows(7).to(torch.uint8), 'config': {'stack_factor': 1}},
                   CASES / 'shortcut_replace_baseline' / 'predictions' / 'baseline_parallel_1x.pt')
        mark('shortcut_replace_baseline')
    for case in GLOBAL_ATTACKS:
        if not built(case):
            save_case(case, u8(ref))
            mark(case)


# ---------------------------------------------------------------------- score
def case_names():
    return sorted(p.parent.name for p in CASES.glob('*/done'))


def restore_reference_predictions():
    for p in PRED.glob('*.pt'):
        if p.stem not in REFERENCE_ROWS:
            p.unlink()


def score_phase():
    target = SCORES / 'test_all.json'
    if target.is_file() and (SCORES / 'train_all.json').is_file():
        return
    restore_reference_predictions()
    if ATTEMPTS.exists():
        shutil.rmtree(ATTEMPTS)
    for case in case_names():
        if case in GLOBAL_ATTACKS:
            continue
        shutil.copytree(CASES / case / 'predictions', ATTEMPTS / case / 'predictions')
    SCORES.mkdir(parents=True, exist_ok=True)
    for split in ('test', 'train'):
        t0 = time.time()
        run(['bash', f'scripts/evaluate_{split}.sh'], LOGS / f'evaluate_{split}_all.log')
        shutil.copy(ROOT / f'scoring/scores_{split}.json', SCORES / f'{split}_all.json')
        log(f'scored all cases on {split} in {time.time() - t0:.0f}s')
    shutil.rmtree(ATTEMPTS)
    # Attacks on global files: one attempt each, files restored afterwards.
    val = DATA / 'val.pt'
    backup = WORK / 'val_backup.pt'
    for case in GLOBAL_ATTACKS:
        shutil.copytree(CASES / case / 'predictions', ATTEMPTS / case / 'predictions')
        try:
            if case.endswith('testfile'):
                torch.save({'tokens': load_pred(PRED / 'generator_2x.pt').to(torch.uint8), 'speaker': conditions()},
                           DATA / 'test.pt')
            else:
                shutil.copy(val, backup)
                item = torch.load(val, weights_only=True)
                item['tokens'] = item['tokens'][torch.randperm(len(item['tokens']), generator=torch.Generator().manual_seed(1))]
                torch.save(item, val)
            for split in ('test', 'train'):
                run(['bash', f'scripts/evaluate_{split}.sh'], LOGS / f'evaluate_{split}_{case}.log')
                shutil.copy(ROOT / f'scoring/scores_{split}.json', SCORES / f'{split}_{case}.json')
        finally:
            (DATA / 'test.pt').unlink(missing_ok=True)
            if backup.is_file():
                shutil.copy(backup, val)
                backup.unlink()
            shutil.rmtree(ATTEMPTS)


# -------------------------------------------------------------------- summary
def cohort(split):
    """Every case's evaluator output (attempt view), plus the global attacks."""
    data = json.loads((SCORES / f'{split}_all.json').read_text())
    attempts = {name: value for name, value in data['_attempts'].items() if not name.startswith('files:')}
    for case in GLOBAL_ATTACKS:
        path = SCORES / f'{split}_{case}.json'
        if path.is_file():
            attempts[case] = json.loads(path.read_text())['_attempts'][case]
    return data, attempts


def reference_view(data):
    return {'experiments': {e: {'results': {r: data['experiments'][e]['results'][r]}}
                            for e, r in (('codec_stack2', 'generator_2x'), ('codec_stack4', 'generator_4x'))},
            '_failures': {}}


def block_stats(tokens, speakers, ev, blocks):
    """Per block of rows: per-speaker/global pair-feature sums, products and
    counts, and token-pair counts (the evaluator's sufficient statistics)."""
    out = []
    for idx in torch.arange(len(tokens)).chunk(blocks):
        means_sum = torch.zeros(S + 1, ev.PAIR, dtype=torch.float64)
        prod = torch.zeros(S + 1, ev.PAIR, ev.PAIR, dtype=torch.float64)
        cnt = torch.zeros(S + 1, dtype=torch.float64)
        z = ev.codec_latent(tokens[idx])
        pairs = torch.cat([z[:, :-1], z[:, 1:]], -1)
        spk = speakers[idx]
        for s in range(S + 1):
            x = (pairs if s == S else pairs[spk == s]).reshape(-1, ev.PAIR)
            cnt[s] += len(x)
            means_sum[s] += x.sum(0)
            prod[s] += x.T @ x
        out.append((means_sum, prod, cnt, ev.pair_counts(tokens[idx]).double()))
    return [torch.stack(parts) for parts in zip(*out)]


def metrics_from(stats, weights, ev, ref):
    sums, prods, cnts, pairs = [(x * weights.view(-1, *([1] * (x.dim() - 1)))).sum(0) for x in stats]
    means = sums / cnts[:, None]
    cov = (prods - sums[:, :, None] * sums[:, None, :] / cnts[:, None, None]) / (cnts - 1)[:, None, None]
    rm, rc = ref['moments']
    d = [ev.frechet_moments(means[i], cov[i], rm[i], rc[i]) for i in range(S + 1)]
    js = ev.pair_divergences(pairs, ref['pairs'])
    return {'codec_fd': d[-1], 'speaker_conditioned_codec_fd': sum(d[:-1]) / S, 'codebook_pair_divergence': float(js.mean())}


def paired_bootstrap(sigmas, spec, names, replicates=200, blocks=64):
    """Paired block bootstrap against the reference: identical held-out rows and
    identical hidden references for both methods, cohort standard deviations
    held fixed. Reports the 95% interval of the normalized improvement (for
    each cohort in `sigmas`) and of every raw metric difference."""
    from eval.test import evaluate as ev
    ctx_ref = ev.build_context()
    rows, speakers = ctx_ref['rows'], ctx_ref['speakers']
    g = torch.Generator().manual_seed(2026)
    draws = [torch.bincount(torch.randint(blocks, (blocks,), generator=g), minlength=blocks).double()
             for _ in range(replicates)]

    def per_draw(name):
        out = []
        for s in STACKS:
            tokens = load_pred(CASES / name / 'predictions' / f'generator_{s}x.pt')[rows]
            stats = block_stats(tokens, speakers, ev, blocks)
            out.append([metrics_from(stats, w, ev, ctx_ref['reference']) for w in draws])
        return out

    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * replicates)], values[int(0.975 * replicates) - 1]]

    ref = per_draw('reference')
    out = {}
    for name in names:
        new = per_draw(name)
        entry = {'raw_delta_95_percent': {}}
        for i, s in enumerate(STACKS):
            for metric in METRICS:
                deltas = [new[i][r][metric] - ref[i][r][metric] for r in range(replicates)]
                entry['raw_delta_95_percent'][f'generator_{s}x/{metric}'] = interval(deltas)
        for cohort_name, sigma in sigmas.items():
            totals = []
            for r in range(replicates):
                total = 0.0
                for i, s in enumerate(STACKS):
                    exp = EXPS[s]
                    cfg = spec['experiments'][exp]
                    for metric in METRICS:
                        key = f'{exp}/generator_{s}x/{metric}'
                        coef = cfg['weight'] * abs(cfg['metrics'][metric]['coefficient'])
                        direction = 1 if cfg['metrics'][metric]['higher_is_better'] else -1
                        if sigma[key]:
                            total += coef * direction * (new[i][r][metric] - ref[i][r][metric]) / sigma[key]
                totals.append(total)
            entry[f'{cohort_name}_mean'] = statistics.fmean(totals)
            entry[f'{cohort_name}_95_percent'] = interval(totals)
        out[name] = entry
        log(f'bootstrap {name}')
    return out


def summary_phase():
    sys.path.insert(0, str(ROOT))
    from eval.test.ranking import rank_attempts
    spec = json.loads((ROOT / 'scoring/reference.json').read_text())
    table = {}
    sigmas = {}
    for split in ('test', 'train'):
        data, attempts = cohort(split)
        ranking = rank_attempts(attempts, reference_view(data), spec)
        sigmas[split] = ranking['standard_deviations']
        for name, result in ranking['attempts'].items():
            entry = table.setdefault(name, {})
            entry[split] = round(result['improvement'], 6)
            entry[f'{split}_feasible'] = result['feasible']
            att = attempts[name]
            entry[f'{split}_failure'] = att['_failures'].get('generator_4x') or att['_failures'].get('generator_2x')
            entry[f'{split}_raw'] = {row: {k: round(v, 6) for k, v in att['_raw'].get(row, {}).items()}
                                     for row in ('generator_2x', 'generator_4x')}
        table.setdefault('_comparison_rows', {})[split] = {k: {m: round(v, 6) for m, v in r.items()} for k, r in data['_raw'].items()
                                                         if k.startswith('baseline_')}
        table.setdefault('_failed_run_values', {})[split] = data['_metadata']['failed_run_values']
    # Second view: the cohort of feasible (honest-looking) attempts only, so
    # failed runs at random-token values do not dominate the standard deviations.
    data, attempts = cohort('test')
    feasible = {n: a for n, a in attempts.items() if table.get(n, {}).get('test_feasible')}
    ranking = rank_attempts(feasible, reference_view(data), spec)
    sigmas['test_feasible_cohort'] = ranking['standard_deviations']
    for name, result in ranking['attempts'].items():
        table[name]['test_feasible_cohort'] = round(result['improvement'], 6)
    groups = {}
    for group, members in SEED_GROUPS.items():
        present = [m for m in members if m in table]
        if present:
            for key in ('test', 'test_feasible_cohort'):
                vals = [table[m].get(key, float('nan')) for m in present]
                groups[f'{group}/{key}'] = {'members': dict(zip(present, vals)), 'mean': statistics.fmean(vals),
                                            'min': min(vals), 'max': max(vals), 'range': max(vals) - min(vals)}
    noise = {}
    for key in ('test', 'test_feasible_cohort'):
        gen = [table[m].get(key) for m in ('reference', 'reference_sample1', 'reference_sample2') if m in table]
        noise[key] = {'training_seed_range': groups.get(f'reference/{key}', {}).get('range'),
                      'generation_seed_range': (max(gen) - min(gen)) if gen else None}
    others = sorted(n for n in table if not n.startswith('_') and n != 'reference' and table[n]['test_feasible'])
    boot = paired_bootstrap({'test': sigmas['test'], 'test_feasible_cohort': sigmas['test_feasible_cohort']},
                            spec, others) if others else {}
    # The finite-sample bound: a fresh sample of the hidden process itself,
    # measured like any row (not an attempt; diagnostic only).
    from eval.test import evaluate as ev
    ctx = ev.build_context()
    floor = {}
    for seed in (5_000_001, 5_000_002):
        _, raw, _ = ev.score_tokens(ev.sample_tokens(ctx['speakers'], seed), ctx)
        floor[str(seed)] = {k: round(v, 6) for k, v in raw.items()}
    summary = {'cases': {k: v for k, v in table.items() if not k.startswith('_')}, 'comparison_rows': table['_comparison_rows'],
               'perfect_sampler_bound_test': floor,
               'failed_run_values': table['_failed_run_values'], 'seed_groups': groups, 'noise': noise,
               'paired_bootstrap_test': boot, 'standard_deviations': sigmas}
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    lines = [f"{'case':38s} {'test':>9s} {'train':>9s} {'feas.coh':>9s}  outcome"]
    for name in sorted(summary['cases'], key=lambda n: (-summary['cases'][n]['test_feasible'], -summary['cases'][n]['test'])):
        c = summary['cases'][name]
        outcome = 'feasible' if c['test_feasible'] else f"FAILED: {c['test_failure']}"
        fc = f"{c['test_feasible_cohort']:+9.4f}" if 'test_feasible_cohort' in c else ' ' * 9
        lines.append(f"{name:38s} {c['test']:+9.4f} {c['train']:+9.4f} {fc}  {outcome[:100]}")
    lines.append('')
    for name in sorted(summary['cases']):
        for row in ('generator_2x', 'generator_4x'):
            r = summary['cases'][name]['test_raw'].get(row, {})
            if r:
                lines.append(f"{name:38s} {row}  fd={r['codec_fd']:.4f} spk_fd={r['speaker_conditioned_codec_fd']:.4f} "
                             f"pair_js={r['codebook_pair_divergence']:.5f} excess={r['visible_window_excess']:.4f} "
                             f"half={r['copied_fraction']:.3f}")
    lines.append('')
    for name, r in sorted(table['_comparison_rows']['test'].items()):
        lines.append(f"{name:38s} fd={r['codec_fd']:.4f} spk_fd={r['speaker_conditioned_codec_fd']:.4f} pair_js={r['codebook_pair_divergence']:.5f}")
    lines.append('')
    for seed, r in floor.items():
        lines.append(f"{'perfect_sampler_bound_' + seed:38s} fd={r['codec_fd']:.4f} spk_fd={r['speaker_conditioned_codec_fd']:.4f} "
                     f"pair_js={r['codebook_pair_divergence']:.5f} excess={r['visible_window_excess']:.4f}")
    lines.append('')
    lines.append(json.dumps({'seed_groups': groups, 'noise': noise, 'paired_bootstrap_test': boot}, indent=1))
    (HERE / 'summary.txt').write_text('\n'.join(lines) + '\n')
    log('summary written')


# -------------------------------------------------------------------- restore
def restore_phase():
    restore_reference_predictions()
    if ATTEMPTS.exists():
        shutil.rmtree(ATTEMPTS)
    run(['bash', 'scripts/evaluate_train.sh'], LOGS / 'reference_train.log')
    shutil.copy(ROOT / 'scoring/scores_train.json', ROOT / 'scoring/scores.json')
    run(['bash', 'scripts/evaluate_test.sh'], LOGS / 'reference_test.log')
    scores = json.loads((ROOT / 'scoring/scores_test.json').read_text())
    spec = json.loads((ROOT / 'scoring/reference.json').read_text())
    for exp, cfg in spec['experiments'].items():
        for row, result in cfg['results'].items():
            for metric in METRICS:
                result[metric] = scores['experiments'][exp]['results'][row][metric]
    (ROOT / 'scoring/reference.json').write_text(json.dumps(spec, indent=2) + '\n')
    # Rescore so the ranking block in the score files uses the final reference.json.
    run(['bash', 'scripts/evaluate_test.sh'], LOGS / 'reference_test.log')
    run(['bash', 'scripts/evaluate_train.sh'], LOGS / 'reference_train.log')
    shutil.copy(ROOT / 'scoring/scores_train.json', ROOT / 'scoring/scores.json')
    run([sys.executable, 'validate.py', '--post-design'], LOGS / 'validation.log')
    log('reference-only scores restored; validate.py passed')


if __name__ == '__main__':
    torch.set_num_threads(1)
    phases = sys.argv[1:] or ['reference', 'variants', 'shortcuts', 'score', 'summary', 'restore']
    for phase in phases:
        log(f'phase {phase}')
        globals()[f'{phase}_phase']()
