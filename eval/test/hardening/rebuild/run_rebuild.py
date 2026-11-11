"""Rebuild check: one bundled, resumable job for the hidden-process codec gym.

The data process changed in the previous iteration (eval/test/codec_process.py),
and the first job (4a264730-2a8) trained the reference, the comparison models
and every variant for the two main operating points from that code. Finished
models, predictions and cases are skipped on a rerun, so the closing job adds
the few-shot-speaker rows (row key 'fewshot') to every case, the few-shot
reference and comparison models, the private-search (moment-matching
selection) shortcut, a low-rank-speaker variant and a few-shot memorizer, and
rescores everything with all three experiments. The two-experiment scores of
the first job are kept in scores_two_experiments/ and summary_two_experiments.*.

Phases:
  reference  The end-to-end reference recipe (scripts/reproduce.sh arguments):
             AR local transformer at 2x/4x, parallel heads at 1x/2x/4x,
             MaskGIT local transformer at 2x/4x, and the AR reference, parallel
             and MaskGIT at 4x on the few-shot data, 10000 steps each; then
             `python -m eval.test.prepare_fixtures`.
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
             repair; operating-point reuse; integrity attacks; private search
             (per row, one of three samples chosen so that the set's codec
             moments match the public training data); a model that memorizes
             the few-shot speakers' utterances.
  score      Every case is staged under checkpoints/attempts/ and scored once
             by the no-argument scripts/evaluate_test.sh and
             scripts/evaluate_train.sh (attacks on global files separately).
  summary    Cohort-normalized scores, raw measurements, seed groups, a paired
             block bootstrap on identical held-out rows, and the
             finite-sample bound of a perfect sampler.
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
DATA = {'main': ROOT / 'data' / 'synthetic_codec', 'fewshot': ROOT / 'data' / 'synthetic_codec_fewshot'}
N, T, K, V = 16384, 24, 8, 32
# Row key -> (file stem, task, stack factor, experiment)
ROWS = {2: ('generator_2x', 'main', 2, 'codec_stack2'),
        4: ('generator_4x', 'main', 4, 'codec_stack4'),
        'fewshot': ('generator_fewshot_4x', 'fewshot', 4, 'codec_fewshot_stack4')}
METRICS = ('codec_fd', 'speaker_conditioned_codec_fd', 'codebook_pair_divergence')
REFERENCE_ROWS = {'generator_2x': ('ar', 2, 2, 'main'), 'generator_4x': ('ar', 4, 2, 'main'),
                  'baseline_maskgit_2x': ('maskgit', 2, 2, 'main'), 'baseline_maskgit_4x': ('maskgit', 4, 2, 'main'),
                  'baseline_parallel_1x': ('none', 1, 3, 'main'), 'baseline_parallel_2x': ('none', 2, 3, 'main'),
                  'baseline_parallel_4x': ('none', 4, 3, 'main'),
                  'generator_fewshot_4x': ('ar', 4, 2, 'fewshot'),
                  'baseline_maskgit_fewshot_4x': ('maskgit', 4, 2, 'fewshot'),
                  'baseline_parallel_fewshot_4x': ('none', 4, 3, 'fewshot')}
PARALLEL = {2: 'baseline_parallel_2x', 4: 'baseline_parallel_4x', 'fewshot': 'baseline_parallel_fewshot_4x'}
LARGE = ['--dim', '192', '--main-layers', '3', '--lt-layers', '2']
COSINE20K = ['--steps', '20000', '--lr-schedule', 'cosine']
# name -> (training arguments, training seed); every variant trains every row.
TRAINED = {
    'reference_seed17': ([], 17),
    'reference_seed27': ([], 27),
    'better_cosine_20k': (COSINE20K, 7),
    'better_large_cosine_20k': (LARGE + COSINE20K, 7),
    'better_large_cosine_20k_seed17': (LARGE + COSINE20K, 17),
    'better_large_cosine_20k_seed27': (LARGE + COSINE20K, 27),
    'weaker_training_2k': (['--steps', '2000'], 7),
    # Few-shot-targeted idea on top of the 20k cosine recipe: a rank-16
    # speaker table shared by all speakers.
    'better_speaker_rank16_cosine_20k': (['--speaker-rank', '16'] + COSINE20K, 7),
}
# Extra generation seeds of a trained case, used as private-search candidates.
SAMPLED = {'better_large_cosine_20k': (1, 2)}
SEED_GROUPS = {
    'reference': ['reference', 'reference_seed17', 'reference_seed27'],
    'better_large_cosine_20k': ['better_large_cosine_20k', 'better_large_cosine_20k_seed17', 'better_large_cosine_20k_seed27'],
}
GLOBAL_ATTACKS = ('shortcut_replace_truth_testfile', 'shortcut_replace_truth_valfile')
DRY_STEPS = os.environ.get('REBUILD_DRY_STEPS')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run(cmd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, 'w') as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)


def train(ckpt_dir, variant, stack, main_layers, extra=(), seed=7, task='main'):
    if (ckpt_dir / 'model.pt').is_file():
        return
    args = ['--data', str(DATA[task]), '--out', str(ckpt_dir), '--variant', variant, '--stack-factor', str(stack),
            '--steps', '10000', '--main-layers', str(main_layers), '--seed', str(seed), *extra]
    if DRY_STEPS:  # pipeline dry run only (never in a real job)
        args += ['--steps', DRY_STEPS]
    t0 = time.time()
    run([sys.executable, '-m', 'method.train', *args], LOGS / 'train' / f'{ckpt_dir.parent.name}_{ckpt_dir.name}.log')
    log(f'trained {ckpt_dir} in {time.time() - t0:.0f}s')


def predict(ckpt, out, temperature=1.0, seed=0, task='main'):
    if out.is_file():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + '.partial.pt')
    if DRY_STEPS:  # pipeline dry run only: stand-in samples instead of slow CPU generation
        from eval.test import evaluate as ev
        tokens = ev.sample_tokens(task, conditions(task), abs(hash((str(ckpt), temperature, seed))) % 100_000)
        torch.save({'pred_tokens': tokens, 'config': {'stack_factor': int(out.stem[-2])}, 'split': 'test'}, out)
        return
    run([sys.executable, '-m', 'method.predict', '--data', str(DATA[task]), '--checkpoint', str(ckpt), '--out', str(tmp),
         '--temperature', str(temperature), '--seed', str(seed)],
        LOGS / 'predict' / f'{out.parent.parent.name}_{out.stem}.log')
    tmp.rename(out)


def row_path(case, key):
    return CASES / case / 'predictions' / f'{ROWS[key][0]}.pt'


def have(case, key):
    return row_path(case, key).is_file()


def put(case, key, tokens, config=None):
    target = row_path(case, key)
    target.parent.mkdir(parents=True, exist_ok=True)
    stack = ROWS[key][2]
    cfg = {'stack_factor': stack} if config is None else config(stack)
    torch.save({'pred_tokens': tokens, 'config': cfg, 'split': 'test'}, target.with_suffix('.tmp'))
    target.with_suffix('.tmp').rename(target)


def complete(case):
    return all(have(case, key) for key in ROWS)


def load_pred(path):
    return torch.load(path, map_location='cpu', weights_only=False)['pred_tokens'].long()


def reference_row(key):
    return load_pred(PRED / f'{ROWS[key][0]}.pt')


# ------------------------------------------------------------------ reference
def reference_phase():
    for name, (variant, stack, layers, task) in REFERENCE_ROWS.items():
        train(ROOT / 'checkpoints' / name, variant, stack, layers, task=task)
        predict(ROOT / 'checkpoints' / name / 'model.pt', PRED / f'{name}.pt', task=task)
    log('reference and comparison predictions ready')
    stamp = WORK / 'fixtures_done_with_fewshot'
    if not stamp.is_file():
        import hashlib
        fixed = sorted(p for split in ('train', 'test') for p in (ROOT / 'eval' / split / 'fixed').rglob('*') if p.is_file())
        before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in fixed}
        run([sys.executable, '-m', 'eval.test.prepare_fixtures'], LOGS / 'prepare_fixtures.log')
        after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for split in ('train', 'test') for p in sorted((ROOT / 'eval' / split / 'fixed').rglob('*')) if p.is_file()}
        check = {'unchanged': sorted(k for k in before if after.get(k) == before[k]),
                 'changed': sorted(k for k in before if k in after and after[k] != before[k]),
                 'new': sorted(k for k in after if k not in before)}
        (HERE / 'fixtures_check.json').write_text(json.dumps(check, indent=1) + '\n')
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text('ok\n')
        log(f"fixtures frozen from the reference outputs; changed: {check['changed']}")


# ------------------------------------------------------------------ variants
def variants_phase():
    for key in ROWS:
        if not have('reference', key):
            put('reference', key, reference_row(key).to(torch.uint8))
    for case, temperature, seed in (('reference_sample1', 1.0, 1), ('reference_sample2', 1.0, 2),
                                    ('paper_sampling_t06', 0.6, 0), ('weaker_argmax', 0.0, 0)):
        for key, (stem, task, _, _) in ROWS.items():
            predict(ROOT / 'checkpoints' / stem / 'model.pt', row_path(case, key), temperature, seed, task)
        log(f'built {case}')
    for key, name in PARALLEL.items():
        if not have('weaker_no_local', key):
            put('weaker_no_local', key, load_pred(PRED / f'{name}.pt').to(torch.uint8))
    # The 2x model's own fresh sample declared as the 4x row (false compute claim).
    case = 'shortcut_resample_2x_as_4x'
    if not have(case, 2):
        put(case, 2, reference_row(2).to(torch.uint8))
    if not have(case, 4):
        predict(ROOT / 'checkpoints' / 'generator_2x' / 'model.pt', WORK / 'resample_2x.pt', 1.0, 5)
        put(case, 4, load_pred(WORK / 'resample_2x.pt').to(torch.uint8))
    if not have(case, 'fewshot'):
        put(case, 'fewshot', reference_row('fewshot').to(torch.uint8))
    for case, (extra, seed) in TRAINED.items():
        if complete(case):
            continue
        for key, (stem, task, stack, _) in ROWS.items():
            ckpt = model_dir(case, key)
            train(ckpt, 'ar', stack, 2, extra, seed, task)
            predict(ckpt / 'model.pt', row_path(case, key), task=task)
        log(f'built {case}')
    for case, seeds in SAMPLED.items():
        for seed in seeds:
            for key, (stem, task, _, _) in ROWS.items():
                predict(model_dir(case, key) / 'model.pt', sample_path(case, seed, key), 1.0, seed, task)
        log(f'sampled {case} seeds {seeds}')
    fewshot_memorizer()
    shortfall_predictions()


def model_dir(case, key):
    """Checkpoint of a case's row: the reference files, or a trained variant."""
    stem, task, stack, _ = ROWS[key]
    if case == 'reference':
        return ROOT / 'checkpoints' / stem
    return WORK / 'models' / case / (f'{stack}x' if task == 'main' else 'fewshot_4x')


def sample_path(case, seed, key):
    if case == 'reference' and seed in (1, 2):
        return row_path(f'reference_sample{seed}', key)
    return WORK / 'samples' / case / f'seed{seed}' / f'{ROWS[key][0]}.pt'


def fewshot_memorizer():
    """Shortcut: an AR 4x model trained only on the 512 training utterances of
    the 64 few-shot speakers for 5000 steps (about 1250 epochs), which
    reproduces them; the main rows are the reference's."""
    case = 'shortcut_fewshot_memorize'
    if complete(case):
        return
    data = WORK / 'fewshot_only_data'
    if not (data / 'train.pt').is_file():
        data.mkdir(parents=True, exist_ok=True)
        tr = torch.load(DATA['fewshot'] / 'train.pt', weights_only=True)
        keep = tr['speaker'] >= 192
        torch.save({'tokens': tr['tokens'][keep], 'speaker': tr['speaker'][keep]}, data / 'train.pt')
        for name in ('val.pt', 'meta.pt', 'test_conditions.pt'):
            shutil.copy(DATA['fewshot'] / name, data / name)
    ckpt = WORK / 'models' / case / 'fewshot_4x'
    if not (ckpt / 'model.pt').is_file():
        args = ['--data', str(data), '--out', str(ckpt), '--variant', 'ar', '--stack-factor', '4',
                '--steps', DRY_STEPS or '5000', '--batch-size', '64', '--seed', '7']
        t0 = time.time()
        run([sys.executable, '-m', 'method.train', *args], LOGS / 'train' / f'{case}_fewshot_4x.log')
        log(f'trained {ckpt} in {time.time() - t0:.0f}s')
    predict(ckpt / 'model.pt', row_path(case, 'fewshot'), task='fewshot')
    for key in (2, 4):
        if not have(case, key):
            put(case, key, reference_row(key).to(torch.uint8))
    log(f'built {case}')


RICH = list(range(160, 192))      # 32 well-covered speakers, 256 rows each (as a few-shot slice)
SHORTFALL_CASES = ('reference', 'better_cosine_20k', 'better_large_cosine_20k', 'better_speaker_rank16_cosine_20k')


def shortfall_predictions():
    """Diagnostic for the few-shot experiment: the same few-shot-task models
    generating for 32 well-covered speakers (160 training utterances each)."""
    data = WORK / 'rich_conditions'
    data.mkdir(parents=True, exist_ok=True)
    torch.save({'speaker': torch.tensor(RICH).repeat(256)}, data / 'test_conditions.pt')
    for case in SHORTFALL_CASES:
        out = WORK / 'shortfall' / f'{case}.pt'
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + '.partial.pt')
            if DRY_STEPS:
                from eval.test import evaluate as ev
                torch.save({'pred_tokens': ev.sample_tokens('fewshot', torch.tensor(RICH).repeat(256), 77)}, out)
                continue
            run([sys.executable, '-m', 'method.predict', '--data', str(data), '--checkpoint',
                 str(model_dir(case, 'fewshot') / 'model.pt'), '--out', str(tmp)],
                LOGS / 'predict' / f'shortfall_{case}.log')
            tmp.rename(out)
    log('shortfall predictions ready')


# ------------------------------------------------------------------ shortcuts
def conditions(task='main'):
    return torch.load(DATA[task] / 'test_conditions.pt', weights_only=True)['speaker'].long()


def training_set(task='main'):
    tr = torch.load(DATA[task] / 'train.pt', weights_only=True)
    return tr['tokens'].long(), tr['speaker'].long()


def copy_rows(seed, task='main'):
    """For every conditioning row, a random training utterance of the same speaker."""
    toks, spk_tr = training_set(task)
    spk = conditions(task)
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(N, T, K, dtype=torch.long)
    for s in spk.unique().tolist():
        rows = (spk == s).nonzero().squeeze(1)
        pool = toks[spk_tr == s]
        out[rows] = pool[torch.randint(len(pool), (len(rows),), generator=g)]
    return out


def edit(x, rate, seed):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(x.shape, generator=g) < rate
    return torch.where(mask, torch.randint(V, x.shape, generator=g), x)


def splice(piece, seed, task='main'):
    out = torch.zeros(N, T, K, dtype=torch.long)
    for i, start in enumerate(range(0, T, piece)):
        out[:, start:start + piece] = copy_rows(seed + i, task)[:, start:start + piece]
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


def gaussian_latent(seed, task='main'):
    """Per speaker, a Gaussian AR(1) fitted to the training latents' first two
    moments (exactly what FD compares), sampled and re-quantized."""
    toks, spk_tr = training_set(task)
    z = latent(toks)
    spk = conditions(task)
    g = torch.Generator().manual_seed(seed)
    d = 3 * K
    out = torch.zeros(N, T, K, dtype=torch.long)
    eye = (1e-9 if task == 'main' else 1e-4) * torch.eye(d, dtype=torch.float64)
    for s in spk.unique().tolist():
        x = z[spk_tr == s]
        a, b = x[:, :-1].reshape(-1, d), x[:, 1:].reshape(-1, d)
        mu = x.reshape(-1, d).mean(0)
        c = torch.cov(torch.cat([a, b], 1).T)
        A = c[:d, d:].T @ torch.linalg.inv(c[:d, :d] + eye)
        noise = torch.linalg.cholesky(c[d:, d:] - A @ c[:d, d:] + eye)
        first = torch.linalg.cholesky(torch.cov(x[:, 0].T) + eye) if len(x) > d else torch.linalg.cholesky(c[:d, :d] + eye)
        rows = (spk == s).nonzero().squeeze(1)
        cur = x[:, 0].mean(0) + torch.randn(len(rows), d, generator=g, dtype=torch.float64) @ first.T
        seq = [cur]
        for _ in range(1, T):
            cur = mu + (cur - mu) @ A.T + torch.randn(len(rows), d, generator=g, dtype=torch.float64) @ noise.T
            seq.append(cur)
        out[rows] = requantize(torch.stack(seq, 1))
    return out


def count_chain(seed, task='main', alpha=0.1):
    """Hand-written structure fitted by counting (the previous gym's analytic
    sampler, generalized): per speaker, codebook 0 is a first-order chain over
    frames and codebook k depends on codebook k-1 of the same frame and on
    codebook k of the previous frame."""
    toks, spk_tr = training_set(task)
    spk = conditions(task)
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(N, T, K, dtype=torch.long)
    for s in spk.unique().tolist():
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


def frame_lookup(seed, task='main'):
    """Nonparametric frame bigram: each next frame is a training frame whose
    predecessor matched the current frame on codebooks 0 and 1 (same speaker)."""
    toks, spk_tr = training_set(task)
    spk = conditions(task)
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(N, T, K, dtype=torch.long)
    for s in spk.unique().tolist():
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


def old_analytic_sampler(seed, task='main'):
    """The previous gym's hand-written sampler (affine residual rule, coarse walk)."""
    spk = conditions(task)
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


def old_rule_repair(tokens, task='main'):
    """The previous gym's rule repair: snap outputs onto its public validity rule."""
    spk = conditions(task)
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


def splice_partial(key, seed):
    """The reference outputs with 12% of rows replaced by 4-frame training
    splices, sized to stay under the window threshold."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.rand(N, generator=g) < 0.12
    mixed = reference_row(key).clone()
    mixed[rows] = splice(4, 300 + seed, ROWS[key][1])[rows]
    return mixed


def select_moments(source, key, sweeps=2):
    """Private search with public data only: for every row, choose one of three
    generation samples of `source` (seeds 0, 1, 2) so that the chosen set's
    consecutive-frame codec-latent moments match those of the task's public
    training and validation utterances, globally and per conditioning speaker
    (greedy coordinate descent on 1.0 x global FD + 0.6 x mean speaker FD, the
    scored distances, computed with the public FSQ latent)."""
    from eval.test import evaluate as ev
    stem, task, _, _ = ROWS[key]
    cands = torch.stack([load_pred(row_path(source, key) if seed == 0 else sample_path(source, seed, key))
                         for seed in (0, 1, 2)])
    spk = conditions(task)
    data = [torch.load(DATA[task] / f'{s}.pt', weights_only=True) for s in ('train', 'val')]
    tok = torch.cat([d['tokens'].long() for d in data])
    tspk = torch.cat([d['speaker'].long() for d in data])
    groups = sorted(set(spk.tolist()))
    keep = torch.isin(tspk, torch.tensor(groups))
    tm, tc = ev.feature_moments(tok[keep], tspk[keep], groups)
    n_groups = len(groups)
    gidx = torch.tensor([groups.index(s) for s in spk.tolist()])

    def feats(t):
        z = ev.codec_latent(t)
        return torch.cat([z[:, :-1], z[:, 1:]], -1)

    s1 = torch.stack([feats(c).sum(1) for c in cands])                       # C, N, 48
    s2 = torch.stack([torch.einsum('nti,ntj->nij', feats(c), feats(c)) for c in cands])
    n = torch.zeros(n_groups + 1, dtype=torch.float64)
    gs = torch.zeros(n_groups + 1, ev.PAIR, dtype=torch.float64)
    gp = torch.zeros(n_groups + 1, ev.PAIR, ev.PAIR, dtype=torch.float64)
    gs.index_add_(0, gidx, s1[0]); gp.index_add_(0, gidx, s2[0]); n.index_add_(0, gidx, torch.full((len(spk),), float(T - 1), dtype=torch.float64))
    gs[-1], gp[-1], n[-1] = gs[:-1].sum(0), gp[:-1].sum(0), n[:-1].sum()
    choice = torch.zeros(len(spk), dtype=torch.long)

    def fd(s, p, cnt, g):
        mean = s / cnt
        cov = (p - torch.outer(s, s) / cnt) / (cnt - 1)
        return ev.frechet_moments(mean, cov, tm[g], tc[g])

    g = torch.Generator().manual_seed(11)
    for _ in range(sweeps):
        for i in torch.randperm(len(spk), generator=g).tolist():
            gi, cur = int(gidx[i]), int(choice[i])
            best, best_value = cur, None
            for c in range(len(cands)):
                ds, dp = s1[c, i] - s1[cur, i], s2[c, i] - s2[cur, i]
                value = fd(gs[-1] + ds, gp[-1] + dp, n[-1], n_groups) + 0.6 * fd(gs[gi] + ds, gp[gi] + dp, n[gi], gi) / n_groups
                if best_value is None or value < best_value - 1e-12:
                    best, best_value = c, value
            if best != cur:
                ds, dp = s1[best, i] - s1[cur, i], s2[best, i] - s2[cur, i]
                gs[gi] += ds; gp[gi] += dp; gs[-1] += ds; gp[-1] += dp
                choice[i] = best
    return cands[choice, torch.arange(len(spk))]


SEEDS = {2: 1, 4: 2, 'fewshot': 3}


def shortcut_builders():
    """case -> function(row key) -> tokens, each built from public data only."""
    task = lambda key: ROWS[key][1]
    return {
        'shortcut_copy_train': lambda k: copy_rows(SEEDS[k], task(k)),
        'shortcut_copy_edit_3pct': lambda k: edit(copy_rows(SEEDS[k], task(k)), 0.03, 2 + SEEDS[k]),
        'shortcut_copy_edit_12pct': lambda k: edit(copy_rows(SEEDS[k], task(k)), 0.125, 2 + SEEDS[k]),
        'shortcut_copy_edit_25pct': lambda k: edit(copy_rows(SEEDS[k], task(k)), 0.25, 2 + SEEDS[k]),
        'shortcut_splice_2frame': lambda k: splice(2, 100 * SEEDS[k], task(k)),
        'shortcut_splice_3frame': lambda k: splice(3, 100 * SEEDS[k], task(k)),
        'shortcut_splice_4frame': lambda k: splice(4, 100 * SEEDS[k], task(k)),
        'shortcut_splice_6frame': lambda k: splice(6, 100 * SEEDS[k], task(k)),
        'shortcut_splice_partial': lambda k: splice_partial(k, 9 + SEEDS[k]),
        # The same two sub-threshold splices on the main rows only, with the
        # reference's honest few-shot row (the few-shot copy checks would
        # otherwise fail the whole run).
        'shortcut_splice_partial_mainonly': lambda k: (reference_row(k) if k == 'fewshot'
                                                       else splice_partial(k, 9 + SEEDS[k])),
        'shortcut_splice_2frame_mainonly': lambda k: (reference_row(k) if k == 'fewshot'
                                                      else splice(2, 100 * SEEDS[k], task(k))),
        'shortcut_random_tokens': lambda k: torch.randint(V, (N, T, K), generator=torch.Generator().manual_seed(ROWS[k][2] + (10 if k == 'fewshot' else 0))),
        'shortcut_constant': lambda k: torch.zeros(N, T, K, dtype=torch.long),
        'shortcut_gaussian_latent': lambda k: gaussian_latent(SEEDS[k], task(k)),
        'shortcut_count_chain': lambda k: count_chain(SEEDS[k], task(k)),
        'shortcut_frame_lookup': lambda k: frame_lookup(SEEDS[k], task(k)),
        'shortcut_old_analytic_sampler': lambda k: old_analytic_sampler(SEEDS[k], task(k)),
        'shortcut_old_rule_repair': lambda k: old_rule_repair(reference_row(k), task(k)),
        'shortcut_reuse_2x_as_4x': lambda k: reference_row(2 if k == 4 else k),
    }


def shortcuts_phase():
    for case, build in shortcut_builders().items():
        for key in ROWS:
            if not have(case, key):
                put(case, key, build(key).to(torch.uint8))
        log(f'built {case}')
    # Private search: per-row choice among three samples of a trained model.
    for case, source in (('shortcut_select_moments', 'reference'),
                         ('shortcut_select_moments_large', 'better_large_cosine_20k')):
        for key in ROWS:
            if not have(case, key):
                t0 = time.time()
                put(case, key, select_moments(source, key).to(torch.uint8))
                log(f'built {case} row {ROWS[key][0]} in {time.time() - t0:.0f}s')
    # Private search on the main rows only, with the reference's few-shot row.
    case = 'shortcut_select_moments_mainonly'
    for key in ROWS:
        if not have(case, key):
            put(case, key, (reference_row(key) if key == 'fewshot' else load_pred(row_path('shortcut_select_moments', key))).to(torch.uint8))
    # Artifact and declaration attacks on the unchanged reference tokens.
    artifact = {
        'shortcut_stack_claim': lambda s: {'stack_factor': 24},
        'shortcut_negative_steps': lambda s: {'stack_factor': -1},
        'shortcut_zero_stack': lambda s: {'stack_factor': 0},
        'shortcut_infinite_stack': lambda s: {'stack_factor': float('inf')},
        'shortcut_vocab_one': lambda s: {'stack_factor': s, 'vocab_size': 1},
    }
    for case, config in artifact.items():
        for key in ROWS:
            if not have(case, key):
                put(case, key, reference_row(key).to(torch.uint8), config)
    for key in ROWS:
        if not have('shortcut_out_of_range', key):
            put('shortcut_out_of_range', key, torch.full((N, T, K), -1, dtype=torch.long))
        if not have('shortcut_empty', key):
            put('shortcut_empty', key, torch.zeros(0, T, K, dtype=torch.uint8))
        if not have('shortcut_write_scores', key):
            target = row_path('shortcut_write_scores', key)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'pred_tokens': reference_row(key).to(torch.uint8), 'config': {'stack_factor': ROWS[key][2]},
                        'split': 'test', 'metrics': {'codec_fd': 0.0, 'codebook_pair_divergence': 0.0}, 'success': True}, target)
        for case in ('shortcut_replace_baseline', *GLOBAL_ATTACKS):
            if not have(case, key):
                put(case, key, reference_row(key).to(torch.uint8))
    # Reference rows unchanged; the reserved parallel 1x file replaced by copies.
    baseline = CASES / 'shortcut_replace_baseline' / 'predictions' / 'baseline_parallel_1x.pt'
    if not baseline.is_file():
        torch.save({'pred_tokens': copy_rows(7).to(torch.uint8), 'config': {'stack_factor': 1}}, baseline)


# ---------------------------------------------------------------------- score
def case_names():
    return sorted(p.name for p in CASES.iterdir() if p.is_dir() and complete(p.name))


def restore_reference_predictions():
    for p in PRED.glob('*.pt'):
        if p.stem not in REFERENCE_ROWS:
            p.unlink()


def score_phase():
    target = SCORES / 'test_all.json'
    if target.is_file() and 'codec_fewshot_stack4' not in json.loads(target.read_text())['experiments']:
        # First job's two-experiment scores: keep them as the before state.
        old = HERE / 'scores_two_experiments'
        if old.exists():
            shutil.rmtree(old)
        SCORES.rename(old)
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
    val = DATA['main'] / 'val.pt'
    backup = WORK / 'val_backup.pt'
    for case in GLOBAL_ATTACKS:
        shutil.copytree(CASES / case / 'predictions', ATTEMPTS / case / 'predictions')
        try:
            if case.endswith('testfile'):
                torch.save({'tokens': reference_row(2).to(torch.uint8), 'speaker': conditions()}, DATA['main'] / 'test.pt')
            else:
                shutil.copy(val, backup)
                item = torch.load(val, weights_only=True)
                item['tokens'] = item['tokens'][torch.randperm(len(item['tokens']), generator=torch.Generator().manual_seed(1))]
                torch.save(item, val)
            for split in ('test', 'train'):
                run(['bash', f'scripts/evaluate_{split}.sh'], LOGS / f'evaluate_{split}_{case}.log')
                shutil.copy(ROOT / f'scoring/scores_{split}.json', SCORES / f'{split}_{case}.json')
        finally:
            (DATA['main'] / 'test.pt').unlink(missing_ok=True)
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
    return {'experiments': {exp: {'results': {stem: data['experiments'][exp]['results'][stem]}}
                            for stem, _, _, exp in ROWS.values()},
            '_failures': {}}


def block_stats(tokens, speakers, groups, ev, blocks):
    """Per block of rows: per-speaker/global pair-feature sums, products and
    counts, and token-pair counts (the evaluator's sufficient statistics)."""
    out = []
    n = len(groups) + 1
    for idx in torch.arange(len(tokens)).chunk(blocks):
        sums = torch.zeros(n, ev.PAIR, dtype=torch.float64)
        prod = torch.zeros(n, ev.PAIR, ev.PAIR, dtype=torch.float64)
        cnt = torch.zeros(n, dtype=torch.float64)
        z = ev.codec_latent(tokens[idx])
        pairs = torch.cat([z[:, :-1], z[:, 1:]], -1)
        spk = speakers[idx]
        for i in range(n):
            x = (pairs if i == n - 1 else pairs[spk == groups[i]]).reshape(-1, ev.PAIR)
            cnt[i] += len(x)
            sums[i] += x.sum(0)
            prod[i] += x.T @ x
        out.append((sums, prod, cnt, ev.pair_counts(tokens[idx]).double()))
    return [torch.stack(parts) for parts in zip(*out)]


def metrics_from(stats, weights, ev, ref):
    sums, prods, cnts, pairs = [(x * weights.view(-1, *([1] * (x.dim() - 1)))).sum(0) for x in stats]
    means = sums / cnts[:, None]
    cov = (prods - sums[:, :, None] * sums[:, None, :] / cnts[:, None, None]) / (cnts - 1)[:, None, None]
    rm, rc = ref['moments']
    n = len(ref['groups'])
    d = [ev.frechet_moments(means[i], cov[i], rm[i], rc[i]) for i in range(n + 1)]
    js = ev.pair_divergences(pairs, ref['pairs'])
    return {'codec_fd': d[-1], 'speaker_conditioned_codec_fd': sum(d[:-1]) / n, 'codebook_pair_divergence': float(js.mean())}


def paired_bootstrap(ctx, sigmas, spec, names, replicates=200, blocks=64):
    """Paired block bootstrap against the reference: identical held-out rows and
    identical hidden references for both methods, cohort standard deviations
    held fixed. Reports the 95% interval of the normalized improvement (for
    each cohort in `sigmas`) and of every raw metric difference. Blocks are
    drawn independently per task."""
    from eval.test import evaluate as ev
    g = torch.Generator().manual_seed(2026)
    draws = {task: [torch.bincount(torch.randint(blocks, (blocks,), generator=g), minlength=blocks).double()
                    for _ in range(replicates)] for task in ('main', 'fewshot')}

    def per_draw(name):
        out = {}
        for key, (stem, task, _, _) in ROWS.items():
            tctx = ctx[task]
            tokens = load_pred(row_path(name, key))[tctx['rows']]
            stats = block_stats(tokens, tctx['speakers'], tctx['reference']['groups'], ev, blocks)
            out[key] = [metrics_from(stats, w, ev, tctx['reference']) for w in draws[task]]
        return out

    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * replicates)], values[int(0.975 * replicates) - 1]]

    ref = per_draw('reference')
    out = {}
    for name in names:
        new = per_draw(name)
        entry = {'raw_delta_95_percent': {}}
        for key, (stem, _, _, _) in ROWS.items():
            for metric in METRICS:
                deltas = [new[key][r][metric] - ref[key][r][metric] for r in range(replicates)]
                entry['raw_delta_95_percent'][f'{stem}/{metric}'] = interval(deltas)
        for cohort_name, sigma in sigmas.items():
            totals = []
            for r in range(replicates):
                total = 0.0
                for key, (stem, _, _, exp) in ROWS.items():
                    cfg = spec['experiments'][exp]
                    for metric in METRICS:
                        skey = f'{exp}/{stem}/{metric}'
                        coef = cfg['weight'] * abs(cfg['metrics'][metric]['coefficient'])
                        direction = 1 if cfg['metrics'][metric]['higher_is_better'] else -1
                        if sigma[skey]:
                            total += coef * direction * (new[key][r][metric] - ref[key][r][metric]) / sigma[skey]
                totals.append(total)
            entry[f'{cohort_name}_mean'] = statistics.fmean(totals)
            entry[f'{cohort_name}_95_percent'] = interval(totals)
        out[name] = entry
        log(f'bootstrap {name}')
    return out


def normalized(raw_by_row, ref_raw, spec, sigma):
    """The rank's quantity for raw measurements of every row (diagnostic)."""
    total = 0.0
    for stem, _, _, exp in ROWS.values():
        cfg = spec['experiments'][exp]
        for metric in METRICS:
            key = f'{exp}/{stem}/{metric}'
            if sigma[key]:
                total += cfg['weight'] * abs(cfg['metrics'][metric]['coefficient']) * (ref_raw[stem][metric] - raw_by_row[stem][metric]) / sigma[key]
    return total


def summary_phase():
    from eval.test.ranking import rank_attempts
    from eval.test import evaluate as ev
    spec = json.loads((ROOT / 'scoring/reference.json').read_text())
    stems = [stem for stem, _, _, _ in ROWS.values()]
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
            by_exp = {}
            for cell, value in result['contributions'].items():
                by_exp[cell.split('/')[0]] = by_exp.get(cell.split('/')[0], 0.0) + value
            entry[f'{split}_by_experiment'] = {k: round(v, 6) for k, v in by_exp.items()}
            att = attempts[name]
            entry[f'{split}_failure'] = next((att['_failures'][s] for s in stems if s in att['_failures']), None)
            entry[f'{split}_raw'] = {row: {k: round(v, 6) for k, v in att['_raw'].get(row, {}).items()} for row in stems}
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
        by_exp = {}
        for cell, value in result['contributions'].items():
            by_exp[cell.split('/')[0]] = by_exp.get(cell.split('/')[0], 0.0) + value
        table[name]['test_feasible_cohort_by_experiment'] = {k: round(v, 6) for k, v in by_exp.items()}
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
    ctx = ev.build_context()
    others = sorted(n for n in table if not n.startswith('_') and n != 'reference' and table[n]['test_feasible'])
    boot = paired_bootstrap(ctx, {'test': sigmas['test'], 'test_feasible_cohort': sigmas['test_feasible_cohort']},
                            spec, others) if others else {}
    # The finite-sample bound: a fresh sample of each hidden process itself,
    # measured like any row (not an attempt; diagnostic only).
    floor = {}
    for seed in (5_000_001, 5_000_002):
        rows = {}
        for stem, task, _, _ in ROWS.values():
            if task not in rows:
                _, raw, _ = ev.score_tokens(ev.sample_tokens(task, ctx[task]['speakers'], seed), ctx[task])
                rows[task] = {k: round(v, 6) for k, v in raw.items()}
        floor[str(seed)] = rows
    ref_raw = table['reference']['test_raw']
    bound_scores = {}
    for cohort_name in ('test', 'test_feasible_cohort'):
        bound_scores[cohort_name] = statistics.fmean(
            normalized({stem: f[task] for stem, task, _, _ in ROWS.values()}, ref_raw, spec, sigmas[cohort_name])
            for f in floor.values())
    # Few-shot shortfall: the same few-shot-task models on 32 well-covered
    # speakers (same rows per speaker as a few-shot slice), against a fresh
    # reference sample of those speakers drawn here.
    shortfall = {}
    rich = torch.tensor(RICH).repeat(256)
    rich_ref_spk = rich.repeat_interleave(ev.REFERENCES_PER_ROW)
    rich_ref = ev.sample_tokens('fewshot', rich_ref_spk, 64_000_001)
    rich_reference = {'moments': ev.feature_moments(rich_ref, rich_ref_spk, RICH), 'pairs': ev.pair_counts(rich_ref), 'groups': RICH}
    rich_rows = {case: load_pred(WORK / 'shortfall' / f'{case}.pt') for case in SHORTFALL_CASES if (WORK / 'shortfall' / f'{case}.pt').is_file()}
    rich_rows['perfect_sampler'] = ev.sample_tokens('fewshot', rich, 64_000_002)
    for case, tokens in rich_rows.items():
        well = {k: round(v, 6) for k, v in ev.distribution_metrics(tokens, rich, rich_reference).items()}
        few = (table.get(case, {}).get('test_raw', {}).get('generator_fewshot_4x')
               or {k: v for k, v in floor[next(iter(floor))]['fewshot'].items()})
        shortfall[case] = {'well_covered_speakers_160_191': well,
                           'few_shot_test_speakers_224_255': {k: few[k] for k in METRICS}}
    summary = {'cases': {k: v for k, v in table.items() if not k.startswith('_')}, 'comparison_rows': table['_comparison_rows'],
               'fewshot_shortfall': shortfall,
               'perfect_sampler_bound_test': floor, 'perfect_sampler_bound_score': bound_scores,
               'failed_run_values': table['_failed_run_values'], 'seed_groups': groups, 'noise': noise,
               'paired_bootstrap_test': boot, 'standard_deviations': sigmas}
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    lines = [f"{'case':38s} {'test':>9s} {'train':>9s} {'feas.coh':>9s}  outcome"]
    for name in sorted(summary['cases'], key=lambda n: (-summary['cases'][n]['test_feasible'], -summary['cases'][n]['test'])):
        c = summary['cases'][name]
        outcome = 'feasible' if c['test_feasible'] else f"FAILED: {c['test_failure']}"
        fc = f"{c['test_feasible_cohort']:+9.4f}" if 'test_feasible_cohort' in c else ' ' * 9
        lines.append(f"{name:38s} {c['test']:+9.4f} {c['train']:+9.4f} {fc}  {outcome[:100]}")
    lines.append(f"{'perfect_sampler_bound (not an attempt)':38s} {bound_scores['test']:+9.4f} {'':9s} {bound_scores['test_feasible_cohort']:+9.4f}")
    lines.append('')
    lines.append('feasible-cohort score by experiment (test)')
    for name in sorted(summary['cases'], key=lambda n: -summary['cases'][n].get('test_feasible_cohort', -1e9)):
        by = summary['cases'][name].get('test_feasible_cohort_by_experiment')
        if by:
            lines.append(f"{name:38s} " + ' '.join(f'{k}={v:+.4f}' for k, v in sorted(by.items())))
    lines.append('')
    lines.append('few-shot shortfall: few-shot-task models on well-covered vs few-shot speakers (256 rows each, 32 speakers)')
    for case, r in shortfall.items():
        w, f = r['well_covered_speakers_160_191'], r['few_shot_test_speakers_224_255']
        lines.append(f"{case:38s} well-covered fd={w['codec_fd']:.4f} spk_fd={w['speaker_conditioned_codec_fd']:.4f} "
                     f"pair_js={w['codebook_pair_divergence']:.5f} | few-shot fd={f['codec_fd']:.4f} "
                     f"spk_fd={f['speaker_conditioned_codec_fd']:.4f} pair_js={f['codebook_pair_divergence']:.5f}")
    lines.append('')
    for name in sorted(summary['cases']):
        for row in stems:
            r = summary['cases'][name]['test_raw'].get(row, {})
            if r:
                lines.append(f"{name:38s} {row:20s} fd={r['codec_fd']:.4f} spk_fd={r['speaker_conditioned_codec_fd']:.4f} "
                             f"pair_js={r['codebook_pair_divergence']:.5f} excess={r['visible_window_excess']:.4f} "
                             f"half={r['copied_fraction']:.3f}")
    lines.append('')
    for name, r in sorted(table['_comparison_rows']['test'].items()):
        lines.append(f"{name:38s} fd={r['codec_fd']:.4f} spk_fd={r['speaker_conditioned_codec_fd']:.4f} pair_js={r['codebook_pair_divergence']:.5f}")
    lines.append('')
    for seed, rows in floor.items():
        for task, r in rows.items():
            lines.append(f"{'perfect_sampler_' + task + '_' + seed:38s} fd={r['codec_fd']:.4f} spk_fd={r['speaker_conditioned_codec_fd']:.4f} "
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
