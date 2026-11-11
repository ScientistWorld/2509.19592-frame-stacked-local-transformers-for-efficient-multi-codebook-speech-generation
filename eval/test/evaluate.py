#!/usr/bin/env python3
"""Hidden test evaluator for the synthetic multi-codebook codec task.

Everything on the reference side of the comparison is fixed here: the codec
process (eval/test/codec_process.py), the hidden reference utterances (drawn
at evaluation time from a seed that exists only in this file), the evaluated
rows, the operating points, the failure thresholds, artifact defaults and the
comparison rows. Method artifacts contribute generated tokens only; no number,
flag or config value written by a method is used as a score.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from eval.test import codec_process as process

SLICE = "test"
SEQ_LEN, NUM_CODEBOOKS, VOCAB, NUM_SPEAKERS = 24, 8, 32, 16
NUM_CONDITIONS = 16384
PARTITION_SEED = 250919592
HIDDEN_REFERENCE_SEED = 83_160_271
REFERENCES_PER_ROW = 4
RANDOM_CAP_SEED = 424_243
# Canonical visible sequences: public train/validation splits and the
# train-slice reference sample. Copying any of them is not generation.
VISIBLE_SPLITS = ((32768, 1001), (2048, 1002))
TRAIN_SLICE_REFERENCE_SEED = 20_250_919
COPY_WINDOW = SEQ_LEN // 2  # aligned 12-frame halves, all 8 codebooks
MAX_COPIED_FRACTION = 0.01
# Memorization test on 4-frame windows (32 tokens, same frame position). A
# window "hits" a set if it agrees with one of the set's windows on all 32
# tokens or on any of 8 fixed subsets of 12 of its tokens, so copies with
# scattered edits still hit. The excess is the hit rate against the visible
# utterances minus the hit rate against a fresh sample of the data process
# with the same size and speaker composition (hidden seed): its expectation
# is 0 for any generator that does not reproduce visible segments.
EXCESS_WINDOW = 4
WINDOW_TOKENS = EXCESS_WINDOW * NUM_CODEBOOKS
NUM_SUBSETS, SUBSET_SIZE, SUBSET_SEED = 8, 12, 8_675_309
MAX_WINDOW_EXCESS = 0.05
# A run whose distribution is not clearly closer to the reference than
# uniform random tokens fails the task: every scored measurement must be
# below this fraction of the random-token measurement.
FAIL_FRACTION_OF_RANDOM = 0.5
CONTROL_SEED = 61_803_398
EXPERIMENTS = {2: "codec_stack2", 4: "codec_stack4"}
REQUIRED_ROWS = {"generator_2x": 2, "generator_4x": 4}
SCORED = ('codec_fd', 'speaker_conditioned_codec_fd', 'codebook_pair_divergence')
FIXED_DIR = Path(__file__).resolve().parent / "fixed"
FIXED_ROWS = {
    "baseline_parallel_1x": 1,
    "baseline_parallel_2x": 2,
    "baseline_parallel_4x": 4,
    "baseline_maskgit_2x": 2,
    "baseline_maskgit_4x": 4,
}


def sample_tokens(speakers, seed):
    return process.sample_tokens(speakers, seed)


def visible_split(num, seed):
    return process.make_split(num, seed)['tokens']


def conditions():
    return torch.arange(NUM_CONDITIONS) % NUM_SPEAKERS


def slice_rows():
    order = torch.randperm(NUM_CONDITIONS, generator=torch.Generator().manual_seed(PARTITION_SEED))
    return order[NUM_CONDITIONS // 2 :].sort().values


# ---------------------------------------------------------------------- metrics
_LEVELS = torch.tensor(process.LEVELS)
_RADIX = torch.tensor([process.LEVELS[1] * process.LEVELS[2], process.LEVELS[2], 1])
LATENT = NUM_CODEBOOKS * len(process.LEVELS)        # 24-d codec latent per frame
PAIR = 2 * LATENT                                   # consecutive-frame pair: 48-d


def codec_latent(tokens):
    """Public FSQ dequantization: each codebook token -> centres of its 4x4x2 grid cell in [-1, 1]^3."""
    t = tokens.long().unsqueeze(-1)
    digits = (t // _RADIX) % _LEVELS
    centres = (digits.double() + 0.5) / _LEVELS * 2 - 1
    return centres.reshape(*tokens.shape[:-1], LATENT)


def feature_moments(tokens, speakers):
    sums = torch.zeros(NUM_SPEAKERS + 1, PAIR, dtype=torch.float64)
    products = torch.zeros(NUM_SPEAKERS + 1, PAIR, PAIR, dtype=torch.float64)
    counts = torch.zeros(NUM_SPEAKERS + 1, dtype=torch.float64)
    for start in range(0, len(tokens), 1024):
        z = codec_latent(tokens[start:start + 1024])
        spk = speakers[start:start + 1024]
        pairs = torch.cat([z[:, :-1], z[:, 1:]], dim=-1)
        for speaker in range(NUM_SPEAKERS + 1):
            x = pairs if speaker == NUM_SPEAKERS else pairs[spk == speaker]
            x = x.reshape(-1, PAIR)
            if len(x):
                counts[speaker] += len(x)
                sums[speaker] += x.sum(0)
                products[speaker] += x.T @ x
    means = sums / counts[:, None]
    cov = (products - sums[:, :, None] * sums[:, None, :] / counts[:, None, None]) / (counts - 1)[:, None, None]
    return means, cov


def frechet_moments(mx, cx, my, cy):
    ex, vx = torch.linalg.eigh(cx)
    sx = (vx * ex.clamp_min(0).sqrt()) @ vx.T
    m = sx @ cy @ sx
    cross = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(0).sqrt().sum()
    value = float(((mx - my) ** 2).sum() + torch.trace(cx) + torch.trace(cy) - 2 * cross)
    if not math.isfinite(value) or value < -1e-7:
        raise ValueError('nonfinite or impossible Frechet distance')
    return max(0.0, value)


WITHIN_PAIRS = [(i, j) for i in range(NUM_CODEBOOKS) for j in range(i + 1, NUM_CODEBOOKS)]


def pair_counts(tokens):
    """Joint token counts for the 28 within-frame codebook pairs and the 64
    (codebook at frame t, codebook at frame t+1) pairs: (92, 32*32) int64."""
    t = tokens.long()
    frames = t.reshape(-1, NUM_CODEBOOKS)
    cur, nxt = t[:, :-1].reshape(-1, NUM_CODEBOOKS), t[:, 1:].reshape(-1, NUM_CODEBOOKS)
    rows = [torch.bincount(frames[:, i] * VOCAB + frames[:, j], minlength=VOCAB * VOCAB) for i, j in WITHIN_PAIRS]
    rows += [torch.bincount(cur[:, i] * VOCAB + nxt[:, j], minlength=VOCAB * VOCAB)
             for i in range(NUM_CODEBOOKS) for j in range(NUM_CODEBOOKS)]
    return torch.stack(rows)


def pair_divergences(counts, ref_counts):
    """Jensen-Shannon divergence (bits, in [0, 1]) per pair view."""
    p = counts.double() / counts.sum(1, keepdim=True)
    q = ref_counts.double() / ref_counts.sum(1, keepdim=True)
    m = (p + q) / 2

    def kl(a):
        return torch.where(a > 0, a * torch.log2(a / m.clamp_min(1e-300)), torch.zeros_like(a)).sum(1)

    js = 0.5 * kl(p) + 0.5 * kl(q)
    if not bool(torch.isfinite(js).all()) or float(js.min()) < -1e-9:
        raise ValueError('nonfinite or impossible divergence')
    return js.clamp_min(0.0)


def distribution_metrics(tokens, speakers, ref):
    means, cov = feature_moments(tokens, speakers)
    rm, rc = ref['moments']
    distances = [frechet_moments(means[i], cov[i], rm[i], rc[i]) for i in range(NUM_SPEAKERS + 1)]
    js = pair_divergences(pair_counts(tokens), ref['pairs'])
    within = len(WITHIN_PAIRS)
    return {
        'codec_fd': distances[-1],
        'speaker_conditioned_codec_fd': sum(distances[:-1]) / NUM_SPEAKERS,
        'codebook_pair_divergence': float(js.mean()),
        'within_frame_pair_divergence': float(js[:within].mean()),
        'temporal_pair_divergence': float(js[within:].mean()),
    }


# ----------------------------------------------------------- memorization tests
def window_keys(tokens):
    """Per utterance, byte keys of its aligned 12-frame halves (all codebooks)."""
    halves = tokens.to(torch.uint8).reshape(tokens.shape[0], SEQ_LEN // COPY_WINDOW, -1).numpy()
    return [[bytes([h]) + half.tobytes() for h, half in enumerate(row)] for row in halves]


def copied_fraction(tokens, windows):
    """Fraction of utterances with at least one half identical to a half in `windows`."""
    return sum(any(key in windows for key in keys) for keys in window_keys(tokens)) / len(tokens)


def window_set(tokens):
    return {key for keys in window_keys(tokens) for key in keys}


def _hash_tables():
    g = torch.Generator().manual_seed(SUBSET_SEED)
    masks = torch.zeros(NUM_SUBSETS + 1, WINDOW_TOKENS, dtype=torch.long)
    masks[0] = 1  # the whole window
    for r in range(1, NUM_SUBSETS + 1):
        masks[r, torch.randperm(WINDOW_TOKENS, generator=g)[:SUBSET_SIZE]] = 1
    mult = (torch.randint(0, 2 ** 61, (NUM_SUBSETS + 1, WINDOW_TOKENS), generator=g) * 2 + 1) * masks
    salt = torch.randint(0, 2 ** 62, (SEQ_LEN, NUM_SUBSETS + 1), generator=g)
    return mult.numpy().astype(np.uint64).T.copy(), salt.numpy().astype(np.uint64)


_MULT, _SALT = _hash_tables()
NUM_STARTS = SEQ_LEN - EXCESS_WINDOW + 1


def subset_keys(tokens):
    """(b, 21 starts, 9) 48-bit keys: the whole 4-frame window and 8 token subsets, salted by start."""
    t = tokens.long()
    w = t.unfold(1, EXCESS_WINDOW, 1).permute(0, 1, 3, 2).reshape(t.shape[0], NUM_STARTS, WINDOW_TOKENS)
    x = (w + 1).numpy().astype(np.uint64)
    with np.errstate(over='ignore'):
        keys = np.matmul(x, _MULT) + _SALT[:NUM_STARTS]
    return keys >> np.uint64(16)  # 48-bit keys


def key_set(tokens):
    keys = []
    for start in range(0, len(tokens), 8192):
        keys.append(subset_keys(tokens[start:start + 8192]).ravel())
    return np.unique(np.concatenate(keys))


def _hits(keys, sorted_keys):
    idx = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[idx] == keys


def window_excess(tokens, visible_keys, control_keys):
    """(any-key excess, whole-window excess): fraction of 4-frame windows that hit
    the visible set minus the fraction that hit the equally composed fresh sample."""
    keys = subset_keys(tokens)
    vis, ctl = _hits(keys, visible_keys), _hits(keys, control_keys)
    return (float(vis.any(-1).mean() - ctl.any(-1).mean()), float(vis[..., 0].mean() - ctl[..., 0].mean()))


def visible_speakers():
    """Speakers of fixed/visible.pt in order: train split, validation split,
    then the train slice's four reference utterances per conditioning row."""
    parts = [torch.randint(NUM_SPEAKERS, (num,), generator=torch.Generator().manual_seed(seed + 7919))
             for num, seed in VISIBLE_SPLITS]
    order = torch.randperm(NUM_CONDITIONS, generator=torch.Generator().manual_seed(PARTITION_SEED))
    parts.append(conditions()[order[:NUM_CONDITIONS // 2].sort().values].repeat_interleave(REFERENCES_PER_ROW))
    return torch.cat(parts)


def build_memorization_sets():
    # Frozen canonical visible examples, never mutable method-side data.
    item = torch.load(FIXED_DIR / 'visible.pt', map_location='cpu', weights_only=True)
    visible = item['tokens'].reshape(-1, SEQ_LEN, NUM_CODEBOOKS)
    speakers = visible_speakers()
    if len(speakers) != len(visible):
        raise RuntimeError('visible fixture does not match its speaker composition')
    control = sample_tokens(speakers, CONTROL_SEED)
    return {'halves': window_set(visible), 'visible_keys': key_set(visible), 'control_keys': key_set(control)}


# --------------------------------------------------------------------- scoring
def operating_point(name):
    if name in FIXED_ROWS:
        return FIXED_ROWS[name]
    match = re.search(r"_(\d+)x$", name)
    if not match:
        return None
    stack = int(match.group(1))
    return stack if stack > 0 and SEQ_LEN % stack == 0 else None


def load_tokens(path, stack):
    """Return (tokens, failure_reason). Only generated tokens are used."""
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # unreadable or unsafe artifact
        return None, f"artifact could not be loaded safely: {type(exc).__name__}"
    if not isinstance(item, dict) or not torch.is_tensor(item.get("pred_tokens")):
        return None, "artifact has no pred_tokens tensor"
    if set(item) - {'pred_tokens', 'config', 'split'}:
        return None, 'unsupported artifact fields (submitted metrics and verdicts are not accepted)'
    pred = item["pred_tokens"]
    if pred.layout != torch.strided:
        return None, "pred_tokens must be a dense tensor"
    if pred.dtype.is_floating_point or pred.dtype.is_complex or pred.dtype == torch.bool:
        return None, f"pred_tokens must be an integer tensor, got {pred.dtype}"
    if tuple(pred.shape) != (NUM_CONDITIONS, SEQ_LEN, NUM_CODEBOOKS):
        return None, f"pred_tokens shape {tuple(pred.shape)} != {(NUM_CONDITIONS, SEQ_LEN, NUM_CODEBOOKS)}"
    pred = pred.long()
    if int(pred.min()) < 0 or int(pred.max()) >= VOCAB:
        return None, "token ids outside [0, vocab_size)"
    cfg = item.get("config")
    declared = cfg.get("stack_factor") if isinstance(cfg, dict) else None
    if isinstance(declared, bool) or not isinstance(declared, int):
        return None, "config.stack_factor must be a positive integer"
    if declared <= 0 or SEQ_LEN % declared != 0 or declared != stack:
        return None, f"declared stack factor {declared!r} does not match the fixed {SEQ_LEN // stack}-step budget of this row"
    for key in ('vocab_size', 'num_codebooks', 'seq_len', 'num_speakers'):
        expected = {'vocab_size': VOCAB, 'num_codebooks': NUM_CODEBOOKS, 'seq_len': SEQ_LEN, 'num_speakers': NUM_SPEAKERS}[key]
        if key in cfg and (type(cfg[key]) is not int or cfg[key] != expected):
            return None, f'config.{key} conflicts with the fixed task'
    return pred, None


def score_tokens(tokens, ctx):
    raw = distribution_metrics(tokens, ctx['speakers'], ctx['reference'])
    raw['copied_fraction'] = copied_fraction(tokens, ctx['memorization']['halves'])
    raw['visible_window_excess'], raw['visible_exact_window_excess'] = window_excess(
        tokens, ctx['memorization']['visible_keys'], ctx['memorization']['control_keys'])
    caps = ctx['caps']
    failure = None
    if any(not math.isfinite(v) for v in raw.values()):
        failure = 'nonfinite metric'
    elif raw['copied_fraction'] > MAX_COPIED_FRACTION:
        failure = f"{raw['copied_fraction']:.3f} of sequences repeat a 12-frame half of a visible utterance"
    elif raw['visible_window_excess'] > MAX_WINDOW_EXCESS:
        failure = (f"{raw['visible_window_excess']:.3f} excess of 4-frame windows matching visible utterances "
                   'over a fresh data-process sample (copied, edited or spliced segments)')
    else:
        for key in SCORED:
            if raw[key] >= FAIL_FRACTION_OF_RANDOM * caps[key]:
                failure = f'{key} is not below half that of uniform random tokens'
                break
    # Raw measurements are returned here; evaluate_directory replaces every
    # scored value of a failed run by its worst value.
    scored = {key: raw[key] for key in SCORED}
    return scored, raw, failure


_FIXED_CACHE = {}


def run_groups(names):
    """Rows of one run: `<prefix>_2x` and `<prefix>_4x` belong together."""
    groups = {}
    for name in names:
        prefix, _, point = name.rpartition('_')
        groups.setdefault(prefix if point in ('2x', '4x') else name, []).append(name)
    return groups


def evaluate_directory(pred_dir, ctx, check_integrity=True):
    rows, worst = ctx['rows'], dict(ctx['caps'])
    candidates = {path.stem: path for path in sorted(pred_dir.glob('*.pt'))}
    for name in FIXED_ROWS:
        candidates[name] = FIXED_DIR / f'{name}.pt'
    for name in REQUIRED_ROWS:
        candidates.setdefault(name, None)
    experiments = {name: {'slice': SLICE, 'num_examples': int(rows.numel()), 'results': {}} for name in EXPERIMENTS.values()}
    failures, raw_all, comparison_status, generated, full = {}, {}, {}, {}, {}
    integrity_failure = None
    if check_integrity:
        expected_hashes = json.loads((FIXED_DIR / 'data_hashes.json').read_text())
        for relative, expected in expected_hashes.items():
            path = Path(relative)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                integrity_failure = 'fixed training data changed'
        if Path('data/synthetic_codec/test.pt').exists():
            integrity_failure = 'forbidden method-side target file'
        for name, stack in FIXED_ROWS.items():
            submitted = pred_dir / f'{name}.pt'
            if submitted.exists():
                got, reason = load_tokens(submitted, stack)
                fixed, _ = load_tokens(FIXED_DIR / f'{name}.pt', stack)
                if reason or not torch.equal(got[rows], fixed[rows]):
                    integrity_failure = 'reserved fixed comparison output changed'

    for name, path in sorted(candidates.items()):
        stack = operating_point(name)
        if stack is None or stack not in (1, 2, 4):
            failures[name] = 'file must identify a fixed 1x, 2x or 4x operating point'
            continue
        experiment = EXPERIMENTS[4] if stack == 4 else EXPERIMENTS[2]
        if name in FIXED_ROWS:
            # Evaluator-frozen comparison rows: raw measurements are shown, and
            # failing a check is a comparison outcome, not a run failure.
            if name not in _FIXED_CACHE:
                tokens, reason = load_tokens(path, stack)
                if tokens is None:
                    raise RuntimeError(f'frozen comparison {name} is unreadable: {reason}')
                _FIXED_CACHE[name] = score_tokens(tokens[rows], ctx)
            scored, raw_all[name], status = _FIXED_CACHE[name]
            experiments[experiment]['results'][name] = dict(scored)
            if status:
                comparison_status[name] = status
            continue
        if path is None or not path.exists():
            tokens, failure = None, 'missing prediction artifact'
        else:
            tokens, failure = load_tokens(path, stack)
        scored = dict(worst)
        if tokens is not None:
            full[name] = tokens
            generated[name] = tokens[rows]
            scored, raw_all[name], failure = score_tokens(generated[name], ctx)
        failure = integrity_failure or failure
        if failure:
            failures[name] = failure
        experiments[experiment]['results'][name] = scored

    groups = run_groups([n for exp in experiments.values() for n in exp['results'] if n not in FIXED_ROWS])
    for names in groups.values():
        # The two operating points must be separately generated samples: no
        # scored row may repeat a 12-frame half found anywhere (any row, scored
        # or not) in the sibling operating point's file.
        present = [n for n in sorted(names) if n in generated]
        for a in present:
            reused = window_set(full[a])
            for b in present:
                if b == a:
                    continue
                fraction = copied_fraction(generated[b], reused)
                if fraction > MAX_COPIED_FRACTION:
                    for n in (a, b):
                        failures.setdefault(n, f'{fraction:.3f} of {b} sequences repeat a 12-frame half of {a}')
        # A failed run gets the worst value of every scored metric in every
        # experiment: no failed row can carry a good measurement into a rank.
        failed = [failures[n] for n in sorted(names) if n in failures]
        if failed:
            for n in names:
                failures.setdefault(n, f'run failed: {failed[0]}')
                for exp in experiments.values():
                    if n in exp['results']:
                        exp['results'][n] = dict(worst)
    return {'_failures': failures, '_comparison_status': comparison_status, '_raw': raw_all, 'experiments': experiments}


def build_context():
    rows = slice_rows()
    speakers = conditions()[rows]
    ref_speakers = speakers.repeat_interleave(REFERENCES_PER_ROW)
    ref = sample_tokens(ref_speakers, HIDDEN_REFERENCE_SEED)
    reference = {'moments': feature_moments(ref, ref_speakers), 'pairs': pair_counts(ref)}
    ctx = {'rows': rows, 'speakers': speakers, 'reference': reference}
    random_tokens = torch.randint(VOCAB, (rows.numel(), SEQ_LEN, NUM_CODEBOOKS), generator=torch.Generator().manual_seed(RANDOM_CAP_SEED))
    cap = distribution_metrics(random_tokens, speakers, reference)
    ctx['caps'] = {key: cap[key] for key in SCORED}
    ctx['memorization'] = build_memorization_sets()
    return ctx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="checkpoints/predictions")
    parser.add_argument("--out", default="scoring/scores_test.json")
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)

    ctx = build_context()
    output = evaluate_directory(pred_dir, ctx)
    output['_metadata'] = {'slice': SLICE, 'num_examples': int(ctx['rows'].numel()),
                           'references_per_row': REFERENCES_PER_ROW,
                           'max_copied_half_fraction': MAX_COPIED_FRACTION, 'max_window_excess': MAX_WINDOW_EXCESS,
                           'method_source': str(pred_dir),
                           'failed_run_values': dict(ctx['caps']),
                           'failed_run_values_note': 'worst value of each scored metric: the measurement of uniform random tokens; '
                                                     'a run fails if any measurement is not below half of it'}
    # Rank every saved attempt from its raw predictions, never submitted scores.
    from eval.test.ranking import rank_attempts
    reference = evaluate_directory(FIXED_DIR / 'reference_predictions', ctx, check_integrity=False)
    attempts = {'current': output}
    # Additional flat files use the same interface, grouped by their prefix.
    all_rows = {name for exp in output['experiments'].values() for name in exp['results']}
    prefixes = {name.rsplit('_', 1)[0] for name in all_rows
                if name not in FIXED_ROWS and name not in REQUIRED_ROWS}
    for prefix in sorted(prefixes):
        view = {'experiments': {}, '_failures': {}}
        for stack, exp in EXPERIMENTS.items():
            actual, canonical = f'{prefix}_{stack}x', f'generator_{stack}x'
            values = output['experiments'][exp]['results'].get(actual)
            failed = values is None or actual in output['_failures']
            view['experiments'][exp] = {'results': {canonical: dict(ctx['caps']) if failed else values}}
            if failed:
                view['_failures'][canonical] = output['_failures'].get(actual, 'missing operating point')
        attempts[f'files:{prefix}'] = view
    attempts_dir = pred_dir.parent / 'attempts'
    if attempts_dir.is_dir():
        for folder in sorted(attempts_dir.iterdir()):
            if (folder / 'predictions').is_dir():
                label = folder.name if folder.name not in attempts else f'folder:{folder.name}'
                attempts[label] = evaluate_directory(folder / 'predictions', ctx)
    spec = json.loads(Path('scoring/reference.json').read_text())
    output['_ranking'] = rank_attempts(attempts, reference, spec)
    output['_attempts'] = {name: value for name, value in attempts.items() if name != 'current'}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({'slice': SLICE, 'failures': output['_failures'], 'ranking': output['_ranking']}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
