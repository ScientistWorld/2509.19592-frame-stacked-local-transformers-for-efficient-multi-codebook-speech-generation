#!/usr/bin/env python3
"""Hidden test evaluator for the synthetic multi-codebook codec task.

Everything on the reference side of the comparison is fixed here: the codec
process, the hidden reference utterances (drawn at evaluation time from a seed
that exists only in this file), the evaluated rows, the operating points, the
failure threshold, artifact defaults and the comparison rows. Method artifacts
contribute generated tokens only; no number, flag or config value written by a
method is used as a score.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch

SLICE = "test"
SEQ_LEN, NUM_CODEBOOKS, VOCAB, NUM_SPEAKERS = 24, 8, 32, 16
NUM_CONDITIONS = 16384
PARTITION_SEED = 250919592
HIDDEN_REFERENCE_SEED = 73_517_903
REFERENCES_PER_ROW = 4
RANDOM_CAP_SEED = 424_243
EMBED_SEED = 4242
# Canonical visible sequences: public train/validation splits and the
# train-slice reference sample. Copying any of them is not generation.
VISIBLE_SPLITS = ((8192, 1001), (1024, 1002))
TRAIN_SLICE_REFERENCE_SEED = 20_250_919
COPY_WINDOW = SEQ_LEN
MAX_COPIED_FRACTION = 0.01
VALIDITY_THRESHOLD = 0.5
COARSE_THRESHOLD = 0.5
INTRA_THRESHOLD = 0.5
EXPERIMENTS = {2: "codec_stack2", 4: "codec_stack4"}
REQUIRED_ROWS = {"generator_2x": 2, "generator_4x": 4}
FIXED_DIR = Path(__file__).resolve().parent / "fixed"
FIXED_ROWS = {
    "baseline_parallel_1x": 1,
    "baseline_parallel_2x": 2,
    "baseline_parallel_4x": 4,
    "baseline_maskgit_2x": 2,
    "baseline_maskgit_4x": 4,
}
WORST = {"dependency_validity": 0.0}


# ----------------------------------------------------------------- codec process
def speaker_tables():
    increments = torch.tensor((1, 6, 12, 19), dtype=torch.long).repeat(NUM_SPEAKERS, 1)
    style_p = torch.zeros(NUM_SPEAKERS, NUM_CODEBOOKS - 1)
    for s in range(NUM_SPEAKERS):
        for k in range(1, NUM_CODEBOOKS):
            style_p[s, k - 1] = 0.75 if (s + 2 * k) % 3 == 0 else 0.25
    return increments, style_p


INCREMENTS, STYLE_P = speaker_tables()


def sample_tokens(speakers, seed):
    gen = torch.Generator().manual_seed(seed)
    b = speakers.shape[0]
    style = (torch.rand(b, NUM_CODEBOOKS - 1, generator=gen) < STYLE_P[speakers]).long()
    coarse = torch.zeros(b, SEQ_LEN, dtype=torch.long)
    coarse[:, 0] = torch.randint(VOCAB, (b,), generator=gen)
    for t in range(1, SEQ_LEN):
        choice = torch.randint(4, (b,), generator=gen)
        coarse[:, t] = (coarse[:, t - 1] + INCREMENTS[speakers, choice]) % VOCAB
    tokens = torch.zeros(b, SEQ_LEN, NUM_CODEBOOKS, dtype=torch.long)
    tokens[..., 0] = coarse
    pos = torch.arange(SEQ_LEN).view(1, SEQ_LEN)
    spk = speakers.view(b, 1)
    for k in range(1, NUM_CODEBOOKS):
        tokens[..., k] = (5 * tokens[..., k - 1] + 7 * spk + 11 * k + pos + style[:, k - 1].view(b, 1)) % VOCAB
    return tokens


def visible_split(num, seed):
    speakers = torch.randint(NUM_SPEAKERS, (num,), generator=torch.Generator().manual_seed(seed + 7919))
    return sample_tokens(speakers, seed)


def conditions():
    return torch.arange(NUM_CONDITIONS) % NUM_SPEAKERS


def slice_rows():
    order = torch.randperm(NUM_CONDITIONS, generator=torch.Generator().manual_seed(PARTITION_SEED))
    return order[NUM_CONDITIONS // 2 :].sort().values


# ---------------------------------------------------------------------- metrics
def validity_parts(tokens, speakers):
    b = tokens.shape[0]
    k = torch.arange(1, NUM_CODEBOOKS).view(1, 1, -1)
    pos = torch.arange(SEQ_LEN).view(1, SEQ_LEN, 1)
    residual = (tokens[..., 1:] - 5 * tokens[..., :-1] - 7 * speakers.view(b, 1, 1) - 11 * k - pos) % VOCAB
    intra = residual <= 1
    step = (tokens[:, 1:, 0] - tokens[:, :-1, 0]) % VOCAB
    coarse = (step.unsqueeze(-1) == INCREMENTS[speakers].unsqueeze(1)).any(-1)
    # Diagnostic only (distributional, not a hard check): fraction of
    # consecutive frames whose residual style bit is unchanged (1.0 in data).
    persist = intra[:, 1:] & intra[:, :-1] & (residual[:, 1:] == residual[:, :-1])
    total = intra.numel() + coarse.numel()
    return {
        "dependency_validity": float((intra.sum() + coarse.sum()).item() / total),
        "intra_frame_validity": float(intra.float().mean()),
        "coarse_transition_validity": float(coarse.float().mean()),
        "residual_persistence": float(persist.float().mean()),
    }


CODEC_EMBED = torch.randn(NUM_CODEBOOKS, VOCAB, 32, generator=torch.Generator().manual_seed(EMBED_SEED)).double()
CODEC_EMBED = CODEC_EMBED * (0.75 ** torch.arange(NUM_CODEBOOKS, dtype=torch.float64)).view(-1, 1, 1)


def frame_pair_embedding(tokens):
    """RVQ-style codec latent (sum of per-codebook vectors) for consecutive frame pairs."""
    z = CODEC_EMBED[torch.arange(NUM_CODEBOOKS).view(1, 1, -1), tokens].sum(2)
    return torch.cat([z[:, :-1], z[:, 1:]], dim=-1).reshape(-1, 64)


def feature_moments(tokens, speakers):
    sums = torch.zeros(NUM_SPEAKERS + 1, 64, dtype=torch.float64)
    products = torch.zeros(NUM_SPEAKERS + 1, 64, 64, dtype=torch.float64)
    counts = torch.zeros(NUM_SPEAKERS + 1, dtype=torch.float64)
    for start in range(0, len(tokens), 256):
        batch = tokens[start:start + 256]
        spk = speakers[start:start + 256]
        pairs = frame_pair_embedding(batch).reshape(len(batch), SEQ_LEN - 1, 64)
        for speaker in range(NUM_SPEAKERS + 1):
            x = pairs if speaker == NUM_SPEAKERS else pairs[spk == speaker]
            x = x.reshape(-1, 64)
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


def distribution_metrics(tokens, speakers, ref_moments):
    means, cov = feature_moments(tokens, speakers)
    rm, rc = ref_moments
    distances = [frechet_moments(means[i], cov[i], rm[i], rc[i]) for i in range(NUM_SPEAKERS + 1)]
    return distances[-1], sum(distances[:-1]) / NUM_SPEAKERS


def copied_fraction(tokens, visible_windows):
    flat = tokens.to(torch.uint8).reshape(tokens.shape[0], -1).numpy()
    return sum(row.tobytes() in visible_windows for row in flat) / len(flat)


def build_visible_windows():
    # Frozen canonical visible examples, never mutable method-side data.
    item = torch.load(FIXED_DIR / 'visible.pt', map_location='cpu', weights_only=True)
    return {row.tobytes() for row in item['tokens'].reshape(-1, SEQ_LEN * NUM_CODEBOOKS).numpy()}


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


def score_tokens(tokens, speakers, ref_moments, caps, visible_windows):
    raw = validity_parts(tokens, speakers)
    raw['codec_fd'], raw['speaker_conditioned_codec_fd'] = distribution_metrics(tokens, speakers, ref_moments)
    raw['copied_fraction'] = copied_fraction(tokens, visible_windows)
    failure = None
    if any(not math.isfinite(v) for v in raw.values()):
        failure = 'nonfinite metric'
    elif raw['copied_fraction'] > MAX_COPIED_FRACTION:
        failure = f"{raw['copied_fraction']:.3f} of sequences duplicate a complete visible utterance"
    elif raw['dependency_validity'] < VALIDITY_THRESHOLD:
        failure = f"dependency_validity {raw['dependency_validity']:.3f} < {VALIDITY_THRESHOLD}"
    elif raw['coarse_transition_validity'] < COARSE_THRESHOLD:
        failure = 'coarse transition validity below 0.5'
    elif raw['intra_frame_validity'] < INTRA_THRESHOLD:
        failure = 'within-frame validity below 0.5'
    # Keep scientific measurements for diagnostics and baseline comparisons.
    # Failure is a separate rank tier, so good marginals can never hide failure.
    scored = {key: raw[key] for key in ('dependency_validity', 'codec_fd', 'speaker_conditioned_codec_fd')}
    return scored, raw, failure


_FIXED_CACHE = {}


def evaluate_directory(pred_dir, rows, speakers, ref_moments, caps, visible_windows):
    candidates = {path.stem: path for path in sorted(pred_dir.glob('*.pt'))}
    for name in FIXED_ROWS:
        candidates[name] = FIXED_DIR / f'{name}.pt'
    for name in REQUIRED_ROWS:
        candidates.setdefault(name, None)
    experiments = {name: {'slice': SLICE, 'num_examples': int(rows.numel()), 'results': {}} for name in EXPERIMENTS.values()}
    failures, raw_all = {}, {}
    integrity_failure = None
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
        if name in _FIXED_CACHE:
            scored, raw_all[name], failure = _FIXED_CACHE[name]
            experiments[experiment]['results'][name] = scored
            if failure:
                failures[name] = failure
            continue
        if path is None or not path.exists():
            tokens, failure = None, 'missing prediction artifact'
        else:
            tokens, failure = load_tokens(path, stack)
        if tokens is None:
            scored = {'dependency_validity': 0.0, **caps}
        else:
            scored, raw_all[name], failure = score_tokens(tokens[rows], speakers, ref_moments, caps, visible_windows)
        if name in FIXED_ROWS and name in raw_all:
            _FIXED_CACHE[name] = (scored, raw_all[name], failure)
        if integrity_failure and name not in FIXED_ROWS:
            failure = integrity_failure
        if failure:
            failures[name] = failure
        experiments[experiment]['results'][name] = scored
    return {'_failures': failures, '_raw': raw_all, 'experiments': experiments}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="checkpoints/predictions")
    parser.add_argument("--out", default="scoring/scores_test.json")
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)

    rows = slice_rows()
    speakers = conditions()[rows]
    ref_speakers = speakers.repeat_interleave(REFERENCES_PER_ROW)
    ref = sample_tokens(ref_speakers, HIDDEN_REFERENCE_SEED)
    ref_moments = feature_moments(ref, ref_speakers)
    random_tokens = torch.randint(VOCAB, (rows.numel(), SEQ_LEN, NUM_CODEBOOKS), generator=torch.Generator().manual_seed(RANDOM_CAP_SEED))
    cap_fd, cap_speaker_fd = distribution_metrics(random_tokens, speakers, ref_moments)
    caps = {'codec_fd': cap_fd, 'speaker_conditioned_codec_fd': cap_speaker_fd}
    visible_windows = build_visible_windows()
    output = evaluate_directory(pred_dir, rows, speakers, ref_moments, caps, visible_windows)
    output['_metadata'] = {'slice': SLICE, 'num_examples': int(rows.numel()),
                           'references_per_row': REFERENCES_PER_ROW, 'validity_threshold': VALIDITY_THRESHOLD,
                           'method_source': str(pred_dir), 'invalid_artifact_defaults': caps}
    # Rank every saved attempt from its raw predictions, never submitted scores.
    from eval.test.ranking import rank_attempts
    reference = evaluate_directory(FIXED_DIR / 'reference_predictions', rows, speakers, ref_moments, caps, visible_windows)
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
            view['experiments'][exp] = {'results': {canonical: values or {'dependency_validity': 0.0, **caps}}}
            if values is None or actual in output['_failures']:
                view['_failures'][canonical] = output['_failures'].get(actual, 'missing operating point')
        attempts[f'files:{prefix}'] = view
    attempts_dir = pred_dir.parent / 'attempts'
    if attempts_dir.is_dir():
        for folder in sorted(attempts_dir.iterdir()):
            if (folder / 'predictions').is_dir():
                label = folder.name if folder.name not in attempts else f'folder:{folder.name}'
                attempts[label] = evaluate_directory(folder / 'predictions', rows, speakers, ref_moments, caps, visible_windows)
    spec = json.loads(Path('scoring/reference.json').read_text())
    output['_ranking'] = rank_attempts(attempts, reference, spec)
    output['_attempts'] = {name: value for name, value in attempts.items() if name != 'current'}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({'slice': SLICE, 'failures': output['_failures'], 'ranking': output['_ranking']}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
