#!/usr/bin/env python3
"""Train-slice evaluator for the synthetic multi-codebook codec task.

Scores the first half of the evaluation conditions against a reference sample
drawn from the codec process with a visible seed. It applies the same metrics,
operating points, failure threshold and caps as the hidden test, which scores
the other half against hidden reference utterances. Method artifacts
contribute generated tokens only.
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch

SLICE = "train"
SEQ_LEN, NUM_CODEBOOKS, VOCAB, NUM_SPEAKERS = 24, 8, 32, 16
NUM_CONDITIONS = 4096
PARTITION_SEED = 250919592
REFERENCES_PER_ROW = 4
RANDOM_CAP_SEED = 424_243
EMBED_SEED = 4242
# Canonical visible sequences: public train/validation splits and this
# slice's reference sample. Copying any of them is not generation.
VISIBLE_SPLITS = ((8192, 1001), (1024, 1002))
REFERENCE_SEED = 20_250_919
COPY_WINDOW = 12
MAX_COPIED_FRACTION = 0.01
VALIDITY_THRESHOLD = 0.5
EXPERIMENTS = {2: "codec_stack2", 4: "codec_stack4"}
REQUIRED_ROWS = {"generator_2x": 2, "generator_4x": 4}
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
    return order[: NUM_CONDITIONS // 2].sort().values


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


def frechet_distance(x, y):
    mx, my = x.mean(0), y.mean(0)
    cx, cy = torch.cov(x.T), torch.cov(y.T)
    ex, vx = torch.linalg.eigh(cx)
    sx = (vx * ex.clamp_min(0).sqrt()) @ vx.T
    m = sx @ cy @ sx
    cross = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(0).sqrt().sum()
    return float(((mx - my) ** 2).sum() + torch.trace(cx) + torch.trace(cy) - 2 * cross)


def distribution_metrics(tokens, speakers, ref_tokens, ref_speakers):
    fd = frechet_distance(frame_pair_embedding(tokens), frame_pair_embedding(ref_tokens))
    per_speaker = []
    for s in range(NUM_SPEAKERS):
        gen, ref = tokens[speakers == s], ref_tokens[ref_speakers == s]
        per_speaker.append(frechet_distance(frame_pair_embedding(gen), frame_pair_embedding(ref)))
    return fd, sum(per_speaker) / len(per_speaker)


def copied_fraction(tokens, visible_windows):
    copied = torch.zeros(tokens.shape[0], dtype=torch.bool)
    flat = tokens.to(torch.uint8)
    for offset in range(SEQ_LEN - COPY_WINDOW + 1):
        windows = flat[:, offset : offset + COPY_WINDOW].reshape(tokens.shape[0], -1)
        for i in range(tokens.shape[0]):
            if not copied[i] and (offset, windows[i].numpy().tobytes()) in visible_windows:
                copied[i] = True
    return float(copied.float().mean())


def build_visible_windows():
    seqs = [visible_split(n, seed) for n, seed in VISIBLE_SPLITS]
    seqs.append(sample_tokens(conditions()[slice_rows()].repeat_interleave(REFERENCES_PER_ROW), REFERENCE_SEED))
    windows = set()
    for tokens in seqs:
        flat = tokens.to(torch.uint8)
        for offset in range(SEQ_LEN - COPY_WINDOW + 1):
            block = flat[:, offset : offset + COPY_WINDOW].reshape(tokens.shape[0], -1).numpy()
            windows.update((offset, row.tobytes()) for row in block)
    return windows


# --------------------------------------------------------------------- scoring
def operating_point(name):
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
    pred = item["pred_tokens"]
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
    if declared <= 0 or SEQ_LEN % declared != 0 or declared < stack:
        return None, f"declared stack factor {declared!r} is impossible or exceeds the {SEQ_LEN // stack}-step budget of this row"
    return pred, None


def score_tokens(tokens, speakers, ref, ref_speakers, caps, visible_windows):
    raw = validity_parts(tokens, speakers)
    raw["codec_fd"], raw["speaker_conditioned_codec_fd"] = distribution_metrics(tokens, speakers, ref, ref_speakers)
    raw["copied_fraction"] = copied_fraction(tokens, visible_windows)
    failure = None
    if raw["copied_fraction"] > MAX_COPIED_FRACTION:
        failure = f"{raw['copied_fraction']:.3f} of sequences copy a {COPY_WINDOW}-frame window of visible data"
    elif raw["dependency_validity"] < VALIDITY_THRESHOLD:
        failure = f"dependency_validity {raw['dependency_validity']:.3f} < {VALIDITY_THRESHOLD}"
    if failure:
        scored = {"dependency_validity": WORST["dependency_validity"], **caps}
    else:
        scored = {
            "dependency_validity": raw["dependency_validity"],
            "codec_fd": min(raw["codec_fd"], caps["codec_fd"]),
            "speaker_conditioned_codec_fd": min(raw["speaker_conditioned_codec_fd"], caps["speaker_conditioned_codec_fd"]),
        }
    return scored, raw, failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="checkpoints/predictions")
    parser.add_argument("--out", default="scoring/scores_train.json")
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)

    rows = slice_rows()
    speakers = conditions()[rows]
    ref_speakers = speakers.repeat_interleave(REFERENCES_PER_ROW)
    ref = sample_tokens(ref_speakers, REFERENCE_SEED)
    random_tokens = torch.randint(VOCAB, (rows.numel(), SEQ_LEN, NUM_CODEBOOKS), generator=torch.Generator().manual_seed(RANDOM_CAP_SEED))
    cap_fd, cap_speaker_fd = distribution_metrics(random_tokens, speakers, ref, ref_speakers)
    caps = {"codec_fd": cap_fd, "speaker_conditioned_codec_fd": cap_speaker_fd}
    visible_windows = build_visible_windows()

    candidates = {path.stem: path for path in sorted(pred_dir.glob("*.pt"))}
    for name in REQUIRED_ROWS:
        candidates.setdefault(name, None)

    experiments = {name: {"slice": SLICE, "num_examples": int(rows.numel()), "results": {}} for name in EXPERIMENTS.values()}
    failures, raw_all = {}, {}
    for name, path in sorted(candidates.items()):
        stack = operating_point(name)
        if stack is None:
            failures[name] = "row name has no valid _<S>x operating point; not scored"
            continue
        experiment = EXPERIMENTS[4] if stack >= 4 else EXPERIMENTS[2]
        if path is None or not path.exists():
            tokens, failure = None, "missing prediction artifact"
        else:
            tokens, failure = load_tokens(path, stack)
        if tokens is None:
            scored = {"dependency_validity": WORST["dependency_validity"], **caps}
        else:
            scored, raw_all[name], failure = score_tokens(tokens[rows], speakers, ref, ref_speakers, caps, visible_windows)
        if failure:
            failures[name] = failure
        experiments[experiment]["results"][name] = scored

    output = {
        "_metadata": {
            "slice": SLICE,
            "num_examples": int(rows.numel()),
            "references_per_row": REFERENCES_PER_ROW,
            "validity_threshold": VALIDITY_THRESHOLD,
            "caps": caps,
            "method_source": str(pred_dir),
        },
        "_failures": failures,
        "_raw": raw_all,
        "experiments": {k: v for k, v in experiments.items() if v["results"]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
