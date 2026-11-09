#!/usr/bin/env python3
"""Generate the reproducible synthetic multi-codebook codec-token dataset.

Each sequence has a speaker id and `seq_len` frames of `num_codebooks` tokens.
The process keeps the structure that makes multi-codebook generation hard:

* within a frame, every residual codebook depends on the previous codebook,
  the speaker, the codebook index and the frame position, plus one bit of
  residual "style" (RVQ-like chain with a little stochasticity);
* across frames, the coarse codebook follows a Markov walk (four allowed
  increments, chosen uniformly), and the seven residual style bits are fixed
  for the whole utterance (drawn with speaker-specific probabilities), so a
  faithful generator must carry them from frame to frame.

Independent (parallel) per-codebook sampling can match every marginal while
producing invalid joint tuples; frame stacking forces a generator to decode
several temporally dependent frames from one coarse step.

Only the train and validation token splits are written. The evaluation split
is a list of conditioning speakers (`test_conditions.pt`): the reference
utterances it is scored against are drawn by the hidden evaluator.
"""

import argparse
from pathlib import Path

import torch

TRAIN_SEED = 1001
VAL_SEED = 1002
NUM_CONDITIONS = 16384


COARSE_INCREMENTS = (1, 6, 12, 19)


def speaker_tables(num_speakers, vocab_size, num_codebooks):
    """Coarse-walk increments (shared) and speaker-specific residual-style probabilities."""
    increments = torch.tensor(COARSE_INCREMENTS, dtype=torch.long).repeat(num_speakers, 1)
    style_p = torch.zeros(num_speakers, num_codebooks - 1)
    for s in range(num_speakers):
        for k in range(1, num_codebooks):
            style_p[s, k - 1] = 0.75 if (s + 2 * k) % 3 == 0 else 0.25
    return increments, style_p


def sample_tokens(speakers, seq_len, num_codebooks, vocab_size, num_speakers, seed):
    gen = torch.Generator().manual_seed(seed)
    increments, style_p = speaker_tables(num_speakers, vocab_size, num_codebooks)
    b = speakers.shape[0]
    style = (torch.rand(b, num_codebooks - 1, generator=gen) < style_p[speakers]).long()
    coarse = torch.zeros(b, seq_len, dtype=torch.long)
    coarse[:, 0] = torch.randint(vocab_size, (b,), generator=gen)
    for t in range(1, seq_len):
        choice = torch.randint(4, (b,), generator=gen)
        coarse[:, t] = (coarse[:, t - 1] + increments[speakers, choice]) % vocab_size
    tokens = torch.zeros(b, seq_len, num_codebooks, dtype=torch.long)
    tokens[..., 0] = coarse
    pos = torch.arange(seq_len).view(1, seq_len)
    spk = speakers.view(b, 1)
    for k in range(1, num_codebooks):
        tokens[..., k] = (
            5 * tokens[..., k - 1] + 7 * spk + 11 * k + pos + style[:, k - 1].view(b, 1)
        ) % vocab_size
    return tokens


def make_split(num_sequences, seq_len, num_codebooks, vocab_size, num_speakers, seed):
    gen = torch.Generator().manual_seed(seed + 7919)
    speakers = torch.randint(num_speakers, (num_sequences,), generator=gen)
    tokens = sample_tokens(speakers, seq_len, num_codebooks, vocab_size, num_speakers, seed)
    return {"tokens": tokens, "speaker": speakers}


def _compact(split):
    return {"tokens": split["tokens"].to(torch.uint8), "speaker": split["speaker"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_codec")
    parser.add_argument("--train", type=int, default=8192)
    parser.add_argument("--val", type=int, default=1024)
    args = parser.parse_args()

    seq_len, num_codebooks, vocab_size, num_speakers = 24, 8, 32, 16
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "seq_len": seq_len,
        "num_codebooks": num_codebooks,
        "vocab_size": vocab_size,
        "num_speakers": num_speakers,
        "num_conditions": NUM_CONDITIONS,
    }
    torch.save(_compact(make_split(args.train, seq_len, num_codebooks, vocab_size, num_speakers, TRAIN_SEED)), out / "train.pt")
    torch.save(_compact(make_split(args.val, seq_len, num_codebooks, vocab_size, num_speakers, VAL_SEED)), out / "val.pt")
    # Evaluation conditions: a balanced speaker list. No reference tokens.
    torch.save({"speaker": torch.arange(NUM_CONDITIONS) % num_speakers}, out / "test_conditions.pt")
    stale = out / "test.pt"
    if stale.exists():
        stale.unlink()
    torch.save(meta, out / "meta.pt")
    print(f"wrote synthetic codec dataset to {out}")


if __name__ == "__main__":
    main()
