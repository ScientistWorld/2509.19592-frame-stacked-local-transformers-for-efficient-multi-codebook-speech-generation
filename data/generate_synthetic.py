#!/usr/bin/env python3
"""Generate a reproducible multi-codebook token dataset.

The synthetic task preserves the paper's key structure: every temporal frame has
multiple codec codebooks, and later codebooks are conditionally dependent on
earlier codebooks in the same frame. The first codebook carries stochastic
content that cannot be known before sampling the frame, so a parallel head can
learn good marginals while still producing invalid codebook combinations.
"""

import argparse
from pathlib import Path

import torch


def make_split(num_sequences, seq_len, num_codebooks, vocab_size, num_speakers, seed):
    gen = torch.Generator().manual_seed(seed)
    speakers = torch.randint(num_speakers, (num_sequences,), generator=gen)
    tokens = torch.zeros(num_sequences, seq_len, num_codebooks, dtype=torch.long)
    for i in range(num_sequences):
        spk = int(speakers[i])
        carrier = torch.randint(vocab_size, (seq_len,), generator=gen)
        for t in range(seq_len):
            tokens[i, t, 0] = (carrier[t] + 3 * spk + t) % vocab_size
            for k in range(1, num_codebooks):
                # RVQ-like chain: each residual codebook depends on the previous
                # codebook, speaker condition, codebook index, and frame position.
                tokens[i, t, k] = (
                    5 * int(tokens[i, t, k - 1]) + 7 * spk + 11 * k + t
                ) % vocab_size
    return {"tokens": tokens, "speaker": speakers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/downloads/synthetic_codec")
    parser.add_argument("--train", type=int, default=8192)
    parser.add_argument("--val", type=int, default=1024)
    parser.add_argument("--test", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--num-speakers", type=int, default=16)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "seq_len": args.seq_len,
        "num_codebooks": args.num_codebooks,
        "vocab_size": args.vocab_size,
        "num_speakers": args.num_speakers,
        "rule": "c[k]=(5*c[k-1]+7*speaker+11*k+t) mod vocab_size for k>0",
    }
    torch.save(make_split(args.train, args.seq_len, args.num_codebooks, args.vocab_size, args.num_speakers, 1001), out / "train.pt")
    torch.save(make_split(args.val, args.seq_len, args.num_codebooks, args.vocab_size, args.num_speakers, 1002), out / "val.pt")
    torch.save(make_split(args.test, args.seq_len, args.num_codebooks, args.vocab_size, args.num_speakers, 1003), out / "test.pt")
    torch.save(meta, out / "meta.pt")
    print(f"wrote synthetic codec dataset to {out}")


if __name__ == "__main__":
    main()
