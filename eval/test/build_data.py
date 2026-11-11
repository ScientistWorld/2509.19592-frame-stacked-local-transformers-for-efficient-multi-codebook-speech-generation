#!/usr/bin/env python3
"""Generate the fixed synthetic multi-codebook codec-token dataset (packaging only).

Utterances come from the hidden codec process in eval/test/codec_process.py
(a random nonlinear state-space model quantized by an 8-codebook FSQ codec):
24 frames x 8 codebooks, vocabulary 32, 16 speakers. Only the train and
validation token splits are written. The evaluation split is a list of
conditioning speakers (`test_conditions.pt`): the reference utterances it is
scored against are drawn by the hidden evaluator.
"""

import argparse
from pathlib import Path

import torch

from eval.test import codec_process as process

TRAIN = (32768, 1001)
VAL = (2048, 1002)
NUM_CONDITIONS = 16384


def _compact(split):
    return {"tokens": split["tokens"].to(torch.uint8), "speaker": split["speaker"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_codec")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "seq_len": process.SEQ_LEN,
        "num_codebooks": process.NUM_CODEBOOKS,
        "vocab_size": process.VOCAB,
        "num_speakers": process.NUM_SPEAKERS,
        "num_conditions": NUM_CONDITIONS,
        "fsq_levels": list(process.LEVELS),
    }
    torch.save(_compact(process.make_split(*TRAIN)), out / "train.pt")
    torch.save(_compact(process.make_split(*VAL)), out / "val.pt")
    # Evaluation conditions: a balanced speaker list. No reference tokens.
    torch.save({"speaker": torch.arange(NUM_CONDITIONS) % process.NUM_SPEAKERS}, out / "test_conditions.pt")
    (out / "test.pt").unlink(missing_ok=True)
    torch.save(meta, out / "meta.pt")
    print(f"wrote synthetic codec dataset to {out}")


if __name__ == "__main__":
    main()
