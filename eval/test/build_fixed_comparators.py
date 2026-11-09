#!/usr/bin/env python3
"""Freeze the comparison rows of the hidden test.

Copies the generated tokens of the reference run's comparison models
(parallel heads at 1x/2x/4x, MaskGIT local transformer at 2x/4x) into
eval/test/fixed/. The hidden test always scores these frozen artifacts for the
comparison rows, whatever files with the same names a method leaves on disk.
Run once when (re)building the gym reference; never part of a method run.
"""
import argparse
from pathlib import Path

import torch

FIXED = Path(__file__).resolve().parent / "fixed"
ROWS = {
    "baseline_parallel_1x": 1,
    "baseline_parallel_2x": 2,
    "baseline_parallel_4x": 4,
    "baseline_maskgit_2x": 2,
    "baseline_maskgit_4x": 4,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="checkpoints/predictions")
    args = parser.parse_args()
    FIXED.mkdir(parents=True, exist_ok=True)
    for name, stack in ROWS.items():
        item = torch.load(Path(args.pred_dir) / f"{name}.pt", map_location="cpu", weights_only=True)
        tokens = item["pred_tokens"].long()
        assert tokens.min() >= 0 and tokens.max() < 256, name
        assert int(item["config"]["stack_factor"]) == stack, name
        torch.save({"pred_tokens": tokens.to(torch.uint8), "config": {"stack_factor": stack}, "split": "test"},
                   FIXED / f"{name}.pt")
        print(f"froze {name} ({tuple(tokens.shape)})")


if __name__ == "__main__":
    main()
