#!/usr/bin/env python3
"""Public input preparation; the fixed train/val token artifacts are tracked.

No target-token generator is exposed to a method. The packaging-only builder
lives under eval/test/. This command only (re)writes the conditioning
speaker IDs of both datasets: the 16 main speakers, and the 64 few-shot
speakers (IDs 192-255) of data/synthetic_codec_fewshot.
"""
import argparse
from pathlib import Path
import torch

CONDITIONS = {'synthetic_codec': lambda: torch.arange(16384) % 16,
              'synthetic_codec_fewshot': lambda: 192 + torch.arange(16384) % 64}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data')
    args = p.parse_args()
    for name, speakers in CONDITIONS.items():
        out = Path(args.root) / name
        for item in ('train.pt', 'val.pt', 'meta.pt'):
            if not (out / item).is_file():
                raise FileNotFoundError(f'{out / item}: restore the tracked training artifact')
        torch.save({'speaker': speakers()}, out / 'test_conditions.pt')
        (out / 'test.pt').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
