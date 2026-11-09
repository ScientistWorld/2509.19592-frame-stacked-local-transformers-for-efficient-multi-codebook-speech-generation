#!/usr/bin/env python3
"""Public input preparation; the fixed train/val token artifacts are tracked.

No target-token generator is exposed to a method. The packaging-only builder
lives under eval/test/. This command only writes conditioning speaker IDs.
"""
import argparse
from pathlib import Path
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/synthetic_codec')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('train.pt', 'val.pt', 'meta.pt'):
        if not (out / name).is_file():
            raise FileNotFoundError(f'{out / name}: restore the tracked training artifact')
    torch.save({'speaker': torch.arange(16384) % 16}, out / 'test_conditions.pt')
    meta = torch.load(out / 'meta.pt', weights_only=True)
    meta['num_conditions'] = 16384
    torch.save(meta, out / 'meta.pt')
    (out / 'test.pt').unlink(missing_ok=True)

if __name__ == '__main__':
    main()
