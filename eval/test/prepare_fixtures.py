#!/usr/bin/env python3
"""Packaging-only: freeze public fixtures and both slices' comparison outputs."""
from pathlib import Path
import shutil
import hashlib
import json
import torch
from eval.test import evaluate as test
from eval.test import build_data as data


def main():
    rows = torch.randperm(test.NUM_CONDITIONS, generator=torch.Generator().manual_seed(test.PARTITION_SEED))[:test.NUM_CONDITIONS // 2].sort().values
    speakers = test.conditions()[rows].repeat_interleave(test.REFERENCES_PER_ROW)
    public_ref = test.sample_tokens(speakers, test.TRAIN_SLICE_REFERENCE_SEED).to(torch.uint8)
    public_tokens = torch.cat([data.make_split(n, 24, 8, 32, 16, seed)['tokens'].to(torch.uint8) for n, seed in test.VISIBLE_SPLITS] + [public_ref])
    for split in ('train', 'test'):
        fixed = Path(f'eval/{split}/fixed')
        fixed.mkdir(exist_ok=True, parents=True)
        torch.save({'tokens': public_tokens}, fixed / 'visible.pt')
        hashes = {f'data/synthetic_codec/{name}.pt': hashlib.sha256(Path(f'data/synthetic_codec/{name}.pt').read_bytes()).hexdigest() for name in ('train', 'val')}
        (fixed / 'data_hashes.json').write_text(json.dumps(hashes, indent=2))
        if split == 'train':
            torch.save({'tokens': public_ref, 'speaker': speakers}, fixed / 'reference.pt')
        for name in test.FIXED_ROWS:
            item = torch.load(Path('checkpoints/predictions') / f'{name}.pt', map_location='cpu', weights_only=True)
            tokens = item['pred_tokens'].to(torch.uint8).clone()
            if split == 'train':
                tokens[test.slice_rows()] = 0  # no test-side model outputs in public fixtures
            torch.save({'pred_tokens': tokens, 'config': {'stack_factor': test.FIXED_ROWS[name]}}, fixed / f'{name}.pt')
        refs = fixed / 'reference_predictions'
        refs.mkdir(exist_ok=True)
        for name in test.REQUIRED_ROWS:
            item = torch.load(Path('checkpoints/predictions') / f'{name}.pt', map_location='cpu', weights_only=True)
            tokens = item['pred_tokens'].to(torch.uint8).clone()
            if split == 'train':
                tokens[test.slice_rows()] = 0
            torch.save({'pred_tokens': tokens, 'config': {'stack_factor': test.REQUIRED_ROWS[name]}}, refs / f'{name}.pt')

if __name__ == '__main__':
    main()
