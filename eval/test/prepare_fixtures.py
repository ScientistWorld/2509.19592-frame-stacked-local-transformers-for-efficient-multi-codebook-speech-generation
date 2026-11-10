#!/usr/bin/env python3
"""Packaging-only: freeze public fixtures and both slices' comparison outputs.

Public train fixtures hold the canonical visible data (train + validation
splits), frozen sufficient statistics of the train slice's reference sample
(never its utterances, which would be extra true-process training data), and
comparison/reference outputs with every test-slice row redacted, and 64-bit
hashes of the 4-frame windows of a fresh data-process sample with the
train/validation speaker composition (the train evaluator's memorization
control; its seed stays here). The hidden test's fixtures additionally hold the
train-slice reference utterances in its copy-check set; it draws its own
control sample at evaluation time.
"""
from array import array
from pathlib import Path
import hashlib
import json
import sys
import torch
from eval.test import evaluate as test
from eval.test import build_data as data


TRAIN_CONTROL_SEED = 31_415_927
EXCESS_WINDOW = 4


def write_train_control(fixed):
    speakers = torch.cat([data.make_split(n, 24, 8, 32, 16, seed)['speaker'] for n, seed in test.VISIBLE_SPLITS])
    control = test.sample_tokens(speakers, TRAIN_CONTROL_SEED).to(torch.uint8).contiguous().numpy()
    keys = {bytes([s]) + row[s:s + EXCESS_WINDOW].tobytes() for row in control for s in range(24 - EXCESS_WINDOW + 1)}
    hashes = array('q', sorted(int.from_bytes(hashlib.blake2b(k, digest_size=8).digest(), 'little', signed=True) for k in keys))
    if sys.byteorder != 'little':
        hashes.byteswap()
    (fixed / 'control_windows.bin').write_bytes(hashes.tobytes())


def main():
    rows = torch.randperm(test.NUM_CONDITIONS, generator=torch.Generator().manual_seed(test.PARTITION_SEED))[:test.NUM_CONDITIONS // 2].sort().values
    speakers = test.conditions()[rows].repeat_interleave(test.REFERENCES_PER_ROW)
    train_ref = test.sample_tokens(speakers, test.TRAIN_SLICE_REFERENCE_SEED)
    public_tokens = torch.cat([data.make_split(n, 24, 8, 32, 16, seed)['tokens'].to(torch.uint8) for n, seed in test.VISIBLE_SPLITS])
    for split in ('train', 'test'):
        fixed = Path(f'eval/{split}/fixed')
        fixed.mkdir(exist_ok=True, parents=True)
        visible = public_tokens if split == 'train' else torch.cat([public_tokens, train_ref.to(torch.uint8)])
        torch.save({'tokens': visible}, fixed / 'visible.pt')
        hashes = {f'data/synthetic_codec/{name}.pt': hashlib.sha256(Path(f'data/synthetic_codec/{name}.pt').read_bytes()).hexdigest() for name in ('train', 'val')}
        (fixed / 'data_hashes.json').write_text(json.dumps(hashes, indent=2))
        if split == 'train':
            means, cov = test.feature_moments(train_ref, speakers)
            torch.save({'means': means, 'cov': cov, 'num_utterances': int(train_ref.shape[0]),
                        'references_per_row': test.REFERENCES_PER_ROW,
                        'note': 'float64 sufficient statistics of 64-d consecutive-frame codec latents; index 16 is global'},
                       fixed / 'reference_moments.pt')
            (fixed / 'reference.pt').unlink(missing_ok=True)
            write_train_control(fixed)
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
