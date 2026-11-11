#!/usr/bin/env python3
"""Packaging-only: freeze public fixtures and both slices' comparison outputs.

Public train fixtures hold the canonical visible data (train + validation
splits); frozen sufficient statistics of the train slice's reference sample
(per-speaker/global moments of codec-latent frame pairs and joint token-pair
counts, never its utterances, which would be extra true-process training
data); the sorted 48-bit window keys of a fresh data-process sample with the
train/validation speaker composition (the train evaluator's memorization
control; its seed stays here); and comparison/reference outputs with every
test-slice row redacted. The hidden test's fixtures additionally hold the
train-slice reference utterances in its visible set; it draws its own
reference and control samples at evaluation time.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval.test import codec_process as process
from eval.test import evaluate as test

TRAIN_CONTROL_SEED = 31_415_927


def write_train_control(fixed):
    speakers = torch.cat([process.make_split(n, seed)['speaker'] for n, seed in test.VISIBLE_SPLITS])
    keys = test.key_set(process.sample_tokens(speakers, TRAIN_CONTROL_SEED))
    # 48-bit keys as 6-byte little-endian records
    (fixed / 'control_keys.bin').write_bytes(keys.astype('<u8').view(np.uint8).reshape(-1, 8)[:, :6].tobytes())


def main():
    torch.set_num_threads(1)  # bit-reproducible float64 accumulation
    order = torch.randperm(test.NUM_CONDITIONS, generator=torch.Generator().manual_seed(test.PARTITION_SEED))
    rows = order[:test.NUM_CONDITIONS // 2].sort().values
    speakers = test.conditions()[rows].repeat_interleave(test.REFERENCES_PER_ROW)
    train_ref = process.sample_tokens(speakers, test.TRAIN_SLICE_REFERENCE_SEED)
    public_tokens = torch.cat([process.make_split(n, seed)['tokens'].to(torch.uint8) for n, seed in test.VISIBLE_SPLITS])
    for split in ('train', 'test'):
        fixed = Path(f'eval/{split}/fixed')
        fixed.mkdir(exist_ok=True, parents=True)
        visible = public_tokens if split == 'train' else torch.cat([public_tokens, train_ref.to(torch.uint8)])
        torch.save({'tokens': visible}, fixed / 'visible.pt')
        hashes = {f'data/synthetic_codec/{name}.pt': hashlib.sha256(Path(f'data/synthetic_codec/{name}.pt').read_bytes()).hexdigest()
                  for name in ('train', 'val')}
        (fixed / 'data_hashes.json').write_text(json.dumps(hashes, indent=2))
        if split == 'train':
            means, cov = test.feature_moments(train_ref, speakers)
            torch.save({'means': means, 'cov': cov, 'num_utterances': int(train_ref.shape[0]),
                        'references_per_row': test.REFERENCES_PER_ROW,
                        'note': 'float64 sufficient statistics of 48-d consecutive-frame FSQ codec latents; index 16 is global'},
                       fixed / 'reference_moments.pt')
            torch.save({'counts': test.pair_counts(train_ref),
                        'note': '28 within-frame codebook pairs then 64 (frame t, frame t+1) codebook pairs; 32x32 cells each'},
                       fixed / 'reference_pairs.pt')
            for stale in ('reference.pt', 'control_windows.bin'):
                (fixed / stale).unlink(missing_ok=True)
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
