#!/usr/bin/env python3
"""Packaging-only: freeze public fixtures and both slices' comparison outputs.

For each task (main, fewshot), the public train fixtures hold:
- the canonical visible data (train + validation splits);
- frozen sufficient statistics of the train slice's reference sample
  (per-speaker/global moments of codec-latent frame pairs and joint
  token-pair counts), never its utterances, which would be extra
  true-process training data;
- the sorted 48-bit window keys of a fresh data-process sample with the
  train/validation speaker composition (the train evaluator's memorization
  control; its seed stays here);
- comparison/reference outputs with every test-slice row redacted.
The hidden test's fixtures additionally hold the train-slice reference
utterances in its visible sets; it draws its own reference and control
samples at evaluation time.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval.test import evaluate as test

TRAIN_CONTROL_SEEDS = {'main': 31_415_927, 'fewshot': 27_182_818}
SUFFIX = {'main': '', 'fewshot': '_fewshot'}


def write_train_control(fixed, task):
    keys = test.key_set(test.sample_tokens(task, test.public_speakers(task), TRAIN_CONTROL_SEEDS[task]))
    # 48-bit keys as 6-byte little-endian records
    (fixed / f'control_keys{SUFFIX[task]}.bin').write_bytes(keys.astype('<u8').view(np.uint8).reshape(-1, 8)[:, :6].tobytes())


def public_tokens(task):
    data = Path(test.TASKS[task]['data'])
    tokens = torch.cat([torch.load(data / f'{name}.pt', weights_only=True)['tokens'].to(torch.uint8) for name in ('train', 'val')])
    speakers = torch.cat([torch.load(data / f'{name}.pt', weights_only=True)['speaker'].long() for name in ('train', 'val')])
    if not torch.equal(speakers, test.public_speakers(task)):
        raise RuntimeError(f'{data}: speakers differ from the canonical splits')
    return tokens


def main():
    torch.set_num_threads(1)  # bit-reproducible float64 accumulation
    hashes = {f"{test.TASKS[task]['data']}/{name}.pt":
              hashlib.sha256(Path(f"{test.TASKS[task]['data']}/{name}.pt").read_bytes()).hexdigest()
              for task in test.TASKS for name in ('train', 'val')}
    for task in test.TASKS:
        train_rows = test.slice_rows(task, 'train')
        test_rows = test.slice_rows(task, 'test')
        speakers = test.conditions(task)[train_rows].repeat_interleave(test.REFERENCES_PER_ROW)
        train_ref = test.sample_tokens(task, speakers, test.TASKS[task]['train_slice_reference_seed'])
        public = public_tokens(task)
        for split in ('train', 'test'):
            fixed = Path(f'eval/{split}/fixed')
            fixed.mkdir(exist_ok=True, parents=True)
            visible = public if split == 'train' else torch.cat([public, train_ref.to(torch.uint8)])
            torch.save({'tokens': visible}, fixed / test.TASKS[task]['visible'])
            (fixed / 'data_hashes.json').write_text(json.dumps(hashes, indent=2))
            if split == 'train':
                groups = sorted(set(test.conditions(task)[train_rows].tolist()))
                means, cov = test.feature_moments(train_ref, speakers, groups)
                torch.save({'means': means, 'cov': cov, 'groups': groups, 'num_utterances': int(train_ref.shape[0]),
                            'references_per_row': test.REFERENCES_PER_ROW,
                            'note': 'float64 sufficient statistics of 48-d consecutive-frame FSQ codec latents, '
                                    'one per speaker in `groups`, then all rows'},
                           fixed / f'reference_moments{SUFFIX[task]}.pt')
                torch.save({'counts': test.pair_counts(train_ref),
                            'note': '28 within-frame codebook pairs then 64 (frame t, frame t+1) codebook pairs; 32x32 cells each'},
                           fixed / f'reference_pairs{SUFFIX[task]}.pt')
                for stale in ('reference.pt', 'control_windows.bin'):
                    (fixed / stale).unlink(missing_ok=True)
                write_train_control(fixed, task)
            redact = test_rows if split == 'train' else None
            rows = {n: s for n, (t, s) in test.FIXED_ROWS.items() if t == task}
            for name, stack in rows.items():
                item = torch.load(Path('checkpoints/predictions') / f'{name}.pt', map_location='cpu', weights_only=True)
                tokens = item['pred_tokens'].to(torch.uint8).clone()
                if redact is not None:
                    tokens[redact] = 0  # no test-side model outputs in public fixtures
                torch.save({'pred_tokens': tokens, 'config': {'stack_factor': stack}}, fixed / f'{name}.pt')
            refs = fixed / 'reference_predictions'
            refs.mkdir(exist_ok=True)
            for name, (t, stack) in test.REQUIRED_ROWS.items():
                if t != task:
                    continue
                item = torch.load(Path('checkpoints/predictions') / f'{name}.pt', map_location='cpu', weights_only=True)
                tokens = item['pred_tokens'].to(torch.uint8).clone()
                if redact is not None:
                    tokens[redact] = 0
                torch.save({'pred_tokens': tokens, 'config': {'stack_factor': stack}}, refs / f'{name}.pt')


if __name__ == '__main__':
    main()
