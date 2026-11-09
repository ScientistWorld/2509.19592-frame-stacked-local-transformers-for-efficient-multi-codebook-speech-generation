#!/usr/bin/env python3
"""Better-variant check (structural): the primary decoder reads the previous
coarse step's slot embeddings concatenated and linearly projected, instead of
the paper's average over the whole frame stack and all codebooks.

Usage: python redteam/hardening/concat_input.py method.train|method.predict [args]
Everything else (local transformer, training recipe, sampling) is the
reference method's.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import method.models as models  # noqa: E402


class ConcatInputModel(models.FrameStackedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_in = nn.Linear(self.tokens_per_step * self.dim, self.dim)

    def _main_hidden(self, packed_tokens, speaker):
        b, steps, slots = packed_tokens.shape
        bos = torch.full((b, 1, slots), self.mask_id, device=packed_tokens.device, dtype=torch.long)
        prev = torch.cat([bos, packed_tokens[:, :-1]], dim=1)
        x = self.concat_in(self._embed(prev).reshape(b, steps, slots * self.dim))
        pos = torch.arange(steps, device=packed_tokens.device)
        x = self.step_in(x + self.speaker_embed(speaker).unsqueeze(1) + self.pos_embed(pos).unsqueeze(0))
        return self.main(x)


models.FrameStackedModel = ConcatInputModel

if __name__ == "__main__":
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
