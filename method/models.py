import math

import torch
from torch import nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    def __init__(self, dim, heads, layers, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        t = x.shape[1]
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        return self.net(x, mask=mask)


class FrameStackedModel(nn.Module):
    def __init__(
        self,
        vocab_size=32,
        num_codebooks=8,
        num_speakers=16,
        stack_factor=1,
        variant="none",
        dim=96,
        heads=4,
        main_layers=2,
        lt_layers=1,
        maskgit_steps=3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.num_speakers = num_speakers
        self.stack_factor = stack_factor
        self.variant = variant
        self.dim = dim
        self.maskgit_steps = maskgit_steps
        self.tokens_per_step = stack_factor * num_codebooks

        self.token_embed = nn.Embedding(vocab_size + 1, dim)
        self.speaker_embed = nn.Embedding(num_speakers, dim)
        self.pos_embed = nn.Embedding(256, dim)
        self.step_in = nn.Linear(dim, dim)
        self.main = CausalBlock(dim, heads, main_layers)

        if variant == "none":
            self.parallel_heads = nn.ModuleList([nn.Linear(dim, vocab_size) for _ in range(self.tokens_per_step)])
        else:
            self.slot_embed = nn.Embedding(self.tokens_per_step + 1, dim)
            self.lt = CausalBlock(dim, heads, lt_layers) if variant == "ar" else self._noncausal_lt(dim, heads, lt_layers)
            self.lt_head = nn.Linear(dim, vocab_size)
            self.mask_id = vocab_size

    def _noncausal_lt(self, dim, heads, layers):
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=4 * dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, layers)

    def _pack(self, tokens):
        b, t, n = tokens.shape
        s = self.stack_factor
        assert t % s == 0, "sequence length must be divisible by stack factor"
        return tokens.reshape(b, t // s, s * n)

    def _main_hidden(self, packed_tokens, speaker):
        b, steps, slots = packed_tokens.shape
        bos = torch.full((b, 1, slots), self.vocab_size, device=packed_tokens.device, dtype=torch.long)
        prev = torch.cat([bos, packed_tokens[:, :-1]], dim=1)
        x = self.token_embed(prev).mean(dim=2)
        pos = torch.arange(steps, device=packed_tokens.device)
        x = self.step_in(x + self.speaker_embed(speaker).unsqueeze(1) + self.pos_embed(pos).unsqueeze(0))
        return self.main(x)

    def loss(self, tokens, speaker):
        packed = self._pack(tokens)
        h = self._main_hidden(packed, speaker)
        if self.variant == "none":
            losses = [F.cross_entropy(head(h).reshape(-1, self.vocab_size), packed[..., i].reshape(-1)) for i, head in enumerate(self.parallel_heads)]
            return torch.stack(losses).mean()
        if self.variant == "ar":
            b, steps, slots = packed.shape
            bos = torch.full((b, steps, 1), self.vocab_size, device=tokens.device, dtype=torch.long)
            prev = torch.cat([bos, packed[..., :-1]], dim=-1)
            x = self.token_embed(prev) + self.slot_embed(torch.arange(slots, device=tokens.device)).view(1, 1, slots, -1)
            x = torch.cat([h.unsqueeze(2) + self.slot_embed.weight[slots].view(1, 1, 1, -1), x], dim=2)
            y = self.lt(x.reshape(b * steps, slots + 1, self.dim))[:, 1:]
            logits = self.lt_head(y).reshape(b, steps, slots, self.vocab_size)
            return F.cross_entropy(logits.reshape(-1, self.vocab_size), packed.reshape(-1))
        if self.variant == "maskgit":
            b, steps, slots = packed.shape
            noise = torch.rand(b, steps, slots, device=tokens.device)
            masked = packed.masked_fill(noise < 0.55, self.mask_id)
            x = self.token_embed(masked) + self.slot_embed(torch.arange(slots, device=tokens.device)).view(1, 1, slots, -1)
            x = torch.cat([h.unsqueeze(2) + self.slot_embed.weight[slots].view(1, 1, 1, -1), x], dim=2)
            y = self.lt(x.reshape(b * steps, slots + 1, self.dim))[:, 1:]
            logits = self.lt_head(y).reshape(b, steps, slots, self.vocab_size)
            return F.cross_entropy(logits.reshape(-1, self.vocab_size), packed.reshape(-1))
        raise ValueError(self.variant)

    @torch.no_grad()
    def generate(self, prompt_tokens, speaker):
        self.eval()
        packed_prompt = self._pack(prompt_tokens)
        h = self._main_hidden(packed_prompt, speaker)
        b, steps, slots = packed_prompt.shape
        if self.variant == "none":
            preds = [head(h).argmax(dim=-1) for head in self.parallel_heads]
            out = torch.stack(preds, dim=-1)
        elif self.variant == "ar":
            out = torch.full((b, steps, slots), self.vocab_size, device=prompt_tokens.device, dtype=torch.long)
            for i in range(slots):
                prev = torch.full((b, steps, slots), self.vocab_size, device=prompt_tokens.device, dtype=torch.long)
                if i > 0:
                    # Training predicts slot i from a shifted sequence:
                    # [BOS, token_0, ..., token_{i-1}]. Keep the same
                    # alignment at sampling time.
                    prev[..., 1 : i + 1] = out[..., :i]
                x = self.token_embed(prev) + self.slot_embed(torch.arange(slots, device=prompt_tokens.device)).view(1, 1, slots, -1)
                x = torch.cat([h.unsqueeze(2) + self.slot_embed.weight[slots].view(1, 1, 1, -1), x], dim=2)
                y = self.lt(x.reshape(b * steps, slots + 1, self.dim))[:, 1 + i]
                out[..., i] = self.lt_head(y).reshape(b, steps, self.vocab_size).argmax(dim=-1)
        elif self.variant == "maskgit":
            out = torch.full((b, steps, slots), self.vocab_size, device=prompt_tokens.device, dtype=torch.long)
            remaining = torch.ones_like(out, dtype=torch.bool)
            for step in range(self.maskgit_steps):
                x = self.token_embed(out) + self.slot_embed(torch.arange(slots, device=prompt_tokens.device)).view(1, 1, slots, -1)
                x = torch.cat([h.unsqueeze(2) + self.slot_embed.weight[slots].view(1, 1, 1, -1), x], dim=2)
                y = self.lt(x.reshape(b * steps, slots + 1, self.dim))[:, 1:]
                logits = self.lt_head(y).reshape(b, steps, slots, self.vocab_size)
                probs = logits.softmax(dim=-1)
                conf, pred = probs.max(dim=-1)
                quota = math.ceil(slots * (step + 1) / self.maskgit_steps)
                top = conf.topk(quota, dim=-1).indices
                new_mask = torch.zeros_like(remaining)
                new_mask.scatter_(-1, top, True)
                fill = new_mask & remaining
                out[fill] = pred[fill]
                remaining = out.eq(self.mask_id)
            out[out.eq(self.mask_id)] = pred[out.eq(self.mask_id)]
        else:
            raise ValueError(self.variant)
        return out.reshape(prompt_tokens.shape[0], prompt_tokens.shape[1], self.num_codebooks)
