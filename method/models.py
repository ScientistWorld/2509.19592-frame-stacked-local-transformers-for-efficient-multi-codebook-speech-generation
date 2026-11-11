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
    """Primary decoder over stacked frames plus an optional local transformer.

    Following the paper, every (frame-in-stack, codebook) slot has its own
    token embedding table; the tables are shared by the primary decoder and the
    local transformer, and the primary decoder input averages them across the
    whole stack.
    """

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
        speaker_rank=0,
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
        self.mask_id = vocab_size  # also used as BOS / "not yet generated"

        slots = self.tokens_per_step
        # Slot-specific embedding tables, parametrized as a table shared by all
        # slots plus a per-slot table (zero-initialized), so the common token
        # geometry is learned once and slot-specific structure on top of it.
        self.token_embed = nn.Embedding(vocab_size + 1, dim)
        self.slot_token_embed = nn.Embedding(slots * (vocab_size + 1), dim)
        nn.init.zeros_(self.slot_token_embed.weight)
        self.register_buffer("slot_offset", torch.arange(slots) * (vocab_size + 1), persistent=False)
        if speaker_rank > 0:
            # Optional low-rank speaker table (the reference uses a full one):
            # every speaker is a point of an r-dimensional space shared by all
            # speakers, so speakers with little data have fewer free parameters.
            self.speaker_embed = nn.Sequential(nn.Embedding(num_speakers, speaker_rank),
                                               nn.Linear(speaker_rank, dim, bias=False))
        else:
            self.speaker_embed = nn.Embedding(num_speakers, dim)
        self.pos_embed = nn.Embedding(256, dim)
        self.step_in = nn.Linear(dim, dim)
        self.main = CausalBlock(dim, heads, main_layers)

        if variant == "none":
            self.parallel_heads = nn.ModuleList([nn.Linear(dim, vocab_size) for _ in range(slots)])
        else:
            self.slot_embed = nn.Embedding(slots + 1, dim)
            self.lt = CausalBlock(dim, heads, lt_layers) if variant == "ar" else self._noncausal_lt(dim, heads, lt_layers)
            self.lt_head = nn.Linear(dim, vocab_size)

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

    def _embed(self, slot_tokens):
        """Embed (..., slots) tokens with their slot-specific tables."""
        return self.token_embed(slot_tokens) + self.slot_token_embed(slot_tokens + self.slot_offset)

    def _pack(self, tokens):
        b, t, n = tokens.shape
        s = self.stack_factor
        assert t % s == 0, "sequence length must be divisible by stack factor"
        return tokens.reshape(b, t // s, s * n)

    def _main_hidden(self, packed_tokens, speaker):
        b, steps, slots = packed_tokens.shape
        bos = torch.full((b, 1, slots), self.mask_id, device=packed_tokens.device, dtype=torch.long)
        prev = torch.cat([bos, packed_tokens[:, :-1]], dim=1)
        x = self._embed(prev).mean(dim=2)
        pos = torch.arange(steps, device=packed_tokens.device)
        x = self.step_in(x + self.speaker_embed(speaker).unsqueeze(1) + self.pos_embed(pos).unsqueeze(0))
        return self.main(x)

    def _lt_input(self, h, slot_tokens):
        """Local-transformer sequence: [hidden state, slot 0, ..., slot S*N-1]."""
        slots = slot_tokens.shape[-1]
        ids = torch.arange(slots, device=slot_tokens.device)
        x = self._embed(slot_tokens) + self.slot_embed(ids)
        start = h + self.slot_embed.weight[slots]
        return torch.cat([start.unsqueeze(-2), x], dim=-2)

    def loss(self, tokens, speaker):
        packed = self._pack(tokens)
        h = self._main_hidden(packed, speaker)
        b, steps, slots = packed.shape
        if self.variant == "none":
            losses = [F.cross_entropy(head(h).reshape(-1, self.vocab_size), packed[..., i].reshape(-1)) for i, head in enumerate(self.parallel_heads)]
            return torch.stack(losses).mean()
        if self.variant == "ar":
            # Teacher forcing: position i+1 holds slot i; the output at position
            # i (hidden state or slot i-1) predicts slot i.
            x = self._lt_input(h, packed)[..., :-1, :]
            y = self.lt(x.reshape(b * steps, slots, self.dim))
            logits = self.lt_head(y).reshape(b, steps, slots, self.vocab_size)
            return F.cross_entropy(logits.reshape(-1, self.vocab_size), packed.reshape(-1))
        if self.variant == "maskgit":
            # MaskGIT training: cosine-schedule masking ratio, loss on masked slots.
            u = torch.rand(b, steps, 1, device=tokens.device)
            n_mask = torch.ceil(torch.cos(u * math.pi / 2) * slots).clamp(1, slots)
            rank = torch.rand(b, steps, slots, device=tokens.device).argsort(dim=-1).argsort(dim=-1)
            mask = rank < n_mask
            masked = packed.masked_fill(mask, self.mask_id)
            x = self._lt_input(h, masked)
            y = self.lt(x.reshape(b * steps, slots + 1, self.dim))[:, 1:]
            logits = self.lt_head(y).reshape(b, steps, slots, self.vocab_size)
            return F.cross_entropy(logits[mask], packed[mask])
        raise ValueError(self.variant)

    @torch.no_grad()
    def generate(self, speaker, num_frames, temperature=1.0):
        """Free-running generation of `num_frames` frames per speaker.

        Each primary step conditions only on the model's own previously
        generated stacked frames; the local transformer (or parallel heads)
        then samples the stacked frame's codebooks.
        """
        self.eval()
        b = speaker.shape[0]
        steps = num_frames // self.stack_factor
        slots = self.tokens_per_step
        out = torch.full((b, steps, slots), self.mask_id, device=speaker.device, dtype=torch.long)
        for step_idx in range(steps):
            h_step = self._main_hidden(out[:, : step_idx + 1], speaker)[:, step_idx]
            out[:, step_idx] = self._generate_slots(h_step, temperature)
        return out.reshape(b, num_frames, self.num_codebooks)

    def _sample(self, logits, temperature):
        if temperature <= 0:
            return logits.argmax(dim=-1)
        probs = (logits / temperature).softmax(dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        return torch.multinomial(flat, 1).reshape(probs.shape[:-1])

    def _generate_slots(self, h_step, temperature):
        b = h_step.shape[0]
        slots = self.tokens_per_step
        if self.variant == "none":
            logits = torch.stack([head(h_step) for head in self.parallel_heads], dim=1)
            return self._sample(logits, temperature)
        out = torch.full((b, slots), self.mask_id, device=h_step.device, dtype=torch.long)
        if self.variant == "ar":
            for i in range(slots):
                x = self._lt_input(h_step, out)[:, : i + 1]
                y = self.lt(x)[:, i]
                out[:, i] = self._sample(self.lt_head(y), temperature)
            return out
        if self.variant == "maskgit":
            # Iterative unmasking with purity sampling: at each of the P steps,
            # the most confident masked slots are sampled and fixed, following a
            # cosine schedule of how many slots stay masked.
            for step in range(self.maskgit_steps):
                remaining = out.eq(self.mask_id)
                logits = self.lt_head(self.lt(self._lt_input(h_step, out))[:, 1:])
                purity = logits.softmax(dim=-1).max(dim=-1).values.masked_fill(~remaining, -1.0)
                keep_masked = int(math.floor(slots * math.cos(math.pi / 2 * (step + 1) / self.maskgit_steps)))
                n_fill = int(remaining[0].sum()) - keep_masked
                top = purity.topk(n_fill, dim=-1).indices
                choose = torch.zeros_like(remaining).scatter_(-1, top, True) & remaining
                sampled = self._sample(logits, temperature)
                out[choose] = sampled[choose]
            return out
        raise ValueError(self.variant)
