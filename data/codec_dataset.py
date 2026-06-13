from pathlib import Path

import torch
from torch.utils.data import Dataset


class CodecTokenDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        item = torch.load(self.root / f"{split}.pt", map_location="cpu")
        self.tokens = item["tokens"].long()
        self.speaker = item["speaker"].long()
        self.meta = torch.load(self.root / "meta.pt", map_location="cpu")

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return {"tokens": self.tokens[idx], "speaker": self.speaker[idx]}


def dependency_validity(tokens, speaker, vocab_size):
    """Fraction of k>0 codebook entries satisfying the synthetic codec rule."""
    expected = tokens[..., 1:].clone()
    seq_len = tokens.shape[1]
    positions = torch.arange(seq_len, device=tokens.device).view(1, seq_len, 1)
    speakers = speaker.to(tokens.device).view(-1, 1, 1)
    k_idx = torch.arange(1, tokens.shape[2], device=tokens.device).view(1, 1, -1)
    prev = tokens[..., :-1]
    expected = (5 * prev + 7 * speakers + 11 * k_idx + positions) % vocab_size
    return (tokens[..., 1:] == expected).float().mean().item()


def codec_features(tokens, speaker, vocab_size, dim=32):
    """Deterministic codec-frame embedding used for FD evaluation."""
    device = tokens.device
    k = torch.arange(tokens.shape[-1], device=device).float().view(1, 1, -1, 1)
    d = torch.arange(dim, device=device).float().view(1, 1, 1, -1)
    x = tokens.float().unsqueeze(-1)
    sp = speaker.float().view(-1, 1, 1, 1)
    emb = torch.sin((x + 1) * (d + 1) * 0.071 + (k + 1) * 0.37)
    emb = emb + torch.cos((x + sp + 1) * (d + 1) * 0.043)
    return emb.mean(dim=2)
