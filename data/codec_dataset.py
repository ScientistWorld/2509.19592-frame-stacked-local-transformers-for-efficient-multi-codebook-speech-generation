from pathlib import Path

import torch
from torch.utils.data import Dataset


class CodecTokenDataset(Dataset):
    """Train/validation token sequences with their speaker ids."""

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


def load_conditions(root):
    """Conditioning speakers for the evaluation predictions (no reference tokens)."""
    return torch.load(Path(root) / "test_conditions.pt", map_location="cpu")["speaker"].long()
