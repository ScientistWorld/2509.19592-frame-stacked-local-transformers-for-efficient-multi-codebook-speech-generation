#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.codec_dataset import CodecTokenDataset
from method.models import FrameStackedModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/downloads/synthetic_codec")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = FrameStackedModel(
        vocab_size=cfg["vocab_size"],
        num_codebooks=cfg["num_codebooks"],
        num_speakers=cfg["num_speakers"],
        stack_factor=cfg["stack_factor"],
        variant=cfg["variant"],
        dim=cfg["dim"],
        main_layers=cfg["main_layers"],
        lt_layers=cfg["lt_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    ds = CodecTokenDataset(args.data, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    preds, refs, speakers = [], [], []
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        speaker = batch["speaker"].to(device)
        with torch.no_grad():
            loss = model.loss(tokens, speaker)
            pred = model.generate(tokens, speaker)
        preds.append(pred.cpu())
        refs.append(tokens.cpu())
        speakers.append(speaker.cpu())
        total_loss += float(loss.cpu()) * tokens.shape[0]
        total_items += tokens.shape[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred_tokens": torch.cat(preds),
            "ref_tokens": torch.cat(refs),
            "speaker": torch.cat(speakers),
            "nll": total_loss / total_items,
            "config": cfg,
        },
        out,
    )
    print(f"wrote predictions to {out}")


if __name__ == "__main__":
    main()
