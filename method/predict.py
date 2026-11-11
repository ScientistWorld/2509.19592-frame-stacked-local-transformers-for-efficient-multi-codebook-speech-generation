#!/usr/bin/env python3
"""Generate evaluation predictions for the conditioning speakers."""
import argparse
from pathlib import Path

import torch

from data.codec_dataset import load_conditions
from method.models import FrameStackedModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_codec")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    # Ancestral sampling at temperature 1: the scored quantity is how well the
    # generated distribution matches real data (Koel-TTS's speech-tuned 0.6 trades
    # diversity for validity here; see HARDENING.md).
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
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
        heads=cfg.get("heads", 4),
        main_layers=cfg["main_layers"],
        lt_layers=cfg["lt_layers"],
        maskgit_steps=cfg.get("maskgit_steps", 3),
        speaker_rank=cfg.get("speaker_rank", 0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    speakers = load_conditions(args.data)
    preds = []
    for start in range(0, speakers.shape[0], args.batch_size):
        spk = speakers[start : start + args.batch_size].to(device)
        preds.append(model.generate(spk, cfg["seq_len"], temperature=args.temperature).cpu())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred_tokens": torch.cat(preds).to(torch.long),
            "config": {**cfg, "temperature": args.temperature, "sample_seed": args.seed},
            "split": "test",
        },
        out,
    )
    print(f"wrote predictions to {out}")


if __name__ == "__main__":
    main()
