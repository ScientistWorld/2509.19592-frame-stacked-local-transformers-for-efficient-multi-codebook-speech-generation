#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.codec_dataset import CodecTokenDataset
from method.models import FrameStackedModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_codec")
    p.add_argument("--out", required=True)
    p.add_argument("--variant", choices=["none", "ar", "maskgit"], required=True)
    p.add_argument("--stack-factor", type=int, default=1)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--main-layers", type=int, default=2)
    p.add_argument("--lt-layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--maskgit-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = CodecTokenDataset(args.data, "train")
    meta = train.meta
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    model = FrameStackedModel(
        vocab_size=meta["vocab_size"],
        num_codebooks=meta["num_codebooks"],
        num_speakers=meta["num_speakers"],
        stack_factor=args.stack_factor,
        variant=args.variant,
        dim=args.dim,
        heads=args.heads,
        main_layers=args.main_layers,
        lt_layers=args.lt_layers,
        maskgit_steps=args.maskgit_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        tokens = batch["tokens"].to(device)
        speaker = batch["speaker"].to(device)
        loss = model.loss(tokens, speaker)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
            print(f"step={step} loss={history[-1]['loss']:.4f}", flush=True)
    ckpt = {
        "model": model.state_dict(),
        "config": {
            "variant": args.variant,
            "stack_factor": args.stack_factor,
            "dim": args.dim,
            "heads": args.heads,
            "main_layers": args.main_layers,
            "lt_layers": args.lt_layers,
            "maskgit_steps": args.maskgit_steps,
            "seed": args.seed,
            "train_steps": args.steps,
            "lr": args.lr,
            **meta,
        },
        "history": history,
    }
    torch.save(ckpt, out / "model.pt")
    (out / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
