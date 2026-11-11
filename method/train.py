#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch
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
    # The reference uses a constant learning rate; `cosine` decays it to zero
    # over the run (a training-schedule variant, same step budget).
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
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
    # The whole token set fits on the device; batches are drawn by reshuffling
    # each epoch and dropping the last partial batch (DataLoader semantics,
    # without worker processes).
    all_tokens = train.tokens.to(device)
    all_speakers = train.speaker.to(device)
    batches_per_epoch = len(train) // args.batch_size
    shuffle_gen = torch.Generator(device="cpu").manual_seed(args.seed)
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
    sched = None
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    order = None
    for step in range(1, args.steps + 1):
        position = (step - 1) % batches_per_epoch
        if position == 0:
            order = torch.randperm(len(train), generator=shuffle_gen).to(device)
        idx = order[position * args.batch_size:(position + 1) * args.batch_size]
        tokens = all_tokens[idx]
        speaker = all_speakers[idx]
        loss = model.loss(tokens, speaker)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
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
            "lr_schedule": args.lr_schedule,
            **meta,
        },
        "history": history,
    }
    torch.save(ckpt, out / "model.pt")
    (out / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
