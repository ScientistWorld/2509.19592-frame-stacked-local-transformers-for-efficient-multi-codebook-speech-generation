#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from data.codec_dataset import codec_features, dependency_validity


def frechet_distance(x, y):
    x = x.reshape(-1, x.shape[-1]).double()
    y = y.reshape(-1, y.shape[-1]).double()
    mx, my = x.mean(dim=0), y.mean(dim=0)
    xc, yc = x - mx, y - my
    cx = xc.T @ xc / max(x.shape[0] - 1, 1)
    cy = yc.T @ yc / max(y.shape[0] - 1, 1)
    # Stable approximation: diagonal covariance FD. This keeps the evaluator
    # dependency-light while retaining the distribution-matching signal.
    vx = torch.diagonal(cx).clamp_min(1e-8)
    vy = torch.diagonal(cy).clamp_min(1e-8)
    return float(((mx - my) ** 2).sum() + (vx + vy - 2 * torch.sqrt(vx * vy)).sum())


def score_file(path):
    item = torch.load(path, map_location="cpu")
    pred = item["pred_tokens"].long()
    ref = item["ref_tokens"].long()
    speaker = item["speaker"].long()
    cfg = item["config"]
    vocab = int(cfg["vocab_size"])
    acc = float((pred == ref).float().mean())
    validity = dependency_validity(pred, speaker, vocab)
    pred_feat = codec_features(pred, speaker, vocab)
    ref_feat = codec_features(ref, speaker, vocab)
    fd = frechet_distance(pred_feat, ref_feat)
    steps_per_24_frames = 24.0 / float(cfg["stack_factor"])
    return {
        "token_accuracy": acc,
        "dependency_validity": validity,
        "codec_fd": fd,
        "nll": float(item["nll"]),
        "primary_steps_per_24_frames": steps_per_24_frames,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", default="checkpoints/predictions")
    p.add_argument("--out", default="scoring/scores.json")
    args = p.parse_args()
    pred_dir = Path(args.pred_dir)
    results = {}
    for path in sorted(pred_dir.glob("*.pt")):
        method = path.stem
        results[method] = score_file(path)
    output = {
        "experiments": {
            "synthetic_codec_unseen": {
                "results": results,
            }
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
