#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from data.codec_dataset import CodecTokenDataset, codec_features, dependency_validity


EXPERIMENT_NAME = "synthetic_codec_unseen"
SLICE = "test"
PARTITION_SEED = 250919592


def frechet_distance(x, y):
    x = x.reshape(-1, x.shape[-1]).double()
    y = y.reshape(-1, y.shape[-1]).double()
    mx, my = x.mean(dim=0), y.mean(dim=0)
    xc, yc = x - mx, y - my
    cx = xc.T @ xc / max(x.shape[0] - 1, 1)
    cy = yc.T @ yc / max(y.shape[0] - 1, 1)
    vx = torch.diagonal(cx).clamp_min(1e-8)
    vy = torch.diagonal(cy).clamp_min(1e-8)
    return float(((mx - my) ** 2).sum() + (vx + vy - 2 * torch.sqrt(vx * vy)).sum())


def speaker_conditioned_fd(pred, ref, speaker, vocab):
    distances = []
    for speaker_id in torch.unique(speaker).sort().values:
        mask = speaker == speaker_id
        if int(mask.sum()) < 2:
            continue
        pred_feat = codec_features(pred[mask], speaker[mask], vocab)
        ref_feat = codec_features(ref[mask], speaker[mask], vocab)
        distances.append(frechet_distance(pred_feat, ref_feat))
    if not distances:
        return float("nan")
    return float(sum(distances) / len(distances))


def split_indices(num_items):
    generator = torch.Generator().manual_seed(PARTITION_SEED)
    order = torch.randperm(num_items, generator=generator)
    midpoint = num_items // 2
    return order[midpoint:].sort().values


def discover_methods(pred_dir):
    return [path.stem for path in sorted(pred_dir.glob("*.pt"))]


def score_file(path, ref, speaker, expected_split, indices):
    item = torch.load(path, map_location="cpu")
    pred = item["pred_tokens"].long()
    cfg = item["config"]
    vocab = int(cfg["vocab_size"])
    if pred.shape != ref.shape:
        raise ValueError(f"{path} predicted shape {tuple(pred.shape)} but expected {tuple(ref.shape)}")
    if int(cfg["num_codebooks"]) != ref.shape[-1]:
        raise ValueError(f"{path} config num_codebooks does not match evaluation split")
    if item.get("split") not in (None, expected_split):
        raise ValueError(f"{path} was generated for split={item.get('split')!r}, expected {expected_split!r}")

    pred_slice = pred[indices]
    ref_slice = ref[indices]
    speaker_slice = speaker[indices]
    acc = float((pred_slice == ref_slice).float().mean())
    validity = dependency_validity(pred_slice, speaker_slice, vocab)
    pred_feat = codec_features(pred_slice, speaker_slice, vocab)
    ref_feat = codec_features(ref_slice, speaker_slice, vocab)
    fd = frechet_distance(pred_feat, ref_feat)
    speaker_fd = speaker_conditioned_fd(pred_slice, ref_slice, speaker_slice, vocab)
    steps_per_24_frames = 24.0 / float(cfg["stack_factor"])
    return {
        "token_accuracy": acc,
        "dependency_validity": validity,
        "codec_fd": fd,
        "speaker_conditioned_codec_fd": speaker_fd,
        "primary_steps_per_24_frames": steps_per_24_frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="checkpoints/predictions")
    parser.add_argument("--data", default="data/synthetic_codec")
    parser.add_argument("--data-split", default="test")
    parser.add_argument("--out", default="scoring/scores_test.json")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    dataset = CodecTokenDataset(args.data, args.data_split)
    ref_tokens = dataset.tokens
    speaker = dataset.speaker
    indices = split_indices(ref_tokens.shape[0])

    methods = discover_methods(pred_dir)
    if not methods:
        raise SystemExit(f"no scoreable prediction artifacts found in {pred_dir}")

    results = {}
    for method in methods:
        path = pred_dir / f"{method}.pt"
        results[method] = score_file(path, ref_tokens, speaker, args.data_split, indices)

    output = {
        "_metadata": {
            "slice": SLICE,
            "data_split": args.data_split,
            "partition_seed": PARTITION_SEED,
            "num_examples": int(indices.numel()),
            "method_source": str(pred_dir),
        },
        "experiments": {
            EXPERIMENT_NAME: {
                "slice": SLICE,
                "partition_seed": PARTITION_SEED,
                "num_examples": int(indices.numel()),
                "results": results,
            }
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
