#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from data.codec_dataset import CodecTokenDataset, codec_features, dependency_validity


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


def score_file(path, ref, speaker, expected_split):
    item = torch.load(path, map_location="cpu")
    pred = item["pred_tokens"].long()
    cfg = item["config"]
    vocab = int(cfg["vocab_size"])
    if pred.shape != ref.shape:
        raise ValueError(f"{path} predicted shape {tuple(pred.shape)} but expected {tuple(ref.shape)}")
    if int(cfg["num_codebooks"]) != ref.shape[-1]:
        raise ValueError(f"{path} config num_codebooks does not match test split")
    if item.get("split") not in (None, expected_split):
        raise ValueError(f"{path} was generated for split={item.get('split')!r}, expected {expected_split!r}")
    acc = float((pred == ref).float().mean())
    validity = dependency_validity(pred, speaker, vocab)
    pred_feat = codec_features(pred, speaker, vocab)
    ref_feat = codec_features(ref, speaker, vocab)
    fd = frechet_distance(pred_feat, ref_feat)
    speaker_fd = speaker_conditioned_fd(pred, ref, speaker, vocab)
    steps_per_24_frames = 24.0 / float(cfg["stack_factor"])
    return {
        "token_accuracy": acc,
        "dependency_validity": validity,
        "codec_fd": fd,
        "speaker_conditioned_codec_fd": speaker_fd,
        "primary_steps_per_24_frames": steps_per_24_frames,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", default="checkpoints/predictions")
    p.add_argument("--data", default="data/synthetic_codec")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="scoring/scores.json")
    p.add_argument("--reference", default="scoring/reference.json")
    p.add_argument(
        "--include-unreferenced",
        action="store_true",
        help="Include prediction files whose method names are not listed for the synthetic experiment in reference.json.",
    )
    args = p.parse_args()
    pred_dir = Path(args.pred_dir)
    dataset = CodecTokenDataset(args.data, args.split)
    ref_tokens = dataset.tokens
    speaker = dataset.speaker
    allowed = None
    reference = Path(args.reference)
    if reference.exists() and not args.include_unreferenced:
        reference_spec = json.loads(reference.read_text())
        allowed = set(reference_spec["experiments"]["synthetic_codec_unseen"]["results"])
    results = {}
    skipped = []
    for path in sorted(pred_dir.glob("*.pt")):
        method = path.stem
        if allowed is not None and method not in allowed:
            skipped.append(str(path))
            continue
        results[method] = score_file(path, ref_tokens, speaker, args.split)
    if not results:
        detail = f" skipped unreferenced artifacts: {', '.join(skipped)}" if skipped else ""
        raise SystemExit(f"no scoreable prediction artifacts found in {pred_dir}.{detail}")
    for path in skipped:
        print(f"skipping unreferenced prediction artifact: {path}")
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
