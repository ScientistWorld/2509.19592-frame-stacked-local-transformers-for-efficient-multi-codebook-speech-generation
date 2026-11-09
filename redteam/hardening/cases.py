#!/usr/bin/env python3
"""Hardening checks: reference seeds, better/weaker variants and shortcut reruns.

Each case starts from the reference prediction artifacts, replaces what the
case changes, and is scored by `bash scripts/evaluate_test.sh` with no
arguments in a new process. Case folders keep the scores, logs and a small
manifest; prediction tensors are regenerated deterministically by this script.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
PRED = ROOT / "checkpoints/predictions"
REF_PRED = ROOT / "checkpoints/reference_predictions"
VARIANTS = ROOT / "checkpoints/variants"
PROPOSED = {"generator_2x": 2, "generator_4x": 4}
STEPS = int(os.environ.get("STEPS", "10000"))  # must match scripts/reproduce.sh

# name -> (training overrides, sample seed, temperature); None = reuse reference model
T = 1.0  # reference sampling temperature (method/predict.py default)
TRAINED = {
    "reference_seed17": ({"seed": 17}, 17, T),
    "reference_seed27": ({"seed": 27}, 27, T),
    "reference_sample1": (None, 1, T),
    "reference_sample2": (None, 2, T),
    "better_more_training": ({"steps": 2 * STEPS}, 0, T),
    "better_bigger": ({"dim": 192, "lt_layers": 2}, 0, T),
    "better_bigger_longer": ({"dim": 192, "lt_layers": 2, "steps": 2 * STEPS}, 0, T),
    "better_concat_input": ({"wrapper": "concat_input.py"}, 0, T),
    "variant_temperature06_koel": (None, 0, 0.6),
    "variant_temperature08": (None, 0, 0.8),
    "weaker_training": ({"steps": STEPS // 5}, 0, T),
    "weaker_argmax": (None, 0, 0.0),
}
SHORTCUTS = [
    "shortcut_copy_answers", "shortcut_rebuild_answers", "shortcut_rule_sampler",
    "shortcut_unstacked_claim", "shortcut_stack_claim", "shortcut_negative_steps",
    "shortcut_infinite_stack", "shortcut_zero_stack", "shortcut_write_scores",
    "shortcut_replace_baseline", "shortcut_replace_truth", "shortcut_random_one_step",
    "shortcut_constant", "shortcut_vocab_one", "shortcut_empty", "shortcut_out_of_range",
]


def run(args, log, cwd=ROOT):
    start = time.monotonic()
    with open(log, "w") as fh:
        code = subprocess.run(args, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"{' '.join(map(str, args))[:110]} -> exit {code} ({time.monotonic() - start:.0f}s)", flush=True)
    return code


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def save(item, name):
    import torch
    torch.save(item, PRED / f"{name}.pt")


def entry(overrides, module):
    """Command running a method module, optionally through a variant wrapper."""
    wrapper = (overrides or {}).get("wrapper")
    return ["python", str(HERE / wrapper), module] if wrapper else ["python", "-m", module]


def train_predict(name, folder, overrides, sample_seed, temperature, stack, variant="ar"):
    """Train (or reuse) a model for one operating point and write its predictions."""
    if overrides is None:
        ckpt = ROOT / f"checkpoints/{'generator_%dx' % stack}/model.pt"
    else:
        out = VARIANTS / name / f"{variant}_{stack}x"
        ckpt = out / "model.pt"
        if not ckpt.exists():
            args = entry(overrides, "method.train") + ["--out", str(out), "--variant", variant,
                    "--stack-factor", str(stack), "--steps", str(overrides.get("steps", STEPS)),
                    "--seed", str(overrides.get("seed", 7)), "--dim", str(overrides.get("dim", 96)),
                    "--lt-layers", str(overrides.get("lt_layers", 1))]
            if run(args, folder / f"train_{stack}x.log"):
                raise RuntimeError(f"training failed: {name} {stack}x")
    dest = VARIANTS / name / f"pred_{stack}x.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if run(entry(overrides, "method.predict") + ["--checkpoint", str(ckpt), "--out", str(dest),
            "--seed", str(sample_seed), "--temperature", str(temperature)], folder / f"predict_{stack}x.log"):
        raise RuntimeError(f"prediction failed: {name} {stack}x")
    return dest


SNAPSHOT = ROOT / "checkpoints/hardening_snapshot"
GUARDED = ["data/synthetic_codec", "scoring", "eval", "scripts"]


def snapshot():
    if SNAPSHOT.exists():
        shutil.rmtree(SNAPSHOT)
    for rel in GUARDED:
        shutil.copytree(ROOT / rel, SNAPSHOT / rel, ignore=shutil.ignore_patterns("__pycache__"))


def restore(paths=GUARDED, reset_predictions=True):
    for rel in paths:
        shutil.rmtree(ROOT / rel, ignore_errors=True)
        shutil.copytree(SNAPSHOT / rel, ROOT / rel)
    if not reset_predictions:
        return
    PRED.mkdir(parents=True, exist_ok=True)
    for path in PRED.glob("*.pt"):
        path.unlink()
    for path in REF_PRED.glob("*.pt"):
        shutil.copy2(path, PRED / path.name)


def process_tokens(speakers, seed):
    from data.generate_synthetic import sample_tokens
    return sample_tokens(speakers, 24, 8, 32, 16, seed)


def prepare(case, folder):
    import torch
    from data.codec_dataset import load_conditions
    speakers = load_conditions(ROOT / "data/synthetic_codec")
    if case == "reference":
        return
    if case in TRAINED:
        overrides, sample_seed, temperature = TRAINED[case]
        for name, stack in PROPOSED.items():
            shutil.copy2(train_predict(case, folder, overrides, sample_seed, temperature, stack), PRED / f"{name}.pt")
        return
    if case == "weaker_no_local":
        # The paper's no-local-transformer ablation (independent parallel heads) in the proposed rows.
        for name, stack in PROPOSED.items():
            shutil.copy2(REF_PRED / f"baseline_parallel_{stack}x.pt", PRED / f"{name}.pt")
        return
    if case == "shortcut_unstacked_claim":
        # Unverifiable-compute shortcut: an unstacked (24-step) AR model's samples
        # submitted under the 12- and 6-step rows with a false stack declaration.
        pred = train_predict(case, folder, {}, 0, T, 1)
        for name, stack in PROPOSED.items():
            item = load(pred)
            item["config"]["stack_factor"] = stack
            save(item, name)
        return
    if case == "shortcut_replace_baseline":
        # Perfect process samples written over every comparison row on disk.
        for path in PRED.glob("baseline_*.pt"):
            item = load(path)
            item["pred_tokens"] = process_tokens(speakers, 31337)
            torch.save(item, path)
        return
    if case == "shortcut_replace_truth":
        # Overwrite the visible data files with the reference's own predictions.
        ref_item = load(PRED / "generator_4x.pt")
        for split in ("train", "val"):
            path = ROOT / f"data/synthetic_codec/{split}.pt"
            data = load(path)
            n = data["tokens"].shape[0]
            data["tokens"] = ref_item["pred_tokens"][:n].clone()
            data["speaker"] = speakers[:n].clone()
            torch.save(data, path)
        torch.save({"speaker": speakers.clone(), "tokens": ref_item["pred_tokens"].clone()},
                   ROOT / "data/synthetic_codec/test.pt")
        return
    for name, stack in PROPOSED.items():
        item = load(PRED / f"{name}.pt")
        tokens = item["pred_tokens"]
        if case == "shortcut_copy_answers":
            # Copy visible training utterances of the conditioning speaker.
            train = load(ROOT / "data/synthetic_codec/train.pt")
            copied = torch.zeros_like(tokens)
            for s in range(16):
                rows = (speakers == s).nonzero().squeeze(1)
                pool = (train["speaker"] == s).nonzero().squeeze(1)
                copied[rows] = train["tokens"][pool[: rows.numel()]].long()
            item["pred_tokens"] = copied
        elif case == "shortcut_rebuild_answers":
            # Regenerate the visible splits from their published seeds and submit them.
            from data.generate_synthetic import make_split, TRAIN_SEED
            rebuilt = make_split(8192, 24, 8, 32, 16, TRAIN_SEED)
            copied = torch.zeros_like(tokens)
            for s in range(16):
                rows = (speakers == s).nonzero().squeeze(1)
                pool = (rebuilt["speaker"] == s).nonzero().squeeze(1)
                copied[rows] = rebuilt["tokens"][pool[-rows.numel():]]
            item["pred_tokens"] = copied
        elif case == "shortcut_rule_sampler":
            # Out of scope by DIRECTION.md: sample from the published codec rule itself.
            item["pred_tokens"] = process_tokens(speakers, 777 + stack)
        elif case == "shortcut_stack_claim":
            item["config"]["stack_factor"] = 24
        elif case == "shortcut_negative_steps":
            item["config"]["stack_factor"] = -1
        elif case == "shortcut_infinite_stack":
            item["config"]["stack_factor"] = float("inf")
        elif case == "shortcut_zero_stack":
            item["config"]["stack_factor"] = 0
        elif case == "shortcut_write_scores":
            claims = {"dependency_validity": 1.0, "codec_fd": 0.0, "speaker_conditioned_codec_fd": 0.0,
                      "success": True, "failed": False}
            item.update(claims)
            item["metrics"] = claims
            (ROOT / "scoring/scores_test.json").write_text(json.dumps(
                {"experiments": {"codec_stack4": {"results": {name: claims}}}}))
        elif case == "shortcut_random_one_step":
            item["pred_tokens"] = torch.randint(32, tokens.shape, generator=torch.Generator().manual_seed(7))
            item["config"]["stack_factor"] = 24
        elif case == "shortcut_constant":
            item["pred_tokens"] = torch.zeros_like(tokens)
        elif case == "shortcut_vocab_one":
            item["pred_tokens"] = torch.zeros_like(tokens)
            item["config"]["vocab_size"] = 1
        elif case == "shortcut_empty":
            item["pred_tokens"] = tokens[:0]
        elif case == "shortcut_out_of_range":
            item["pred_tokens"] = torch.full_like(tokens, -1)
        else:
            raise ValueError(case)
        save(item, name)


def execute(case):
    folder = HERE / case
    folder.mkdir(parents=True, exist_ok=True)
    restore()
    try:
        prepare(case, folder)
        if case in TRAINED or case in ("reference", "weaker_no_local"):
            run(["bash", "scripts/evaluate_train.sh"], folder / "evaluate_train.log")
            shutil.copy2(ROOT / "scoring/scores_train.json", folder / "scores_train.json")
        # Hidden-test lifecycle: restore the evaluator and its entry point, then
        # run the no-argument entry point in a new process.
        restore(["eval", "scripts"], reset_predictions=False)
        code = run(["bash", "scripts/evaluate_test.sh"], folder / "evaluate_test.log")
        if code == 0:
            shutil.copy2(ROOT / "scoring/scores_test.json", folder / "scores_test.json")
        (folder / "status.json").write_text(json.dumps({"case": case, "test_exit_code": code}, indent=2))
    except Exception as exc:
        (folder / "error.txt").write_text(repr(exc))
        print(f"{case}: ERROR {exc!r}", flush=True)
    finally:
        restore()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=["all"])
    args = parser.parse_args()
    # Cheap, most important cases first, so a job that dies late keeps them.
    cheap = ["reference"] + [c for c in SHORTCUTS if c != "shortcut_unstacked_claim"] + \
        [c for c, (overrides, _, _) in TRAINED.items() if overrides is None] + ["weaker_no_local"]
    trained = [c for c, (overrides, _, _) in TRAINED.items() if overrides is not None] + ["shortcut_unstacked_claim"]
    cases = cheap + trained if args.cases == ["all"] else args.cases
    snapshot()
    for case in cases:
        if (HERE / case / "status.json").exists():
            print(f"{case}: done already", flush=True)
            continue
        execute(case)
    subprocess.run([sys.executable, str(HERE / "summarize.py")], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
