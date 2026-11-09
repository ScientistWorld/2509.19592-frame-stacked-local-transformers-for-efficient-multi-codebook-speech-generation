> Historical record from earlier iterations. The final protocol and verification supersede these entries; see `HARDENING.md`, `briefing/evaluation.md` and `scoring/EXPERIMENTS.md`.

# Verification Report

## Initial Assessment

The workspace started at `core_claim_plus`, and that milestone is justified. The reproduced gym result is based on saved prediction artifacts for the synthetic multi-codebook task: the independent baseline has low dependency validity, while the validated dependency-modeling variants reach dependency validity 1.0 at 2x and 4x generation-step reductions. The exact LibriTTS/NanoCodec results remain reference-only because the paper's training data, tokenizer pipeline, model checkpoints, and generated audio are unreleased.

## Issues Found

- Minor: `scoring/EXPERIMENTS.md` documented reference-only LibriTTS experiments even though `scores.json` contains only the runnable `synthetic_codec_unseen` experiment. This could confuse from-scratch scientist agents about what is actually scored.
- Minor: `briefing/evaluation.md` described `primary_steps_per_24_frames` as decoder-specific. The metric is valid, but the method-agnostic evaluation guide should frame it as a hardware-independent generation-effort proxy.
- Cosmetic/pre-existing: `README.md` and `manifest.json` are modified special device nodes, and tracked `trajectory/turn_*.jsonl` files are deleted. These were present before this verification pass and are outside the reproduction gym files; I left them untouched.

## Fixes Applied

- Narrowed `scoring/EXPERIMENTS.md` to the active `synthetic_codec_unseen` experiment and noted that LibriTTS rows are reference-only context in `reference.json`.
- Reworded `briefing/evaluation.md` so `primary_steps_per_24_frames` is described as coarse generation steps declared by the artifact rather than a required architecture.
- Re-ran `python validate.py --compare`; all scoring and structural checks passed.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can run the packaged method and evaluator, and from-scratch agents can use the method-agnostic problem, data, and evaluation interface without importing the paper method. Evaluation reads saved prediction artifacts, not method internals; train/test score artifacts exist and use independent slices; `scripts/evaluate.sh` works standalone for any method that writes the expected artifact format.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: synthetic replacement data only; exact LibriTTS/NanoCodec reproduction is blocked by unreleased resources.
