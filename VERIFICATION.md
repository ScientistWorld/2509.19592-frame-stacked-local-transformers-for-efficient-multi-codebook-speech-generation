# Verification Report

## Initial Assessment

The workspace had a validated synthetic reproduction at `core_claim_plus`, with the paper's unavailable Koel-TTS/NanoCodec resources replaced by a fixed synthetic multi-codebook token task. The claimed milestone was justified by prior container jobs: the stronger generated-token method reached dependency validity 1.000 versus 0.045 for the reference baseline, with 2x and 4x reductions in primary generation steps.

## Issues Found

- Minor: Method-agnostic guidance files leaked method identifiers and architecture names, which would compromise from-scratch mode.
- Minor: The main evaluator had been extended with `speaker_conditioned_codec_fd`, but `scores.json` needed regeneration in the packaged environment.
- Minor: The generated synthetic dataset was small enough to track directly under `data/`, but it was previously kept under `data/downloads/`.
- Cosmetic: Root `README.md`, `manifest.json`, and trajectory files remain dirty system artifacts from the surrounding workflow; they are unrelated to the gym code and were left untouched.

## Fixes Applied

- Removed method-specific language from `briefing/problem.md`, `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, and `scoring/DIRECTION.md`.
- Documented the reusable prediction artifact interface in `briefing/evaluation.md`.
- Moved the synthetic dataset to tracked `data/synthetic_codec` and updated script/evaluator defaults.
- Ran CPU validation job `856c925d-a5f`; it regenerated `scores.json`, `scores_train.json`, and `scores_test.json` and passed `python validate.py --compare`.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can build on the provided method and checkpoints, while from-scratch agents can use the problem, data, and evaluator without method leakage. The main limitation is that the runnable benchmark is a synthetic substitute for unreleased paper data and audio metrics.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: synthetic replacement data only
