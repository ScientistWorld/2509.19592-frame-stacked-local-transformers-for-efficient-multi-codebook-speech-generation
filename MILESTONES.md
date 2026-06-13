# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-13 19:48] - core_claim
- Verified the paper method/evaluation against the arXiv HTML/source files and confirmed no official code release is available.
- Audited the existing gym implementation: the evaluator scores saved prediction artifacts and does not import or rerun `method/`.
- Existing reproduced scores support the central claim on the fixed synthetic codec task: `ar_lt_2x` reaches dependency validity 1.000 versus 0.038 for `baseline_parallel_1x`, while using 12 rather than 24 primary decoder steps per 24 frames.
- Patched script portability and evaluator failure behavior before submitting a CPU container validation job.

### [2026-06-13 19:48] - method_runs
- Continuing audit found that prediction used ground-truth previous frames for temporal context and that evaluation trusted reference/speaker tensors from prediction artifacts.
- Fixed generation to roll out primary steps from model-generated previous frames.
- Fixed evaluation to load the fixed test split from `data/downloads/synthetic_codec` and ignore artifact-provided references/speakers; removed artifact-reported NLL from official scoring.
- Submitting a CPU validation job to regenerate prediction artifacts and scores before restoring `core_claim`.

### [2026-06-13 19:59] - method_runs
- CPU validation job `e966ac72-db7` regenerated free-running prediction artifacts for baseline, AR LT, and MaskGIT, then failed inside the evaluator before writing fresh scores.
- Root cause: evaluator variable shadowing replaced the reference token tensor with the parsed `reference.json` dictionary.
- Fixed evaluator reference handling and split checking; resubmitting CPU validation before restoring `core_claim`.
