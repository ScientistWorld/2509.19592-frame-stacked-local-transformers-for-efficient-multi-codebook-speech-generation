# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-13 18:51] - none
- Started reproduction work in a newly initialized workspace.
- Local paper files were absent, so the official arXiv HTML, PDF, and source were fetched into `paper/`.
- Identified the core method: parallel codebook prediction baseline, AR local transformer, MaskGIT local transformer, and frame stacking.
- Implemented a synthetic multi-codebook codec-token replacement task because the original Koel-TTS data, NanoCodec tokenizer, and trained models are not released.

### [2026-06-13 19:08] - none
- CPU test job `e194b6d3-0a8` trained and predicted with the no-LT parallel baseline.
- The job failed during evaluation due to a codec embedding broadcast bug before local-transformer methods ran.
- Fixed the evaluator/data feature broadcast issue for the next submission.

### [2026-06-13 19:15] - method_runs
- CPU test job `27ab7f98-e2c` trained and evaluated the parallel baseline, AR LT, and MaskGIT LT end-to-end.
- The method reached runnable status, but AR/MaskGIT dependency validity was still at random-baseline level, so `core_claim` is not reached.
- Diagnosed and fixed AR generation alignment: sampled previous slots were not shifted to match the teacher-forced local-transformer input.

### [2026-06-13 19:28] - method_runs
- CPU test job `30c8fb22-0d5` re-evaluated existing checkpoints after the AR sampling fix.
- AR LT improved slightly but remained too close to the baseline (`dependency_validity` 0.055 vs 0.038), so the core claim is not yet reproduced.
- Configured the next job to retrain only AR LT for a longer run using the existing baseline.
