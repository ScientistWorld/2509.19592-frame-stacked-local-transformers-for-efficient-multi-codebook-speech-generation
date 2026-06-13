# Reproduction Milestones

**Current: none**

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
