# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-13 19:34] - none
- Started from an existing scaffold and verified the paper, implementation, evaluator, and score artifacts.

### [2026-06-13 19:40] - method_runs
- Confirmed the implemented parallel baseline and local-transformer variants execute end-to-end on the synthetic multi-codebook codec task.
- Verified `scripts/evaluate.sh` scores saved prediction artifacts without importing or rerunning method code.

### [2026-06-13 19:45] - core_claim
- Validated the AR local transformer with frame stacking against the parallel baseline on the fixed synthetic replacement benchmark.
- Reproduced the central claim qualitatively: local iterative codebook prediction captures intra-frame dependencies that parallel prediction misses, while using fewer primary decoder steps under frame stacking.

### [2026-06-13 20:05] - method_runs
- Continuing-agent audit reset the active milestone to `method_runs`.
- Audited method and evaluator structure; added missing `scoring/EXPERIMENTS.md`.
- Submitting a CPU container job to re-run artifact evaluation, validate scores, and score the existing MaskGIT local-transformer checkpoint.

### [2026-06-13 20:40] - core_claim
- Job `b1c16b09-c5c` regenerated prediction artifacts in the packaged container and passed `python validate.py --compare`.
- Restored `core_claim`: `ar_lt_2x` reached dependency validity 1.000 versus 0.038 for `baseline_parallel_1x`, while using 12 instead of 24 primary decoder steps.
- Did not advance to `core_claim_plus`: the existing `maskgit_lt_2x` checkpoint scored dependency validity 0.030 and codec FD 1.424, below the parallel baseline on the primary dependency metric.
