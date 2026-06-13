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
