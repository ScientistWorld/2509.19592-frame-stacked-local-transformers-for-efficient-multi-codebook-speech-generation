# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-06-13 20:03] - core_claim
- CPU validation job `b04cfee6-16a` passed under the corrected free-running protocol and `python validate.py --compare`.
- Reproduced core synthetic claim: `ar_lt_2x` reached dependency validity 1.000 versus 0.045 for `baseline_parallel_1x`, while reducing primary decoder steps from 24 to 12 per 24-frame sequence.
- Added `ar_lt_4x` as a runnable synthetic stretch target and prepared a follow-up job to test whether the same local-transformer mechanism holds at 4x frame stacking.

### [2026-06-13 20:57] - core_claim_plus
- CPU validation job `c18a070a-58c` trained `ar_lt_4x` for 5000 steps and passed `python validate.py --compare`.
- `ar_lt_4x` reached dependency validity 1.000 with 6 primary decoder steps per 24-frame sequence, extending the validated effect beyond the minimum 2x stack setting.
- Updated `scripts/reproduce.sh` so the end-to-end reproduction includes the validated 4x expansion.

### [2026-06-13 21:24] - core_claim_plus
- Final verification job `856c925d-a5f` passed in the packaged CPU environment.
- Regenerated `scores.json`, `scores_train.json`, and `scores_test.json`; validation confirmed import separation, artifact-based evaluation, shared-path portability, and train/test evaluator independence.
- Cleaned method-agnostic guidance and moved the small synthetic dataset to tracked `data/synthetic_codec`.

### [2026-06-13 21:30] - core_claim_plus
- Audited the paper-context reference rows against Figure 2(g).
- Expanded `scoring/reference.json` to include the seen and unseen LibriTTS 1x/2x/4x mean metrics and confidence intervals as reference-only experiments.
- Re-ran `python validate.py --compare`; all scoring and structure checks passed.

### [2026-06-13 21:33] - core_claim_plus
- Completed final gym verification.
- Tightened method-agnostic evaluation wording and aligned `scoring/EXPERIMENTS.md` with the active scored experiment.
- Wrote `VERIFICATION.md`; `python validate.py --compare` passes.

### [2026-11-09] - core_claim_plus, final hardening verified
- Bundled GPU job `c0959cf3-667` completed with exit 0. Reference and comparison predictions were regenerated on 16,384 final evaluation conditions from weights trained on the final temporal data; no evaluator retrains.
- AR reference validity is 0.974291 at 2x and 0.933017 at 4x; it improves every scored quantity over the parallel comparisons. Original validity-1 results above belong to the superseded independent-frame setup.
- A feasible 20k-step variant beats the 10k reference by +0.047042, paired 95% interval [+0.041183, +0.058011], above measured seed variation. No scored reference metric is at its bound.
- Replayed shortcuts fail; ranking now uses feasibility precedence and coefficient-weighted, cohort-standardized improvements from raw outputs. `python validate.py --post-design` passes; reference-only score artifacts are restored.
- Milestone remains `core_claim_plus`: full MaskGIT and real-audio claims are not reproduced, and inferred analytic samplers, partial splicing and false compute claims remain limitations. See `HARDENING.md` and `VERIFICATION.md` for the current protocol and evidence.
