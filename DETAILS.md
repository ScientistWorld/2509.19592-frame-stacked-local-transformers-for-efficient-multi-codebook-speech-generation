# Codec generation gym

Learn joint multi-codebook generation at fixed sequential budgets. Start with `problem.md` and `evaluation.md`; `method.md` documents the scaled reference. `scoring/reference.json` holds the reproduced comparison and preserves the paper's real-speech results separately. The gym measures synthetic token generation and does not establish real-audio quality.

---

# Problem

Learn to generate joint multi-codebook codec sequences conditioned on a speaker. Each utterance contains 24 frames, 8 codebooks per frame, vocabulary 32, and one of 16 training speakers. Good outputs preserve within-frame codebook dependencies, transitions between frames, speaker characteristics, utterance-level consistency, and diversity.

The fixed synthetic training and validation artifacts contain 8192 and 1024 utterances. They substitute for the paper's unavailable speech mixture and codec. The distribution includes stochastic temporal transitions and persistent speaker-dependent residual variation. There is no text or audio in this scaled task, and no exact-token target for a generated utterance.

Generate one new utterance for each of the 16384 speaker IDs in `data/synthetic_codec/test_conditions.pt`, at each of two fixed operating points: at most 12 primary sequential steps per utterance (2 frames per step), and at most 6 (4 frames per step). Computation within a primary step is unrestricted. The scientific question is how well quality can be preserved when the primary sequence model runs less often.

Use the provided training data to learn the generator. Complete copies of visible utterances, evaluator-side reference generators, and false computation declarations are outside the task. See `scoring/CONSTRAINTS.md`.

---

# Method

The paper studies local transformers (LTs) for multi-codebook speech generation. The baseline predicts all codebooks of a frame in parallel from the primary decoder state, assuming conditional independence across codebooks. The proposed models keep an autoregressive primary decoder over time and add a small LT that expands each primary hidden state into the frame's codebook tokens. With frame stacking, the primary decoder runs once per `S` frames and the LT decodes all `S x N` codebook tokens of the stack.

Implementation (`method/models.py`, scaled down: width 96, 4 heads, 2 primary layers, 1 LT layer, 10000 AdamW steps at learning rate 1e-3, batch 128):

- Every (frame-in-stack, codebook) slot has its own token embedding table, shared by the primary decoder and the LT; the primary decoder input averages the slot embeddings of the previous stack (as in the paper), plus speaker and step-position embeddings.
- `ar` (the reference method): an autoregressive LT samples the stacked slots one by one, conditioned on the primary hidden state and the previously sampled slots.
- `maskgit`: a non-causal LT trained with MaskGIT's cosine masking schedule; at inference it starts fully masked and, over 3 passes, samples and fixes the slots with the highest predicted purity.
- `none` (baseline): independent output heads, one per slot. As in the paper, the baseline's primary decoder gets the LT models' total layer count (3 layers).
- Generation is free-running ancestral sampling at temperature 1 for every model.

Reference artifacts (`scripts/reproduce.sh`): `generator_2x` and `generator_4x` are the AR LT at stack factors 2 and 4 (12 and 6 primary steps per 24 frames). Comparison rows: `baseline_parallel_{1,2,4}x` and `baseline_maskgit_{2,4}x`.

Deviations from the paper:

- Koel-TTS on 18k hours of speech is replaced by a synthetic codec-token process (see `briefing/problem.md`) because data and models are unreleased; there is no text conditioning.
- Models are much smaller (width 96 instead of 768, 2+1 layers instead of 12+4) and trained for far fewer steps.
- Sampling uses temperature 1 without top-k or classifier-free guidance. The paper uses Koel-TTS's speech-tuned sampling (temperature 0.6, top-k 80, CFG); top-k 80 is a no-op with 32 tokens, there is no text to guide, and on this distribution-matching task temperature 0.6 trades diversity (FD) for validity.
- Evaluation uses dependency validity and codec-latent Fréchet distances instead of WER, TitaNet SSIM and UTMOSv2.

The final AR reference beats the parallel comparisons on every scored quantity. The MaskGIT comparison improves over matched-stack parallel decoding at 4x but fails the validity threshold at both points and does not reproduce the full quality claim. This partial reproduction and the remaining synthetic-task shortcuts are documented in `HARDENING.md`.

---

# Evaluation

Submit `checkpoints/predictions/generator_2x.pt` and `generator_4x.pt`. Each is a PyTorch dictionary containing `pred_tokens`, an integer tensor of shape `(16384, 24, 8)` with IDs from 0 through 31, and `config`, a dictionary containing the exact fixed `stack_factor` (2 or 4 respectively). Optional top-level `split` is accepted. Other top-level fields, including submitted metrics and verdicts, fail the artifact. Additional configuration fields are ignored except contradictory task dimensions, which fail. Generation uses speaker IDs only.

Both scripts run without arguments and score all files in `checkpoints/predictions/`. Additional `<prefix>_2x.pt` / `<prefix>_4x.pt` pairs use the same interface and are ranked as additional attempts. Saved attempts may also be stored at `checkpoints/attempts/<attempt>/predictions/` with the two canonical filenames. Every attempt is rescored from raw tensors; saved scores are never ranking inputs. Baseline filenames are reserved for evaluator-frozen comparisons. Changing those outputs or the fixed train/validation data fails the proposed run.

The visible train evaluator scores one fixed half of the conditions (8192 utterances); the hidden test scores the other half. They have independent implementations, reference samples, and fixtures. Both compare against four reference utterances per conditioning row. The test reference sampler and seed are held only by `eval/test/`. All methods see the same speaker conditions and references. `scripts/evaluate_train.sh` writes `scoring/scores_train.json`; `scripts/evaluate_test.sh` writes `scoring/scores_test.json`.

Each experiment measures:

- `dependency_validity` ↑: fraction of 191 structural checks per utterance that pass (168 within-frame checks and 23 temporal checks).
- `codec_fd` ↓: full-covariance Fréchet distance between consecutive-frame pairs in a fixed 32-dimensional codec latent space (64 dimensions per pair).
- `speaker_conditioned_codec_fd` ↓: the same Fréchet distance computed for each conditioning speaker and averaged. This measures conditional distribution quality; it is not TitaNet speaker cosine similarity.

Distances are not clipped. Finite-sample reference distances need not be zero, even for a perfect distribution sampler. Raw diagnostic components remain available for failed runs, so comparisons still show their scientific behavior.

A run fails if either required artifact is missing or malformed, contains invalid IDs or contradictory dimensions, claims a different operating point, duplicates more than 1% complete visible utterances, or scores below 0.5 on total dependency validity, within-frame validity, or temporal validity. Duplicate checks use frozen canonical visible data. The former 12-frame window rule was removed because it rejected stronger learned models. A method must still avoid copying or splicing training data; partial-copy detection is left to the separate shortcut review. A method-side `data/synthetic_codec/test.pt` is forbidden.

Ranking is lexicographic: every feasible attempt ranks above every failed attempt. Within a tier, add each experiment weight times the absolute value of each metric's coefficient times its improvement over the reference, using `higher_is_better` for the direction, divided by that metric's population standard deviation across the attempts currently ranked and the reference. A zero-variance metric contributes zero. Standard deviations and per-metric contributions are emitted under `_ranking`. Thus scores change when the comparison cohort changes; compare attempts together. The reference measurements are recomputed from evaluator-frozen raw predictions each time. Invalid artifacts get finite default measurements and an explicit failure, never a high rank.

The 2x experiment has weight 0.4; the 4x experiment has weight 0.6, emphasizing the harder aggressive operating point. Per metric, validity has coefficient 1.0, global FD 0.6, and conditional FD 0.4. Efficiency is a constraint, not a rewarded method-written number. Token artifacts cannot prove actual sequential compute; false declarations remain subject to shortcut review.

The paper's real-speech metrics remain unchanged under `_paper_reported` in `scoring/reference.json`; they are context rather than numerical targets for this synthetic task.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 16m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-06-13 19:44] - core_claim
- GPU test job `0f9f2e6f-d80` retrained the AR local transformer for 5000 steps after the sampling fix.
- The synthetic core experiment now supports the paper's central claim: AR LT reaches `dependency_validity` 1.000 versus 0.038 for the parallel baseline, with NLL 0.435 versus 3.470.
- Frame stacking is active in the reproduced method: `ar_lt_2x` uses 12 primary steps per 24-frame sequence versus 24 for the unstacked parallel baseline.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 21m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 8 total (6 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 9 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 9 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 10 total (8 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 10 total (8 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.3h
- **Jobs**: 10 total (8 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **GPU**: 0.0h
- **Jobs**: 10 total (8 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 1.3h | **GPU**: 1.7h
- **Jobs**: 11 total (9 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: gpt-6.1-sol (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 22m | **GPU**: 0.1h
- **Jobs**: 12 total (10 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**
