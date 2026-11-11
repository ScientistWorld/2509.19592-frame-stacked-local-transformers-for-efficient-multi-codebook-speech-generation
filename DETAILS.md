# Codec generation gym

Learn joint multi-codebook generation at fixed sequential budgets. Start with `problem.md` and `evaluation.md`; `method.md` documents the scaled reference. `scoring/reference.json` holds the reproduced comparison and keeps the paper's real-speech results separately. The gym measures synthetic codec-token generation from a hidden speech-like process; it does not establish real-audio quality.

---

# Problem

Learn to generate joint multi-codebook codec sequences conditioned on a speaker. Each utterance has 24 frames and 8 codebooks per frame, with vocabulary 32 per codebook, and belongs to one of 16 training speakers. Good outputs preserve the dependencies between the codebooks of a frame, the transitions between frames, speaker characteristics, utterance-level consistency and diversity.

The codec is a finite scalar quantizer (FSQ), like the paper's NanoCodec. Each codebook owns 3 dimensions of a 24-dimensional codec latent, and its token is one cell of a 4 x 4 x 2 grid over them. Token `c` has the digits `(c // 8, (c // 2) % 4, c % 2)` on the three dimensions. The latent value of each dimension is the cell centre, `(digit + 0.5) / levels * 2 - 1`, in [-1, 1]. All 8 codebooks of a frame are read out from the same underlying speech state, so they are strongly dependent; nothing makes them independent.

The fixed training and validation artifacts (`data/synthetic_codec/train.pt`, `val.pt`) hold 32768 and 2048 utterances with their speaker IDs. They are drawn from a synthetic speech process that stands in for the paper's unreleased 18k-hour speech mixture. The process is a hidden random nonlinear state-space model with phone-like regimes and persistent utterance style, whose latent trajectory the codec quantizes. Its parameters are not public, and its training data are the only view of it. There is no text or audio in this scaled task, and no exact-token target for a generated utterance.

Generate one new utterance for each of the 16384 speaker IDs in `data/synthetic_codec/test_conditions.pt`, at two fixed operating points:
- at most 12 primary sequential steps per utterance (2 frames per step);
- at most 6 primary steps (4 frames per step).

A second task tests speakers that the training data cover poorly. `data/synthetic_codec_fewshot/` holds an independent speech-like process with the same codec and dimensions, but 256 speakers:
- speakers 0–191 have 160 training utterances each;
- speakers 192–255 have only 8 training utterances each (validation covers speakers 0–191 only).

All of its speakers share one speaker space, so what is learned from well-covered speakers can transfer. Generate one new utterance for each of the 16384 speaker IDs in `data/synthetic_codec_fewshot/test_conditions.pt` (the few-shot speakers 192–255), at the 6-step operating point (4 frames per step). Its training data may be used together with the main task's, but the two processes differ.

Computation within a primary step is unrestricted. The scientific question is how well generation quality can be kept when the primary sequence model runs less often.

Use the provided training data to learn the generator. Outside the task:
- copying, editing or splicing visible utterances;
- calling evaluator-side samplers;
- false computation declarations.

See `scoring/CONSTRAINTS.md`.

---

# Method

The paper studies local transformers (LTs) for multi-codebook speech generation.
- **Baseline:** predicts all codebooks of a frame in parallel from the primary decoder state, assuming the codebooks are conditionally independent.
- **Proposed models:** keep an autoregressive primary decoder over time and add a small LT that expands each primary hidden state into the frame's codebook tokens.
- **Frame stacking:** the primary decoder runs once per `S` frames, and the LT decodes all `S x N` codebook tokens of the stack.

## Implementation

The code is in `method/models.py`, scaled down from the paper:
- width 96, 4 heads, 2 primary layers and 1 LT layer;
- 10000 AdamW steps at a constant learning rate of 1e-3, batch 128.

The paper specifies AdamW but no schedule. `method/train.py --lr-schedule cosine` is an optional variant that the reference does not use.

Model details:
- **Embeddings.** Every (frame-in-stack, codebook) slot has its own token embedding table, shared by the primary decoder and the LT. The primary decoder input averages the slot embeddings of the previous stack (as in the paper), plus speaker and step-position embeddings.
- **`ar` (the reference method).** An autoregressive LT samples the stacked slots one by one, conditioned on the primary hidden state and the previously sampled slots.
- **`maskgit`.** A non-causal LT trained with MaskGIT's cosine masking schedule. At inference it starts fully masked and, over 3 passes, samples and fixes the slots with the highest predicted purity.
- **`none` (baseline).** Independent output heads, one per slot. As in the paper, the baseline's primary decoder gets the LT models' total layer count (3 layers).
- **Generation.** Free-running ancestral sampling at temperature 1 for every model.

## Reference artifacts

`scripts/reproduce.sh` trains everything with the same recipe:
- `generator_2x` and `generator_4x`: the AR LT at stack factors 2 and 4 (12 and 6 primary steps per 24 frames), trained on `data/synthetic_codec`;
- comparison rows: `baseline_parallel_{1,2,4}x` and `baseline_maskgit_{2,4}x`;
- `generator_fewshot_4x`: the same AR LT at 4x, trained on `data/synthetic_codec_fewshot` alone, with comparison rows `baseline_parallel_fewshot_4x` and `baseline_maskgit_fewshot_4x`.

## Deviations from the paper

- **Data.** Koel-TTS on 18k hours of speech is replaced by a synthetic codec-token process (see `briefing/problem.md`), because the data and models are unreleased. There is no text conditioning. The codec is an 8-codebook FSQ, like the paper's NanoCodec, at a much smaller vocabulary (32 per codebook).
- **Scale.** Models are much smaller (width 96 instead of 768, 2+1 layers instead of 12+4) and trained for far fewer steps.
- **Sampling.** Generation uses temperature 1 without top-k or classifier-free guidance. The paper uses Koel-TTS's speech-tuned sampling (temperature 0.6, top-k 80, CFG). Top-k 80 is a no-op with 32 tokens, and there is no text to guide. On this distribution-matching task, temperature 0.6 narrows the generated distribution: on the reference weights it is measured in `HARDENING.md`.
- **Metrics.** Evaluation uses the paper's codec-space Fréchet distance, a nonparametric codebook-pair divergence and a speaker-conditioned FD, instead of WER, TitaNet SSIM and UTMOSv2.

The reproduction results, and how they compare with the paper's claims, are in `HARDENING.md`.

---

# Evaluation

## Interface

Submit three files in `checkpoints/predictions/`:
- `generator_2x.pt` and `generator_4x.pt`, for the main task's conditions;
- `generator_fewshot_4x.pt`, for the few-shot task's conditions.

Each is a PyTorch dictionary with:
- `pred_tokens`: an integer tensor of shape `(16384, 24, 8)` with IDs from 0 through 31, one row per conditioning speaker in file order;
- `config`: a dictionary holding the exact fixed `stack_factor` (2, 4 and 4 respectively).

An optional top-level `split` is accepted. Any other top-level field, including submitted metrics and verdicts, fails the artifact. Other configuration fields are ignored, except task dimensions that contradict the fixed task, which fail. Generation uses speaker IDs only.

Both scripts run without arguments and score all files in `checkpoints/predictions/`:
- additional `<prefix>_2x.pt` / `<prefix>_4x.pt` / `<prefix>_fewshot_4x.pt` triples use the same interface and are scored as additional attempts;
- saved attempts may also be stored at `checkpoints/attempts/<attempt>/predictions/` with the three canonical filenames.

Every attempt is rescored from raw tensors; saved scores are never inputs. Baseline filenames are reserved for evaluator-frozen comparisons. Changing those outputs, or the fixed train/validation data, fails the proposed run.

## Slices

The visible train evaluator scores one fixed half of each task's conditions (8192 utterances); the hidden test scores the other half. For the main task the halves are a fixed random split of the rows. For the few-shot task they are the rows of disjoint speaker halves: speakers 192–223 for train and 224–255 for test. The two evaluators have independent implementations and fixtures. Both compare against four reference utterances per conditioning row, drawn from the task's data process.
- The train evaluator holds only frozen sufficient statistics of its reference sample (means, covariances and token-pair counts), not the utterances.
- The test reference sampler and its seed are held only by `eval/test/`.

All methods see the same speaker conditions and references. `scripts/evaluate_train.sh` writes `scoring/scores_train.json` and `scripts/evaluate_test.sh` writes `scoring/scores_test.json`, each under `experiments.<experiment>.results.<row>`.

## Measurements

There are three experiments: `codec_stack2` (row `generator_2x`), `codec_stack4` (`generator_4x`) and `codec_fewshot_stack4` (`generator_fewshot_4x`). Each measures three quantities; lower is better for all three.

- `codec_fd`: full-covariance Fréchet distance in the codec's embedding space (the public FSQ latent of `briefing/problem.md`), between generated and reference consecutive-frame pairs (48 dimensions per pair).
- `codebook_pair_divergence`: mean Jensen–Shannon divergence, in bits, between the generated and reference joint token distributions of codebook pairs. It covers all 28 within-frame pairs and all 64 pairs of (codebook at frame t, codebook at frame t+1). It detects implausible codebook combinations that second moments can miss.
- `speaker_conditioned_codec_fd`: the codec FD computed for each conditioning speaker of the slice and averaged (16 main speakers; 32 few-shot speakers per slice). This measures conditional distribution quality; it is not TitaNet speaker cosine similarity.

Nothing is clipped. Finite-sample values are not zero even for a perfect sampler of the data process.

## Failure

A run fails if any of the following holds:

- **Artifact.** Either required artifact is missing or malformed, contains invalid IDs or contradictory dimensions, or claims a different operating point.
- **Copied halves.** More than 1% of scored utterances repeat an aligned 12-frame half (frames 0–11 or 12–23, all codebooks) of a frozen visible utterance.
- **Visible windows** (`visible_window_excess`). This compares how often the scored utterances' 4-frame windows match the frozen visible utterances with how often they match a fresh sample of the data process of the same size and speaker composition. A window matches if it agrees, at the same frame positions, on all 32 tokens or on any of 8 fixed subsets of 12 of them. The run fails when the visible match rate exceeds the fresh-sample rate by more than 0.05 of all windows.
  - Copies with scattered edits still match, so the test covers copies, edited copies and splices.
  - For a generator that does not reproduce visible segments, the excess has expectation 0.
  - The test draws its fresh sample at evaluation time. The train evaluator holds only the 48-bit keys of its fresh sample's windows.
- **Reused operating point.** More than 1% of one main operating point's scored utterances repeat a 12-frame half found in any row of the other operating point's file. The two rows must be separate samples.
- **Not better than noise.** Any of the three measurements is not below half of its value for uniform random tokens.
- **Integrity.** The fixed training data or a reserved comparison output was changed, or a method-side `data/synthetic_codec/test.pt` exists.

Copy and window checks use each task's own visible utterances and fresh sample. A failed run receives the worst value of every scored metric in all three experiments: each measurement equals that of uniform random tokens (`_metadata.failed_run_values`). A failed or missing row fails the whole run.
- Raw diagnostics of the actual outputs stay under `_raw`, and the reason under `_failures`.
- Comparison rows are frozen by the evaluator. They keep their raw measurements, and any failed check is recorded under `_comparison_status`.

## Rank

The rank sums over experiments and metrics: experiment weight × |metric coefficient| × improvement over the reference. Direction comes from `higher_is_better`. Each improvement is divided by that metric's variation across the ranked attempts, so scores depend on the comparison cohort and attempts should be compared together. The evaluator's own `_ranking` block is a diagnostic of the same quantity for the attempts on disk. The reference measurements are recomputed each time from evaluator-frozen raw predictions.

Weights and coefficients:
- `codec_stack2` has weight 0.3, `codec_stack4` 0.4 and `codec_fewshot_stack4` 0.3. The aggressive 4x point carries the most weight, in total 0.7 across the two tasks.
- Per metric: codec FD 1.0, codebook-pair divergence 0.8, speaker-conditioned FD 0.6.

Efficiency is a constraint, not a rewarded method-written number. Token artifacts cannot prove actual sequential compute, so false declarations remain subject to shortcut review.

The paper's real-speech metrics remain unchanged under `_paper_reported` in `scoring/reference.json`. They are context, not numerical targets for this synthetic task.

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

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 58m | **GPU**: 1.9h
- **Jobs**: 14 total (11 completed, 2 failed)

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

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 29m | **GPU**: 0.3h
- **Jobs**: 16 total (13 completed, 2 failed)

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

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 53m | **GPU**: 2.1h
- **Jobs**: 17 total (14 completed, 2 failed)

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
