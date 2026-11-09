# Overview

This gym implements the core computation of "Frame-Stacked Local Transformers for Efficient Multi-Codebook Speech Generation": an autoregressive primary decoder over (stacked) frames, a parallel-head baseline, autoregressive and MaskGIT local transformers (LTs), and frame stacking.

The paper's Koel-TTS/NanoCodec data and models are unreleased, so the gym uses a synthetic codec-token process with strong within-frame codebook dependencies, frame-to-frame dynamics and utterance-level residual style. Generators are compared at two fixed operating points, 12 and 6 primary steps per 24 frames (2x and 4x stacking), by dependency validity and codec-latent Fréchet distances against reference utterances drawn by the hidden evaluator. The reference method is the AR LT; parallel heads (1x/2x/4x) and the MaskGIT LT (2x/4x) are frozen comparison rows. Results of the reference run are summarized in `HARDENING.md` and `scoring/scores_test.json`.

---

# Problem Definition

Speech generation systems that operate on neural audio codec tokens must predict a matrix of discrete tokens with shape `(time, codebook)`. At every acoustic frame, several codebooks jointly describe the same audio frame, and they are not independent of each other. A generator must produce codebook combinations that are internally consistent within each frame, coherent from frame to frame, faithful to the conditioning speaker, and as diverse as real speech, not just plausible marginal tokens.

Generation cost matters as much as quality. A generator that runs a large sequence model once per frame is slow; the gym therefore fixes **operating points**: the number of coarse (primary) sequential generation steps the generator may spend per 24-frame utterance. The scored operating points are **12 steps** and **6 steps** per 24 frames, i.e. every coarse step has to produce 2 or 4 complete frames (16 or 32 codebook tokens). Any additional per-step computation inside a coarse step is allowed. Quality is compared at a fixed operating point.

The gym uses a reproducible synthetic codec because the paper's 18k-hour training mixture, codec tokenizer and trained models are not released. Each utterance has a speaker id (16 speakers, all seen in training) and 24 frames of 8 codebooks with vocabulary 32. Its structure (implemented in `data/generate_synthetic.py`):

- Within a frame, each residual codebook is determined by the previous codebook, the speaker, the codebook index, the frame position and one residual "style" bit.
- Across frames, the coarse (first) codebook follows a random walk with four allowed increments, and the seven residual style bits are fixed for the whole utterance (drawn with speaker-specific probabilities).

So within-frame codebooks are strongly dependent, frames depend on the previous frame, utterance-level style must be carried through the whole utterance, and a coarse step that produces several frames must keep them mutually consistent. Producing every codebook independently matches the marginals but breaks the joint structure.

**Task.** Learn a generator from the training split (`data/synthetic_codec/train.pt`, optionally `val.pt`: 8192 and 1024 utterances with speaker ids) that samples new utterances conditioned on a speaker id. For each of the 4096 conditioning speakers in `data/synthetic_codec/test_conditions.pt`, generate one 24-frame utterance at each operating point. Generated utterances are compared, as a distribution, against reference utterances of the same speakers that the evaluator draws itself; there is no per-example target to predict.

Hard requirements:

- Generate; do not copy. Outputs that reproduce long stretches of visible utterances are failed.
- The generator must be learned from data. Writing the dataset's generating rule (or calling the data generator) into the method to produce outputs is out of scope (see `scoring/DIRECTION.md`).
- Respect the operating point you declare (`stack_factor` in the artifact): the number of coarse sequential steps per 24 frames must not exceed 24 / `stack_factor`.

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

---

# Evaluation

## What to submit

One PyTorch file per operating point in `checkpoints/predictions/`:

| File | Operating point | Experiment |
|---|---|---|
| `generator_2x.pt` | at most 12 primary steps per 24 frames (`stack_factor` 2 or more) | `codec_stack2` |
| `generator_4x.pt` | at most 6 primary steps per 24 frames (`stack_factor` 4 or more) | `codec_stack4` |

Each file is a dict with

- `pred_tokens`: integer tensor of shape `(4096, 24, 8)` with ids in `[0, 32)`; row `i` is an utterance generated for speaker `test_conditions.pt["speaker"][i]`;
- `config`: dict with `stack_factor`, a positive integer dividing 24 that declares the coarse steps used (24 / `stack_factor`); it must satisfy the row's operating point.

Nothing else in the artifact is read. Vocabulary size, codebook count, conditioning speakers, references and every score are fixed by the evaluator. Other `*_<S>x.pt` files are scored too (operating point read from the suffix) but are not ranked rows.

## Slices

The 4096 condition rows are split in two fixed halves of 2048. `scripts/evaluate_train.sh` (visible) scores the first half against reference utterances drawn from the data process with a visible seed and writes `scoring/scores_train.json`; `scripts/evaluate.sh` writes the same numbers to `scoring/scores.json`. The hidden test, `scripts/evaluate_test.sh` (no arguments), scores the other half against reference utterances it draws with a seed only the hidden evaluator holds, and writes `scoring/scores_test.json`. Both use identical metrics, so an honest generator scores alike on the two slices. Every method sees exactly the same conditions and is compared against the same references.

## Metrics (per row)

- `dependency_validity` (higher is better, primary): fraction of the codec's hard dependency checks that the generated utterances satisfy. Per utterance there are 191 checks: 168 within-frame residual-codebook constraints (7 per frame) and 23 coarse-codebook frame-to-frame transitions. Independent random tokens score about 0.07.
- `codec_fd` (lower is better): Fréchet distance (full covariance) between consecutive-frame pairs of generated and reference utterances, embedded in a fixed RVQ-style codec latent space (each frame is the sum of fixed per-codebook vectors, 32 dimensions; a pair is 64). It measures how closely the generated distribution matches real data: within-frame structure, frame-to-frame dynamics, persistence of the utterance's residual style, the speaker-specific style frequencies, and diversity (a generator that always emits the same utterance per speaker scores far worse than one that samples). Even a perfect sampler has a positive value (finite-sample floor, about 0.056 on 2048 utterances).
- `speaker_conditioned_codec_fd` (lower is better): the same distance computed separately for each conditioning speaker and averaged (speaker adherence). Its finite-sample floor is about 0.87.

Both distances are capped at the value obtained by uniformly random tokens.

## Failures

A row is **failed** and receives worst-case values on every metric (`dependency_validity` 0, both distances at their cap) if any of these holds:

- the artifact is missing (for `generator_2x` / `generator_4x`), unreadable, not a dict with an integer `pred_tokens` tensor of the exact shape, or contains ids outside `[0, 32)`;
- the declared `stack_factor` is not a positive integer dividing 24, or exceeds the row's step budget;
- more than 1% of the scored utterances share any 12-frame window (at the same position) with a visible utterance (the training and validation splits, or the train slice's reference sample);
- `dependency_validity` is below 0.5 (the output is not usable codec data).

Failures and unclipped diagnostics (validity components, the fraction of consecutive frames that keep their residual style, raw distances, copied fraction) are listed under `_failures` and `_raw` in the score files.

## Comparison rows

`baseline_parallel_1x`, `baseline_parallel_2x`, `baseline_parallel_4x` (each codebook of a stacked step predicted independently from the coarse state) and `baseline_maskgit_2x`, `baseline_maskgit_4x` (an iterative masked decoder with three refinement passes per coarse step) are reference comparison models. On the train slice they are scored from whatever is in `checkpoints/predictions/`; the hidden test always scores frozen copies held by the evaluator.

## Paper context

The paper evaluates real speech with WER, speaker similarity (SSIM), codec-space Fréchet distance and UTMOSv2 on LibriTTS, and reports relative throughput. Those numbers are kept under `_paper_reported` in `scoring/reference.json` for context only; they are not on the gym's scale.

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


---

# Reproduction Milestones

**Current: core_claim_plus**
