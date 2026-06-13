# Overview

This reproduction implements the core computation from "Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation": a primary autoregressive decoder, a no-LT parallel codebook baseline, autoregressive and MaskGIT local transformers, and frame stacking.

Because the paper's Koel-TTS/NanoCodec training resources are not released, the environment uses a synthetic multi-codebook codec-token dataset that preserves the central dependency structure. The validated result is that AR local-transformer decoding reaches dependency validity 1.000 at both 2x and 4x frame stacking, versus 0.045 for independent parallel prediction, while reducing primary decoder steps from 24 to 12 and 6 respectively.

---

# Problem Definition

Speech generation systems that operate on neural audio codec tokens must predict a matrix of discrete tokens with shape `(time, codebook)`. At each acoustic frame, multiple codebooks jointly describe the same audio frame. A successful generator should produce codebook combinations that are accurate and internally consistent, not just plausible marginal tokens.

The gym version of the problem uses a reproducible synthetic codec-token dataset because the paper's 18k-hour Koel-TTS training mixture, NanoCodec tokenizer, and trained models are not released. Each sequence contains a speaker id and 24 frames with 8 codebooks per frame. Later codebooks are deterministic residual-style functions of earlier codebooks, the speaker, and frame position. This creates the same evaluation pressure as multi-codebook audio generation: independent per-codebook sampling can match marginals while producing invalid joint codebook tuples.

The task is to learn a conditional autoregressive codec-token generator that maps previous frames and speaker identity to future multi-codebook frames. A future method may change the model internals, but it must use the same train/validation/test splits and produce `checkpoints/predictions/*.pt` files containing generated tokens, references, speakers, and model metadata.

---

# Method

The paper studies local transformers for multi-codebook speech generation. The baseline predicts all codebooks at an acoustic frame in parallel, assuming conditional independence across codebooks. The proposed models keep a primary autoregressive decoder over time but add a small local transformer that expands each primary hidden state into the frame's codebook tokens.

Implemented variants:

- `baseline_parallel_1x`: no local transformer. The primary decoder predicts all codebook slots in a frame with independent output heads.
- `ar_lt_2x`: autoregressive local transformer. It predicts the stacked frame/codebook slots sequentially, conditioning each slot on the primary hidden state and previously generated slots.
- `maskgit_lt_2x`: MaskGIT-style local transformer. It starts with masked slots, uses non-causal self-attention, and fills high-confidence slots over iterative passes.

Frame stacking is implemented by grouping `S` consecutive frames into one primary decoder step. For stack factor 2, the primary decoder produces 12 steps for a 24-frame sequence, and the local transformer decodes `2 x 8` codebook slots at each step.

Deviations from the paper:

- The original Koel-TTS text/audio training stack is replaced by a synthetic codec-token generator because the data and model release are unavailable.
- Model width and layer count are reduced so the method can run inside this research gym.
- The evaluator uses token validity and codec embedding FD rather than WER/UTMOS/TitaNet speaker similarity, which require unreleased generated audio and large external models.

---

# Evaluation

The evaluator reads prediction artifacts from `checkpoints/predictions/*.pt` and writes `scoring/scores.json`. It does not import or execute method code.

Primary metric:

- `dependency_validity`: fraction of non-first codebook tokens that satisfy the held-out codec dependency rule. Higher is better.

Additional metrics:

- `token_accuracy`: exact token accuracy against the test split. Higher is better.
- `codec_fd`: diagonal Fréchet distance between deterministic 32-dimensional codec-frame embeddings of generated and reference frames. Lower is better.
- `primary_steps_per_24_frames`: number of primary decoder steps needed to produce 24 frames. Lower is better and replaces wall-clock speed as a hardware-independent efficiency metric.

The paper's original audio metrics are WER, speaker similarity, codec-space FD, UTMOSv2, and relative throughput on LibriTTS. Those paper-reported numbers are preserved in `scoring/reference.json` under the `libritts_*` experiments. The reproduced synthetic experiment has its own `synthetic_codec_unseen` reference group for validator compatibility and is not numerically comparable to LibriTTS audio quality.

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


---

# Reproduction Milestones

**Current: core_claim_plus**
