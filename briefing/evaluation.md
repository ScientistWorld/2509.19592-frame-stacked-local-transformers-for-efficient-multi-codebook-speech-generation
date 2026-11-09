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
