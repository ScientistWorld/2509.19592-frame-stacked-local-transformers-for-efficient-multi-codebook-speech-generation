# Method

The paper studies local transformers (LTs) for multi-codebook speech generation.
- **Baseline:** predicts all codebooks of a frame in parallel from the primary decoder state, assuming the codebooks are conditionally independent.
- **Proposed models:** keep an autoregressive primary decoder over time and add a small LT that expands each primary hidden state into the frame's codebook tokens.
- **Frame stacking:** the primary decoder runs once per `S` frames, and the LT decodes all `S x N` codebook tokens of the stack.

## Implementation

The code is in `method/models.py`, scaled down from the paper:
- width 96, 4 heads, 2 primary layers and 1 LT layer;
- 10000 AdamW steps at a constant learning rate of 1e-3, batch 128.

The paper specifies AdamW and the Koel-TTS setup. Koel-TTS trains at 1e-4, annealed by 0.998 every 1000 steps, which keeps the rate almost constant over its 200k steps. The reference keeps that constant shape at this scale. `method/train.py --lr-schedule cosine` and `--speaker-rank r` (a low-rank speaker table shared by all speakers) are optional variants that the reference does not use.

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
