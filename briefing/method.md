# Method

The paper studies local transformers (LTs) for multi-codebook speech generation. The baseline predicts all codebooks of a frame in parallel from the primary decoder state, assuming conditional independence across codebooks. The proposed models keep an autoregressive primary decoder over time and add a small LT that expands each primary hidden state into the frame's codebook tokens. With frame stacking, the primary decoder runs once per `S` frames and the LT decodes all `S x N` codebook tokens of the stack.

Implementation (`method/models.py`, scaled down: width 96, 4 heads, 2 primary layers, 1 LT layer, 10000 AdamW steps at a constant learning rate of 1e-3, batch 128; the paper specifies AdamW but no schedule, and `method/train.py --lr-schedule cosine` is an optional variant that the reference does not use):

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
