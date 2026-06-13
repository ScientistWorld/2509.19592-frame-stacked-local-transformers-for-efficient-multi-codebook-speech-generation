# Experiments

## synthetic_codec_unseen

This is the runnable reproduction experiment for the gym. It replaces the paper's unreleased Koel-TTS/NanoCodec training setup with a fixed synthetic multi-codebook codec-token dataset that preserves the key dependency structure: later codebooks in each frame depend on earlier codebooks, speaker id, and frame position.

Methods are evaluated from saved artifacts in `checkpoints/predictions/*.pt`.

- `baseline_parallel_1x`: parallel no-local-transformer baseline with stack factor 1.
- `ar_lt_2x`: autoregressive local transformer with stack factor 2.
- `ar_lt_4x`: autoregressive local transformer with stack factor 4. This is a synthetic stretch target for `core_claim_plus`, not a paper-reported LibriTTS number.
- `maskgit_lt_2x`: MaskGIT-style local transformer with stack factor 2. This is supported by the code and can be scored if a prediction artifact exists, but it is not a synthetic reference target because the available checkpoint did not reproduce the expected effect.

Primary success criterion: local-transformer methods should greatly improve `dependency_validity` over `baseline_parallel_1x` while using no more than 12 primary decoder steps per 24-frame sequence. The 4x AR LT expansion tightens the step target to 6 primary decoder steps.

## libritts_unseen_quality

Paper-reported reference-only experiment from Figure 2(g). It records the original LibriTTS unseen-speaker quality comparison among parallel prediction, MaskGIT LT, and AR LT at stack factor 1. This experiment is included in `reference.json` for context but is not runnable without the unreleased paper training data, NanoCodec pipeline, and generated audio.

## libritts_unseen_frame_stacking

Paper-reported reference-only experiment from Figure 2(g) and Section 3.3. It records the original stack-factor-2 quality/throughput tradeoff. This experiment is included in `reference.json` for context but is not runnable in this gym for the same unreleased-resource reasons.
