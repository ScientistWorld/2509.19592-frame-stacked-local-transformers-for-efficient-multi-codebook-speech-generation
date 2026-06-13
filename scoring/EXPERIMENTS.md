# Experiments

## synthetic_codec_unseen

This is the runnable reproduction experiment for the gym. It replaces the paper's unreleased Koel-TTS/NanoCodec training setup with a fixed synthetic multi-codebook codec-token dataset that preserves the key dependency structure: later codebooks in each frame depend on earlier codebooks, speaker id, and frame position.

Methods are evaluated from saved artifacts in `checkpoints/predictions/*.pt`.

Primary success criterion: a proposed generator should greatly improve `dependency_validity` over the included reference baseline while using no more than 12 primary decoder steps per 24-frame sequence. The current reproduced scores also include a stronger step-reduction setting with 6 primary decoder steps.

## libritts_seen_frame_stacking

Paper-reported reference-only experiment from Figure 2(g). It records the original LibriTTS seen-speaker quality comparison across 1x, 2x, and 4x frame stacking. This experiment is included in `reference.json` for context but is not runnable without the unreleased paper training data, NanoCodec pipeline, and generated audio.

## libritts_unseen_frame_stacking

Paper-reported reference-only experiment from Figure 2(g) and Section 3.3. It records the original LibriTTS unseen-speaker quality comparison across 1x, 2x, and 4x frame stacking, plus relative throughput values where Section 3.3 states them explicitly. This experiment is included in `reference.json` for context but is not runnable in this gym for the same unreleased-resource reasons.
