# Experiments

## synthetic_codec_unseen

This is the runnable reproduction experiment for the gym. It replaces the paper's unreleased Koel-TTS/NanoCodec training setup with a fixed synthetic multi-codebook codec-token dataset that preserves the key dependency structure: later codebooks in each frame depend on earlier codebooks, speaker id, and frame position.

Methods are evaluated from saved artifacts in `checkpoints/predictions/*.pt`.

Primary success criterion: a proposed generator should greatly improve `dependency_validity` over the included reference baseline while using no more than 12 primary decoder steps per 24-frame sequence. The current reproduced scores also include a stronger step-reduction setting with 6 primary decoder steps.

Reference-only LibriTTS rows in `scoring/reference.json` preserve the paper context, but they are not part of the runnable gym scoring because their data, codec pipeline, and generated audio are unavailable.
