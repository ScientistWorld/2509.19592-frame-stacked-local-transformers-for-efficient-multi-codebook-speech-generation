# Overview

This gym implements the core computation of "Frame-Stacked Local Transformers for Efficient Multi-Codebook Speech Generation": an autoregressive primary decoder over (stacked) frames, a parallel-head baseline, autoregressive and MaskGIT local transformers (LTs), and frame stacking.

The paper's Koel-TTS/NanoCodec data and models are unreleased, so the gym uses a synthetic codec-token process with strong within-frame codebook dependencies, frame-to-frame dynamics and utterance-level residual style. Generators are compared at two fixed operating points, 12 and 6 primary steps per 24 frames (2x and 4x stacking), by dependency validity and codec-latent Fréchet distances against reference utterances drawn by the hidden evaluator. The reference method is the AR LT; parallel heads (1x/2x/4x) and the MaskGIT LT (2x/4x) are frozen comparison rows. Results of the reference run are summarized in `HARDENING.md` and `scoring/scores_test.json`.
