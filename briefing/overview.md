# Overview

This reproduction implements the core computation from "Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation": a primary autoregressive decoder, a no-LT parallel codebook baseline, autoregressive and MaskGIT local transformers, and frame stacking.

Because the paper's Koel-TTS/NanoCodec training resources are not released, the environment uses a synthetic multi-codebook codec-token dataset that preserves the central dependency structure. The expected core result is that local-transformer decoding produces far more internally valid codebook tuples than independent parallel prediction, while frame stacking reduces primary decoder steps.
