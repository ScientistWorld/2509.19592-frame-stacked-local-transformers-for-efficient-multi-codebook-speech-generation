# Overview

This reproduction implements the core computation from "Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation": a primary autoregressive decoder, a no-LT parallel codebook baseline, autoregressive and MaskGIT local transformers, and frame stacking.

Because the paper's Koel-TTS/NanoCodec training resources are not released, the environment uses a synthetic multi-codebook codec-token dataset that preserves the central dependency structure. The validated result is that AR local-transformer decoding reaches dependency validity 1.000 at both 2x and 4x frame stacking, versus 0.045 for independent parallel prediction, while reducing primary decoder steps from 24 to 12 and 6 respectively.
