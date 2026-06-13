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
