# Problem

Learn to generate joint multi-codebook codec sequences conditioned on a speaker. Each utterance has 24 frames and 8 codebooks per frame, with vocabulary 32 per codebook, and belongs to one of 16 training speakers. Good outputs preserve the dependencies between the codebooks of a frame, the transitions between frames, speaker characteristics, utterance-level consistency and diversity.

The codec is a finite scalar quantizer (FSQ), like the paper's NanoCodec. Each codebook owns 3 dimensions of a 24-dimensional codec latent, and its token is one cell of a 4 x 4 x 2 grid over them. Token `c` has the digits `(c // 8, (c // 2) % 4, c % 2)` on the three dimensions. The latent value of each dimension is the cell centre, `(digit + 0.5) / levels * 2 - 1`, in [-1, 1]. All 8 codebooks of a frame are read out from the same underlying speech state, so they are strongly dependent; nothing makes them independent.

The fixed training and validation artifacts (`data/synthetic_codec/train.pt`, `val.pt`) hold 32768 and 2048 utterances with their speaker IDs. They are drawn from a synthetic speech process that stands in for the paper's unreleased 18k-hour speech mixture. The process is a hidden random nonlinear state-space model with phone-like regimes and persistent utterance style, whose latent trajectory the codec quantizes. Its parameters are not public, and its training data are the only view of it. There is no text or audio in this scaled task, and no exact-token target for a generated utterance.

Generate one new utterance for each of the 16384 speaker IDs in `data/synthetic_codec/test_conditions.pt`, at two fixed operating points:
- at most 12 primary sequential steps per utterance (2 frames per step);
- at most 6 primary steps (4 frames per step).

A second task tests speakers that the training data cover poorly. `data/synthetic_codec_fewshot/` holds an independent speech-like process with the same codec and dimensions, but 256 speakers:
- speakers 0–191 have 160 training utterances each;
- speakers 192–255 have only 8 training utterances each (validation covers speakers 0–191 only).

All of its speakers share one speaker space, so what is learned from well-covered speakers can transfer. Generate one new utterance for each of the 16384 speaker IDs in `data/synthetic_codec_fewshot/test_conditions.pt` (the few-shot speakers 192–255), at the 6-step operating point (4 frames per step). Its training data may be used together with the main task's, but the two processes differ.

Computation within a primary step is unrestricted. The scientific question is how well generation quality can be kept when the primary sequence model runs less often.

Use the provided training data to learn the generator. Outside the task:
- copying, editing or splicing visible utterances;
- calling evaluator-side samplers;
- false computation declarations.

See `scoring/CONSTRAINTS.md`.
