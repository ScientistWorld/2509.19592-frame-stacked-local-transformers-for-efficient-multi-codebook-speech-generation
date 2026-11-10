# Problem

Learn to generate joint multi-codebook codec sequences conditioned on a speaker. Each utterance contains 24 frames, 8 codebooks per frame, vocabulary 32, and one of 16 training speakers. Good outputs preserve within-frame codebook dependencies, transitions between frames, speaker characteristics, utterance-level consistency, and diversity.

The fixed synthetic training and validation artifacts contain 8192 and 1024 utterances. They substitute for the paper's unavailable speech mixture and codec. The distribution includes stochastic temporal transitions and persistent speaker-dependent residual variation. There is no text or audio in this scaled task, and no exact-token target for a generated utterance.

Generate one new utterance for each of the 16384 speaker IDs in `data/synthetic_codec/test_conditions.pt`, at each of two fixed operating points: at most 12 primary sequential steps per utterance (2 frames per step), and at most 6 (4 frames per step). Computation within a primary step is unrestricted. The scientific question is how well quality can be preserved when the primary sequence model runs less often.

Use the provided training data to learn the generator. Copies or splices of visible utterances, hand-written or evaluator-side samplers of the data process, and false computation declarations are outside the task. See `scoring/CONSTRAINTS.md`.
