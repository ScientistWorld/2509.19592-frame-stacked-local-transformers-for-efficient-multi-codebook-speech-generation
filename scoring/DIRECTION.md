# Direction

Future work should stay within the paper's approach family: multi-codebook token generation with local dependency modeling and frame stacking.

Acceptable changes include better primary decoders, stronger local transformers, improved MaskGIT schedules, sampling strategies, and training objectives that preserve the same multi-codebook autoregressive setup.

Out-of-scope changes include replacing the task with a classifier, bypassing generation by applying the known codec rule as a post-processing repair, or changing the dataset so independent codebook prediction becomes trivial.
