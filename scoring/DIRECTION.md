# Direction

Future work should stay within the paper's problem family: multi-codebook token generation with explicit modeling of within-frame codebook dependencies and reduced primary generation steps.

Acceptable changes include better sequence models, dependency-modeling mechanisms, sampling strategies, and training objectives that preserve the same multi-codebook conditional generation setup.

Out-of-scope changes include replacing the task with a classifier, bypassing generation by applying the known codec rule as a post-processing repair, or changing the dataset so independent codebook prediction becomes trivial.
