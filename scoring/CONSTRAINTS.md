# Constraints

- Train on the fixed `train.pt` (and optionally `val.pt`) of `data/synthetic_codec/` and `data/synthetic_codec_fewshot/`. Do not change the splits, dimensions, speakers or conditioning order. Evaluator fixtures are not training data.
- Produce new utterances from a generator learned from these data. Do not:
  - call or copy evaluator-side samplers, or read test references;
  - copy, edit or splice visible utterances, whole or in pieces;
  - fill rows with training frames or training segments chosen by lookup.
- Use at most 12 primary sequential steps per 24 frames for 2x, and 6 for 4x (both `generator_4x` and `generator_fewshot_4x`). The exact artifact declarations are 2, 4 and 4; work within a primary step is unrestricted. The 4x row must come from a generator that runs its primary model at most 6 times per utterance. Submitting a 2x (or unstacked) model's samples as the 4x row is a false declaration, even though saved tokens cannot reveal it.
- Submit the raw tensor interface in `briefing/evaluation.md`. No method-written score or verdict is accepted. The run fails, and receives the worst value of every scored metric, on any of:
  - contradictory dimensions or impossible values;
  - copied or reused 12-frame halves;
  - 4-frame windows matching visible utterances (on all tokens or on fixed token subsets) more often than a fresh sample of the data process, by more than 0.05;
  - a measurement not below half of its value for uniform random tokens.
- Evaluation freezes comparison outputs, reference outputs, reference samples, operating points, dimensions, thresholds and test size. Reserved baseline outputs and canonical training files cannot be replaced.
- The hidden test never trains a model. It recomputes scientific measurements from raw generated outputs and scores every submitted method through the same interface.
