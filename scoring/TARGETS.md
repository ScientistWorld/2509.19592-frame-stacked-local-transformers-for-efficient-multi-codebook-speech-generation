# Targets

Optimize the two scored experiments of the hidden test (`scoring/scores_test.json`; the visible train slice in `scoring/scores_train.json` / `scoring/scores.json` measures the same thing on the other half of the conditions).

- `codec_stack4` (weight 0.6), row `generator_4x`: at most 6 primary (coarse) steps per 24-frame utterance.
- `codec_stack2` (weight 0.4), row `generator_2x`: at most 12 primary steps per 24-frame utterance.

Per row, in order of importance:

1. Primary target: raise `dependency_validity` (fraction of the codec's within-frame residual-chain checks and coarse frame-to-frame transitions that the generated utterances satisfy). A row below 0.5 is failed and receives worst-case values on every metric, whatever its distances.
2. Lower `codec_fd` (coefficient 0.6) and `speaker_conditioned_codec_fd` (coefficient 0.4): distribution distances of consecutive-frame pairs to hidden reference utterances, globally and per conditioning speaker. They reward diverse, temporally coherent and speaker-faithful generation, not only valid tuples.

The reference to beat is this gym's reproduction of the paper's autoregressive local transformer at each operating point (values in `scoring/reference.json`, `results.generator_2x` / `results.generator_4x`). The reference is well below the bound on all three metrics at both operating points: it does not carry the coarse codebook's frame-to-frame transitions or the utterance-level residual style reliably, especially across coarse-step boundaries at 4x.
