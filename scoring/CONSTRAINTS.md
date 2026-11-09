# Constraints

- Train on the fixed `data/synthetic_codec/train.pt` and optionally `val.pt`. Do not change the splits, dimensions, speakers or conditioning order.
- Produce new utterances learned from these data. Do not call or copy evaluator-side samplers, read test references, copy/splice visible utterances, or repair/filter tokens with an evaluator's validity rule.
- Use at most 12 primary sequential steps per 24 frames for 2x and 6 for 4x. The exact artifact declarations are 2 and 4; work within a primary step is unrestricted. A false declaration is a shortcut, even if saved tokens cannot reveal it.
- Submit the raw tensor interface in `briefing/evaluation.md`. No method-written score or verdict is accepted. Contradictory dimensions, impossible values and failed validity checks fail the run.
- Evaluation freezes comparison outputs, reference outputs, reference samples, operating points, dimensions, thresholds and test size. Reserved baseline outputs and canonical training files cannot be replaced.
- The hidden test never trains a model. It recomputes scientific measurements from raw generated outputs and ranks every submitted method through the same interface.
