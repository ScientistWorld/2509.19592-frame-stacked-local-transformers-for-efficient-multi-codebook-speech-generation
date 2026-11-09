# Constraints

- Train only on the fixed dataset produced by `data/generate_synthetic.py` (via `scripts/download.sh`): `train.pt` and optionally `val.pt`. Do not change the codec process, split seeds, vocabulary size (32), sequence length (24), number of codebooks (8), number of speakers (16) or the evaluation conditions (`test_conditions.pt`, 4096 speakers).
- Submit `checkpoints/predictions/generator_2x.pt` and `generator_4x.pt` in the format of `briefing/evaluation.md`. The generator behind each file must use at most 24 / `stack_factor` sequential primary (coarse) steps per 24-frame utterance, with the declared `stack_factor` at least 2 and 4 respectively. Work inside a coarse step (a local decoder over the stacked codebooks, refinement passes) is not limited. The evaluator cannot verify the declared step count from tokens; declaring fewer steps than the generator actually uses is a shortcut.
- Generate new utterances. Outputs sharing 12-frame windows with visible utterances (training/validation splits, the train slice's reference sample) are failed.
- The generator must be learned. Do not produce outputs from the dataset's generating rule (or by calling `data/generate_synthetic.py`), and do not repair or filter generated tokens with the known rule.
- Do not score wall-clock time; efficiency is the fixed operating point (primary steps per 24 frames).
- The evaluators score saved prediction artifacts only; they must not import `method/` or regenerate method outputs. Scores, flags or metrics written into artifacts are ignored.
