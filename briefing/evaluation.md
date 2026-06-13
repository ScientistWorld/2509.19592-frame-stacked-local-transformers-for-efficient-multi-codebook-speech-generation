# Evaluation

The evaluator reads prediction artifacts from `checkpoints/predictions/*.pt` and writes `scoring/scores.json`. It does not import or execute method code.

Each prediction artifact is a PyTorch file with:

- `pred_tokens`: integer tensor of shape `(num_examples, 24, 8)`.
- `config`: dictionary containing at least `vocab_size`, `num_codebooks`, and `stack_factor`.
- `split`: the evaluated split name, normally `test`.

References and speaker ids are loaded by the evaluator from the fixed dataset, not trusted from method outputs.

Primary metric:

- `dependency_validity`: fraction of non-first codebook tokens that satisfy the held-out codec dependency rule. Higher is better.

Additional metrics:

- `token_accuracy`: exact token accuracy against the test split. Higher is better.
- `codec_fd`: diagonal Fréchet distance between deterministic 32-dimensional codec-frame embeddings of generated and reference frames. Lower is better.
- `speaker_conditioned_codec_fd`: average per-speaker version of `codec_fd`, included to catch methods that match global token statistics while collapsing speaker-conditioned structure. Lower is better.
- `primary_steps_per_24_frames`: number of primary decoder steps needed to produce 24 frames. Lower is better and replaces wall-clock speed as a hardware-independent efficiency metric.

The paper's original audio metrics are WER, speaker similarity, codec-space FD, UTMOSv2, and relative throughput on LibriTTS. Those paper-reported numbers are preserved in `scoring/reference.json` under the `libritts_*` experiments. The reproduced synthetic experiment has its own `synthetic_codec_unseen` reference group for validator compatibility and is not numerically comparable to LibriTTS audio quality.
