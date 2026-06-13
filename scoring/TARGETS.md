# Targets

Optimize `synthetic_codec_unseen` in `scoring/scores.json`.

Primary target: maximize `dependency_validity` for `ar_lt_2x` or another local-transformer method while keeping `primary_steps_per_24_frames <= 12`.

Secondary targets: lower `codec_fd`, lower `nll`, and improve `token_accuracy`. A valid improvement should beat `baseline_parallel_1x` on dependency validity and codec FD.
