# Targets

Optimize `synthetic_codec_unseen` in `scoring/scores.json`.

Primary target: maximize `dependency_validity` for `ar_lt_2x` or another local-transformer method while keeping `primary_steps_per_24_frames <= 12`.

Stretch target: make `ar_lt_4x` satisfy the same dependency rule with `primary_steps_per_24_frames <= 6`.

Secondary targets: lower `codec_fd` and improve `token_accuracy`. A valid core-claim improvement must beat `baseline_parallel_1x` on dependency validity while preserving the frame-stacking step reduction; codec FD is retained as a diagnostic distribution metric but is not the primary success criterion for the synthetic dependency task.
