# Targets

Optimize `synthetic_codec_unseen` in `scoring/scores.json`.

Primary target: maximize `dependency_validity` for any valid generator while keeping `primary_steps_per_24_frames <= 12`.

Stretch target: satisfy the same dependency rule with `primary_steps_per_24_frames <= 6`.

Secondary targets: lower `codec_fd` and `speaker_conditioned_codec_fd`, and improve `token_accuracy`. A valid core-claim improvement must beat the included reference baseline on dependency validity while preserving the generation-step reduction; codec FD metrics are retained as diagnostic distribution metrics but are not the primary success criterion for the synthetic dependency task.
