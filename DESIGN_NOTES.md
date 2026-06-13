Stage 2 evaluation-design handoff

Checked:
- Read `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, `eval/train/evaluate.py`, `eval/test/evaluate.py`, and `scoring/reference.json` for train/test copy-paste errors, hardcoded method defaults, paper-agreement metrics, score stub files, and method detachment.
- Confirmed the active evaluators read method prediction artifacts from `checkpoints/predictions/*.pt`; deleting or changing method-produced artifacts would change scores.
- Confirmed train and test use opposite halves of the fixed synthetic codec test split with the same partition seed and emit explicit `slice` metadata.
- Confirmed `scripts/evaluate_test.sh` writes `scoring/scores_test.json` and `scripts/evaluate_train.sh` writes `scoring/scores_train.json`.
- Could not read the paper markdown from `/home/user/scratch/AgentWorkspaces/papers/arxiv_mds/2509.19592/2509.19592.md` because that path is absent in this workspace.

Changed:
- Added `speaker_conditioned_codec_fd` to both train and test evaluators. This adds a per-speaker distributional check so a method that matches global codec statistics while collapsing speaker-conditioned structure is more likely to show a train/test gap.
- Removed the test evaluator's dependency on `scoring/scores_train.json`. Test now discovers scoreable methods directly from `checkpoints/predictions/*.pt`, matching train and avoiding stale train-score method lists.
- Made `synthetic_codec_unseen` carry effectively all scoring weight in `scoring/reference.json`, because it is the only experiment computed by the train/test artifact evaluators.
- Set the paper-reported LibriTTS experiment metric coefficients to zero and gave those experiments only validator-minimal positive weights. These remain context only; they no longer reward agreement with static paper values.
- Added required descriptions for every active metric and added `speaker_conditioned_codec_fd` to the synthetic reference result rows.

Runtime concerns for the next iteration:
- `speaker_conditioned_codec_fd` should be checked on actual artifacts to ensure each split half has enough samples per speaker and the metric does not return `NaN`.
- The synthetic split is a random half/half partition of the paper test split, not a harder OOD partition. It should detect overfitting to visible examples, but not all shortcut classes.
- The evaluator still accepts prediction artifacts with no `split` marker for backward compatibility. Runner should verify future artifacts include `split: "test"` to reduce accidental split mismatch.
- `primary_steps_per_24_frames` is derived from artifact config, so methods with incompatible configs must still emit `vocab_size`, `num_codebooks`, and `stack_factor`.

Problems found and fixed:
- Coverage mismatch: active LibriTTS metrics in `reference.json` were not emitted by the train/test evaluators.
- Stale method-list risk: held-out test could previously read method names from `scores_train.json` instead of discovering artifacts directly.
- Metric triangulation gap: global codec FD did not separately test speaker-conditioned collapse.
