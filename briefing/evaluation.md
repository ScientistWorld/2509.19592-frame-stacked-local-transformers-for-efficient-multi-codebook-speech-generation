# Evaluation

Submit `checkpoints/predictions/generator_2x.pt` and `generator_4x.pt`. Each is a PyTorch dictionary containing `pred_tokens`, an integer tensor of shape `(16384, 24, 8)` with IDs from 0 through 31, and `config`, a dictionary containing the exact fixed `stack_factor` (2 or 4 respectively). Optional top-level `split` is accepted. Other top-level fields, including submitted metrics and verdicts, fail the artifact. Additional configuration fields are ignored except contradictory task dimensions, which fail. Generation uses speaker IDs only.

Both scripts run without arguments and score all files in `checkpoints/predictions/`. Additional `<prefix>_2x.pt` / `<prefix>_4x.pt` pairs use the same interface and are ranked as additional attempts. Saved attempts may also be stored at `checkpoints/attempts/<attempt>/predictions/` with the two canonical filenames. Every attempt is rescored from raw tensors; saved scores are never ranking inputs. Baseline filenames are reserved for evaluator-frozen comparisons. Changing those outputs or the fixed train/validation data fails the proposed run.

The visible train evaluator scores one fixed half of the conditions (8192 utterances); the hidden test scores the other half. They have independent implementations, reference samples, and fixtures. Both compare against four reference utterances per conditioning row. The test reference sampler and seed are held only by `eval/test/`. All methods see the same speaker conditions and references. `scripts/evaluate_train.sh` writes `scoring/scores_train.json`; `scripts/evaluate_test.sh` writes `scoring/scores_test.json`.

Each experiment measures:

- `dependency_validity` ↑: fraction of 191 structural checks per utterance that pass (168 within-frame checks and 23 temporal checks).
- `codec_fd` ↓: full-covariance Fréchet distance between consecutive-frame pairs in a fixed 32-dimensional codec latent space (64 dimensions per pair).
- `speaker_conditioned_codec_fd` ↓: the same Fréchet distance computed for each conditioning speaker and averaged. This measures conditional distribution quality; it is not TitaNet speaker cosine similarity.

Distances are not clipped. Finite-sample reference distances need not be zero, even for a perfect distribution sampler. Raw diagnostic components remain available for failed runs, so comparisons still show their scientific behavior.

A run fails if either required artifact is missing or malformed, contains invalid IDs or contradictory dimensions, claims a different operating point, duplicates more than 1% complete visible utterances, or scores below 0.5 on total dependency validity, within-frame validity, or temporal validity. Duplicate checks use frozen canonical visible data. The former 12-frame window rule was removed because it rejected stronger learned models. A method must still avoid copying or splicing training data; partial-copy detection is left to the separate shortcut review. A method-side `data/synthetic_codec/test.pt` is forbidden.

Ranking is lexicographic: every feasible attempt ranks above every failed attempt. Within a tier, add each experiment weight times the absolute value of each metric's coefficient times its improvement over the reference, using `higher_is_better` for the direction, divided by that metric's population standard deviation across the attempts currently ranked and the reference. A zero-variance metric contributes zero. Standard deviations and per-metric contributions are emitted under `_ranking`. Thus scores change when the comparison cohort changes; compare attempts together. The reference measurements are recomputed from evaluator-frozen raw predictions each time. Invalid artifacts get finite default measurements and an explicit failure, never a high rank.

The 2x experiment has weight 0.4; the 4x experiment has weight 0.6, emphasizing the harder aggressive operating point. Per metric, validity has coefficient 1.0, global FD 0.6, and conditional FD 0.4. Efficiency is a constraint, not a rewarded method-written number. Token artifacts cannot prove actual sequential compute; false declarations remain subject to shortcut review.

The paper's real-speech metrics remain unchanged under `_paper_reported` in `scoring/reference.json`; they are context rather than numerical targets for this synthetic task.
