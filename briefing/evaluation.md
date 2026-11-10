# Evaluation

Submit `checkpoints/predictions/generator_2x.pt` and `generator_4x.pt`. Each is a PyTorch dictionary containing `pred_tokens`, an integer tensor of shape `(16384, 24, 8)` with IDs from 0 through 31, and `config`, a dictionary containing the exact fixed `stack_factor` (2 or 4 respectively). Optional top-level `split` is accepted. Other top-level fields, including submitted metrics and verdicts, fail the artifact. Additional configuration fields are ignored except contradictory task dimensions, which fail. Generation uses speaker IDs only.

Both scripts run without arguments and score all files in `checkpoints/predictions/`. Additional `<prefix>_2x.pt` / `<prefix>_4x.pt` pairs use the same interface and are scored as additional attempts. Saved attempts may also be stored at `checkpoints/attempts/<attempt>/predictions/` with the two canonical filenames. Every attempt is rescored from raw tensors; saved scores are never inputs. Baseline filenames are reserved for evaluator-frozen comparisons. Changing those outputs or the fixed train/validation data fails the proposed run.

The visible train evaluator scores one fixed half of the conditions (8192 utterances); the hidden test scores the other half. They have independent implementations and fixtures. Both compare against four reference utterances per conditioning row, drawn from the data process. The train evaluator holds only frozen sufficient statistics (means and covariances) of its reference sample, not the utterances. The test reference sampler and seed are held only by `eval/test/`. All methods see the same speaker conditions and references. `scripts/evaluate_train.sh` writes `scoring/scores_train.json`; `scripts/evaluate_test.sh` writes `scoring/scores_test.json`, each under `experiments.<experiment>.results.<row>`.

Each experiment measures:

- `dependency_validity` ↑: fraction of 191 structural checks per utterance that pass (168 within-frame checks and 23 temporal checks).
- `codec_fd` ↓: full-covariance Fréchet distance between consecutive-frame pairs in a fixed 32-dimensional codec latent space (64 dimensions per pair).
- `speaker_conditioned_codec_fd` ↓: the same Fréchet distance computed for each conditioning speaker and averaged. This measures conditional distribution quality; it is not TitaNet speaker cosine similarity.

Distances are not clipped. Finite-sample reference distances need not be zero, even for a perfect distribution sampler.

A run fails if any of the following holds:

- either required artifact is missing or malformed, contains invalid IDs or contradictory dimensions, or claims a different operating point;
- more than 1% of scored utterances repeat an aligned 12-frame half (frames 0–11 or 12–23, all codebooks) of a frozen visible utterance. This covers complete copies, copies with local edits, and spliced halves; independently learned samples essentially never repeat a half;
- the scored utterances share 4-frame windows (all codebooks, same frame positions) with the frozen visible utterances more often than with a fresh sample of the data process of the same size and speaker composition, by more than 0.05 of all windows (`visible_window_excess`). For a generator that does not reproduce visible segments this excess has expectation 0; the reference and every tested learned variant stay at or below 0.012, while copies, memorized utterances and splices of visible segments of 4 or more frames reach 0.27 or more. The test draws its control sample at evaluation time; the train evaluator holds only 64-bit hashes of its control windows;
- more than 1% of one operating point's scored utterances repeat a 12-frame half found in any row of the other operating point's file. The two rows must be separate samples;
- total dependency validity, within-frame validity, or temporal validity is below 0.5;
- either distance is not below that of uniform random tokens;
- the fixed training data or a reserved comparison output was changed, or a method-side `data/synthetic_codec/test.pt` exists.

A failed run receives the worst value of every scored metric in both experiments: validity 0, and each distance equal to that of uniform random tokens (`_metadata.failed_run_values`). A failed or missing operating point fails the whole run. Raw diagnostics of the actual outputs remain under `_raw`, and the reason under `_failures`. Comparison rows are frozen by the evaluator. They keep their raw measurements, and when they fall below a threshold this is recorded under `_comparison_status`.

The rank adds, for each experiment and metric, the experiment weight times the absolute value of the metric's coefficient times its improvement over the reference. Direction comes from `higher_is_better`, and each improvement is divided by that metric's variation across the ranked attempts. Thus scores depend on the comparison cohort; compare attempts together. The evaluator's own `_ranking` block is a diagnostic of the same quantity for the attempts on disk. The reference measurements are recomputed from evaluator-frozen raw predictions each time.

The 2x experiment has weight 0.4; the 4x experiment has weight 0.6, emphasizing the harder aggressive operating point. Per metric, validity has coefficient 1.0, global FD 0.6, and conditional FD 0.4. Efficiency is a constraint, not a rewarded method-written number. Token artifacts cannot prove actual sequential compute; false declarations remain subject to shortcut review.

The paper's real-speech metrics remain unchanged under `_paper_reported` in `scoring/reference.json`; they are context rather than numerical targets for this synthetic task.
