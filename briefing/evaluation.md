# Evaluation

## Interface

Submit three files in `checkpoints/predictions/`:
- `generator_2x.pt` and `generator_4x.pt`, for the main task's conditions;
- `generator_fewshot_4x.pt`, for the few-shot task's conditions.

Each is a PyTorch dictionary with:
- `pred_tokens`: an integer tensor of shape `(16384, 24, 8)` with IDs from 0 through 31, one row per conditioning speaker in file order;
- `config`: a dictionary holding the exact fixed `stack_factor` (2, 4 and 4 respectively).

An optional top-level `split` is accepted. Any other top-level field, including submitted metrics and verdicts, fails the artifact. Other configuration fields are ignored, except task dimensions that contradict the fixed task, which fail. Generation uses speaker IDs only.

Both scripts run without arguments and score all files in `checkpoints/predictions/`:
- additional `<prefix>_2x.pt` / `<prefix>_4x.pt` / `<prefix>_fewshot_4x.pt` triples use the same interface and are scored as additional attempts;
- saved attempts may also be stored at `checkpoints/attempts/<attempt>/predictions/` with the three canonical filenames.

Every attempt is rescored from raw tensors; saved scores are never inputs. Baseline filenames are reserved for evaluator-frozen comparisons. Changing those outputs, or the fixed train/validation data, fails the proposed run.

## Slices

The visible train evaluator scores one fixed half of each task's conditions (8192 utterances); the hidden test scores the other half. For the main task the halves are a fixed random split of the rows. For the few-shot task they are the rows of disjoint speaker halves: speakers 192–223 for train and 224–255 for test. The two evaluators have independent implementations and fixtures. Both compare against four reference utterances per conditioning row, drawn from the task's data process.
- The train evaluator holds only frozen sufficient statistics of its reference sample (means, covariances and token-pair counts), not the utterances.
- The test reference sampler and its seed are held only by `eval/test/`.

All methods see the same speaker conditions and references. `scripts/evaluate_train.sh` writes `scoring/scores_train.json` and `scripts/evaluate_test.sh` writes `scoring/scores_test.json`, each under `experiments.<experiment>.results.<row>`.

## Measurements

There are three experiments: `codec_stack2` (row `generator_2x`), `codec_stack4` (`generator_4x`) and `codec_fewshot_stack4` (`generator_fewshot_4x`). Each measures three quantities; lower is better for all three.

- `codec_fd`: full-covariance Fréchet distance in the codec's embedding space (the public FSQ latent of `briefing/problem.md`), between generated and reference consecutive-frame pairs (48 dimensions per pair).
- `codebook_pair_divergence`: mean Jensen–Shannon divergence, in bits, between the generated and reference joint token distributions of codebook pairs. It covers all 28 within-frame pairs and all 64 pairs of (codebook at frame t, codebook at frame t+1). It detects implausible codebook combinations that second moments can miss.
- `speaker_conditioned_codec_fd`: the codec FD computed for each conditioning speaker of the slice and averaged (16 main speakers; 32 few-shot speakers per slice). This measures conditional distribution quality; it is not TitaNet speaker cosine similarity.

Nothing is clipped. Finite-sample values are not zero even for a perfect sampler of the data process.

## Failure

A run fails if any of the following holds:

- **Artifact.** Either required artifact is missing or malformed, contains invalid IDs or contradictory dimensions, or claims a different operating point.
- **Copied halves.** More than 1% of scored utterances repeat an aligned 12-frame half (frames 0–11 or 12–23, all codebooks) of a frozen visible utterance.
- **Visible windows** (`visible_window_excess`). This compares how often the scored utterances' 4-frame windows match the frozen visible utterances with how often they match a fresh sample of the data process of the same size and speaker composition. A window matches if it agrees, at the same frame positions, on all 32 tokens or on any of 8 fixed subsets of 12 of them. The run fails when the visible match rate exceeds the fresh-sample rate by more than 0.05 of all windows.
  - Copies with scattered edits still match, so the test covers copies, edited copies and splices.
  - For a generator that does not reproduce visible segments, the excess has expectation 0.
  - The test draws its fresh sample at evaluation time. The train evaluator holds only the 48-bit keys of its fresh sample's windows.
- **Reused operating point.** More than 1% of one main operating point's scored utterances repeat a 12-frame half found in any row of the other operating point's file. The two rows must be separate samples.
- **Not better than noise.** Any of the three measurements is not below half of its value for uniform random tokens.
- **Integrity.** The fixed training data or a reserved comparison output was changed, or a method-side `data/synthetic_codec/test.pt` exists.

Copy and window checks use each task's own visible utterances and fresh sample. A failed run receives the worst value of every scored metric in all three experiments: each measurement equals that of uniform random tokens (`_metadata.failed_run_values`). A failed or missing row fails the whole run.
- Raw diagnostics of the actual outputs stay under `_raw`, and the reason under `_failures`.
- Comparison rows are frozen by the evaluator. They keep their raw measurements, and any failed check is recorded under `_comparison_status`.

## Rank

The rank sums over experiments and metrics: experiment weight × |metric coefficient| × improvement over the reference. Direction comes from `higher_is_better`. Each improvement is divided by that metric's variation across the ranked attempts, so scores depend on the comparison cohort and attempts should be compared together. The evaluator's own `_ranking` block is a diagnostic of the same quantity for the attempts on disk. The reference measurements are recomputed each time from evaluator-frozen raw predictions.

Weights and coefficients:
- `codec_stack2` has weight 0.3, `codec_stack4` 0.4 and `codec_fewshot_stack4` 0.3. The aggressive 4x point carries the most weight, in total 0.7 across the two tasks.
- Per metric: codec FD 1.0, codebook-pair divergence 0.8, speaker-conditioned FD 0.6.

Efficiency is a constraint, not a rewarded method-written number. Token artifacts cannot prove actual sequential compute, so false declarations remain subject to shortcut review.

The paper's real-speech metrics remain unchanged under `_paper_reported` in `scoring/reference.json`. They are context, not numerical targets for this synthetic task.
