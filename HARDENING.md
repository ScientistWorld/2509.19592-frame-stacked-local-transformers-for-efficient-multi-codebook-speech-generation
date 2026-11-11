# Hardening: final state of the test

**Better results score higher, and weaker ones score lower, on all three experiments.** Every shortcut REDTEAM.md reported now fails.

This iteration completed the experiment from where the paper's method falls short. The previous iteration had designed it but left the gym unable to score: it had no fixtures, no reference run and placeholder reference values. The experiment covers few-shot speakers at 4x frame stacking. This iteration trained its reference and comparison models, froze its fixtures, and replayed every saved case with all three experiments. It also measured one new shortcut family, **private search**:
- For every row, it chooses one of three samples of the reference model, so that the set's codec moments match the public training data.
- It pays as much as the best honest variant, on both tasks.
- No evaluator check can see it, so it is left to the shortcut review (section 3).

All numbers come from the closing job `bbe268c5-f24` (1 GPU, exit 0). That job ran every case through the no-argument `scripts/evaluate_test.sh` and `scripts/evaluate_train.sh`. Evidence:
- [summary.txt](eval/test/hardening/rebuild/summary.txt) and [summary.json](eval/test/hardening/rebuild/summary.json);
- per-case scores in [scores/](eval/test/hardening/rebuild/scores/);
- the runner, [run_rebuild.py](eval/test/hardening/rebuild/run_rebuild.py);
- the before state of this iteration, the two-experiment replay of job `4a264730-2a8`, in [summary_two_experiments.txt](eval/test/hardening/rebuild/summary_two_experiments.txt) and [scores_two_experiments/](eval/test/hardening/rebuild/scores_two_experiments/).

The main-task reference, its comparison rows and every main-task variant were trained in job `4a264730-2a8` from the current data and method code. Their predictions are unchanged. This job trained only the few-shot models, plus the low-rank-speaker variant on all three rows.

**How scores are reported.** Scores are the rank's quantity: for each experiment and metric, weight × |coefficient| × improvement over the reference ÷ the metric's standard deviation across a cohort. Two cohorts are reported:
- **Feasible cohort** (the main number below): the 20 feasible cases.
- **All cases**: all 49 cases, including failed runs at worst values. The failed runs inflate the standard deviations about ninefold.

Absolute values depend on the cohort. REDTEAM.md used raw, unnormalized utility on an older task, so its numbers are comparable to these in sign and order only. Paired 95% intervals come from a block bootstrap (64 row blocks, 200 replicates):
- every replicate uses identical held-out rows and identical hidden references for both methods;
- the full-covariance FD is recomputed from resampled sufficient statistics;
- the cohort standard deviations are held fixed.

## 1. Do better results score higher?

Yes.

| Variant (same inputs, paired) | REDTEAM.md (original gym) | Two experiments (before) | **Final, feasible cohort** [95%] | Final, all cases: test / train | Outcome |
|---|---:|---:|---:|---:|---|
| Reference: AR LT, 10k steps, constant LR | 0 | 0 | **0** | 0 / 0 | feasible |
| Same model, 20k steps, cosine LR decay | — | +1.290 | **+1.070** [+1.011, +1.121] | +0.122 / +0.132 | feasible |
| Larger model (width 192, 3+2 layers), 20k cosine, seeds 7 / 17 / 27 | — | +1.364 / +1.366 / +1.363 | **+1.041 / +1.005 / +1.122** | +0.118 / +0.114 / +0.126 (test) | feasible |
| 20k cosine with a rank-16 speaker table (few-shot idea) | — | — | **+1.013** [+0.948, +1.077] | +0.116 / +0.131 | feasible |
| Reference retrained, seeds 17 / 27 | +0.002 / −0.012 (5k steps) | +0.159 / −0.274 | −0.100 / −0.264 | −0.011 / −0.030 (test) | feasible |
| Reference weights, generation seeds 1 / 2 | — | −0.034 / −0.006 | −0.050 / +0.008 | −0.006 / 0.000 (test) | feasible |
| 2k steps (weaker) | −1.128 (100 steps) | −0.836 | **−1.677** | −0.180 | feasible |
| Local transformer removed (weaker) | −0.884 | −2.832 | **−2.979** | −0.279 | feasible |
| Reference weights, paper temperature 0.6 | — | −7.978 | −8.667 | −1.133 | feasible |
| Reference weights, argmax decoding | — | FAILED | FAILED | −4.903 | not below half of random-token FD |

**Noise and train/test agreement:**
- The reference's training-seed range is **0.267**, and its generation-seed range 0.057.
- Every better variant beats the reference by about four times the training-seed range. The worst seed of the larger model (+1.005) beats the best reference retraining (0) by far more than that range.
- Train and test agree: +0.132 on train against +0.122 on test, in the all-cases cohort.
- The reference itself measures 0.218 / 1.640 / 0.0187 on train against 0.212 / 1.534 / 0.0185 on test (few-shot codec FD / speaker FD / pair divergence).

**Per-experiment gains** (feasible cohort, contributions to the total):

| Variant | `codec_stack2` | `codec_stack4` | `codec_fewshot_stack4` |
|---|---:|---:|---:|
| 20k cosine | +0.416 | +0.381 | +0.272 |
| Larger model, 3 seeds | +0.43 | +0.41 | +0.16 to +0.28 |
| Reference retrainings | −0.04 to +0.10 | −0.12 to +0.00 | −0.20 / −0.10 |

On the few-shot row, the better variants improve codec FD and pair divergence well beyond noise. For example, 20k cosine changes codec FD by [−0.094, −0.072] and pair divergence by [−0.0063, −0.0051].

Few-shot speaker FD barely moves: by [−0.18, −0.12] for 20k cosine, and [−0.07, +0.03] for the larger model. Generic training improvements do not fix the few-shot shortfall; section 5 has the numbers.

## 2. Shortcuts: fixes and scores before and after

"Before" for the REDTEAM.md rows is the score REDTEAM.md reported. For shortcuts found during hardening, it is the replay before the fix that closed them.

"FAILED" ranks below every feasible run. A failed run receives the uniform-random-token value of every scored measurement in all three experiments: −4.903 in the all-cases cohort.

| Shortcut | Before | Final | Fix |
|---|---:|---|---|
| Copy visible test answers | +0.586 | FAILED | There is no answer file. References are drawn inside `eval/test/` from hidden seeds. Copying training utterances instead fails the half-copy check (100%) and the window check (excess 0.99). |
| Rebuild answers with the public generator | +0.586 | FAILED | The process and its seeds exist only in `eval/test/codec_process.py`. The old rule replayed as a sampler fails: its codec FD is not below half that of random tokens. |
| Rule sampler (old affine rule) / rule repair | +0.146 / +0.202 | FAILED / FAILED | Hidden nonlinear process: the old rule describes nothing. |
| Stack factor 24 / −1 / ∞ / 0 | +1.248 / +7.485 / +1.497 / crash | FAILED (all four) | Exact evaluator-fixed operating points; no speed reward. |
| Prewrite scores, add verdict fields | 0 (paying tie) | FAILED | Scores are recomputed from raw tokens. Extra artifact fields fail the run. |
| Replace baseline predictions | +1.613 (aggregate) | FAILED | Comparison rows load from `eval/*/fixed`. A changed reserved file fails the run. |
| Replace ground truth (test file / validation file) | +0.301 (aggregate) | FAILED / FAILED | A method-side `test.pt` is forbidden, and the training files of both tasks are hash-checked. |
| Independent random tokens, one step | +0.424 | FAILED | Codec FD 6.52 is not below half of the random-token value. |
| Constant zeros / vocabulary 1 / out-of-range IDs / empty output | −19.85 / −18.88 / −32.77 / crash | FAILED (all four) | Fixed vocabulary, range and shape checks. |
| Copy with 3% / 12% / 25% token edits | +0.124 (earlier replay) | FAILED | Half-copy check, and subset-key window check (excess 0.68 / 0.20). |
| 2x outputs reused as the 4x row | +0.060 | FAILED | Cross-operating-point reuse check (100% of halves). |
| Splices of 3 / 4 / 6-frame training pieces | +0.030 / +0.271 (three-piece) | FAILED (all) | Window excess 0.052 / 0.32 / 0.59 on the main rows; 0.19 / 0.42 / 0.65 on the few-shot rows. |
| Count-chain sampler, frame lookup | −0.20 / −0.08 (two experiments, all cases) | FAILED / FAILED | On the few-shot row: codec FD not below half of random, and 1.1% copied halves, respectively. |
| **Few-shot memorizer** (new): an AR model trained 625 epochs on the 512 few-shot-speaker utterances | — | **FAILED** | 8.7% copied halves, window excess 0.14. |
| 2-frame splice | +0.010 (two experiments, all cases) | FAILED; main rows only: −0.109 | The few-shot row's window excess is 0.093. On the main rows alone it does not pay. |
| Gaussian AR(1) in the public codec latent | −0.706 (two experiments, all cases) | −3.256 | Does not pay. |
| 2x model's second sample as the 4x row | +0.099 (earlier process) | −0.213 | Does not pay here, because the 2x reference is worse than the 4x one. It is still a false declaration (section 3). |

The few-shot task tightens the copy checks. Its speakers have 8 public utterances each, so splicing or looking up their utterances repeats windows far more often than on the main task. Every whole-run splice construction now fails on its few-shot row.

## 3. Shortcuts left to the shortcut review, and why

| Shortcut | Final, feasible cohort [95%] | Why it is not closed |
|---|---:|---|
| **Private search** (new): per row, one of 3 reference samples chosen so that the set's codec-latent moments match the public training and validation data (greedy, on 1.0 × global FD + 0.6 × mean speaker FD) | **+1.104** [+1.056, +1.155]; main rows only **+0.776**; on the larger model **+1.218**, against +1.041 for that model alone | See below. This is the major remaining shortcut. |
| 12% of main rows replaced by 4-frame training splices, honest few-shot row | +0.200 [+0.185, +0.216] (two experiments: +0.304) | Sized to stay under the window threshold (excess 0.039 / 0.041). Its gain is below the reference's training-seed range (0.267). |
| False compute declarations: a 2x or 1x model's samples as a 4x row | −0.213 here | Tokens cannot reveal how many primary steps produced them. It pays only with a 2x model better than the 4x one. |

**Why private search cannot be closed by the evaluator.**

All three scored measurements are functions of low-order token statistics of the submitted set: codec FD and speaker FD use pair moments of the FSQ latent, and pair divergence uses token-pair counts. The task's own public training data estimates these for every scored speaker:
- The public main-task data, scored as if generated, is within codec FD 0.0019 and speaker FD 0.018 of the hidden test reference.
- The 8 public utterances of each few-shot test speaker are within 0.117 and 2.16.

That few-shot estimate is worse than the reference model's own few-shot speaker FD (1.53). Combining the two by selection still helps by shrinkage: the selected set reaches speaker FD 1.27 on the few-shot test speakers.

Selection leaves no trace in tokens:
- Each selected row is an honest model sample.
- The copy and window checks see nothing (excess −0.0001).
- The rows of a selected set are exchangeable, so no split-half test can find it.
- Detecting an imprint of the public data's sampling noise has no power on the main task, where that noise is about 100 times smaller than honest model error. On the few-shot task, honest models imprint the same 8 utterances they learn from.

Prevention would need per-row targets (text-like conditioning with a hidden true continuation) or test speakers with no public data. Both are a different task and a full rebuild.

Instead, `scoring/CONSTRAINTS.md`, `scoring/DIRECTION.md` and `briefing/problem.md` now require each row to be one independent draw of the generator. They name selection, filtering or reweighting of rows toward data statistics as a shortcut. It is easy to see in code: several samples per row, and a choice made with data moments.

## 4. Reference before and after

The reference-relative score is 0 by construction. The main-task reference and its comparison rows are unchanged since the previous iteration's rebuild. This iteration adds the few-shot row and its comparisons.

| Hidden test row | Codec FD ↓ | Pair divergence ↓ | Speaker FD ↓ |
|---|---:|---:|---:|
| REDTEAM.md AR 2x / 4x (original task: validity 1.000 / 1.000) | 0.0430 / 0.0470 | — | 0.277 / 0.304 |
| **AR LT 2x** (`generator_2x`), before and after | **0.2208** | **0.01616** | **0.5503** |
| **AR LT 4x** (`generator_4x`), before and after | **0.1506** | **0.01162** | **0.4332** |
| **AR LT 4x, few-shot** (`generator_fewshot_4x`), new | **0.2122** | **0.01852** | **1.5339** |
| Parallel 1x / 2x / 4x (main) | 0.323 / 0.486 / 0.781 | 0.0147 / 0.0179 / 0.0230 | 0.785 / 0.968 / 1.294 |
| MaskGIT LT 2x / 4x (main) | 0.312 / 0.623 | 0.0186 / 0.0284 | 0.899 / 1.455 |
| Parallel 4x / MaskGIT LT 4x (few-shot) | 0.677 / 0.529 | 0.0289 / 0.0303 | 2.289 / 2.750 |
| Perfect sampler of the process (main / few-shot) | 0.004 / 0.004 | 0.0018 / 0.0019 | 0.045 / 0.086 |

The reference passes every check on both slices: no copied halves, window excess at most 0.0002, every measurement far below half of random. On the train slice its values are 0.226 / 0.140 / 0.218 codec FD at 2x / 4x / few-shot, close to the test values.

**What reproduces the paper:**
- **Core claim.** The AR local transformer beats parallel prediction at the same stacking factor on all three measurements, at 2x and 4x and on the few-shot task.
- **Against unstacked parallel.** The stacked AR rows beat unstacked parallel 1x on codec FD (0.221 and 0.151 against 0.323) and speaker FD (0.550 and 0.433 against 0.785). The paper's "all LT models beat all parallel models on FD" holds for the AR rows.
- **Speaker robustness at 4x.** The same few-shot model reaches speaker FD 0.68 on well-covered speakers and 1.53 on few-shot speakers. This is the scaled counterpart of the paper's SSIM drop for unseen speakers at 4x.

**What does not:**
- On pair divergence, AR 2x (0.0162) is slightly worse than parallel 1x (0.0147).
- The reference recipe does not show the paper's small 2x-to-4x quality drop: 4x scores better than 2x. Converged variants do show it: 20k cosine has codec FD 0.0166 at 2x against 0.0188 at 4x, and the larger model 0.0100 against 0.0121.
- MaskGIT beats parallel on codec FD at every point, but not on pair divergence, nor on speaker FD at 4x. It follows the paper's settings (3 passes, purity selection) and was not tuned toward the claim.

The reference keeps the paper's training-schedule shape. Koel-TTS, the paper's base, trains with Adam at 1e-4 and anneals by 0.998 every 1000 steps, which is almost constant over 200k steps. Cosine decay is therefore a variant, not a restoration of the paper's setting.

Remaining deviations from the paper are documented in `briefing/method.md` and `scoring/reference.json`:
- a scaled synthetic task (width 96, 2+1 layers, 10k steps);
- temperature-1 sampling. On the reference weights, the paper's temperature 0.6 scores −8.67.

The paper's numbers are unchanged under `_paper_reported`.

## 5. Headroom

The reference sits at **no scored bound**; every value is far above the perfect-sampler floor (section 4 table).

**The main task is nearly saturated by a simple variant.** The larger 20k cosine model reaches codec FD 0.010, speaker FD 0.063 and pair divergence 0.0021 at 2x, against floors of 0.004, 0.045 and 0.0018. That is not headroom a better method could use, which is why the few-shot experiment exists.

**The few-shot experiment added.** It covers 4x stacking for speakers with 8 training utterances, in an independent open-speaker process whose speaker effects live in one shared 16-d space. The previous iteration designed it; this iteration trained, froze, scored and verified it.

Its headroom is wide, and generic training does not use it:

| Few-shot-task model | Speaker FD, well-covered speakers 160–191 | Speaker FD, few-shot test speakers 224–255 |
|---|---:|---:|
| Reference | 0.678 | 1.534 |
| 20k cosine | 0.305 | 1.386 |
| Larger model, 20k cosine | 0.250 | 1.521 |
| Rank-16 speaker table, 20k cosine | 0.343 | 1.479 |
| Perfect sampler (floor) | 0.088 | 0.086 |

Better training halves the well-covered error but leaves the few-shot error nearly unchanged. Private search (section 3) reaches 1.27, which shows that the 8 public utterances hold speaker information the reference does not extract. A learned few-shot method, such as speaker adaptation or inferring the speaker from its utterances, has room between 1.39 and 0.09.

**Baselines and brute force:**
- Simple baselines are clearly below the reference everywhere. Parallel few-shot speaker FD is 2.29 against 1.53, and the Gaussian moment-matcher scores −3.26.
- Brute-force enumeration of joint codebook tuples is infeasible: 32^8 per frame.
- Private search is the search that does pay. It is left to review, as section 3 explains.

## 6. Weights

The experiment weights changed from 0.3 / 0.4 / 0.3 to **0.25 / 0.35 / 0.4** (`codec_stack2` / `codec_stack4` / `codec_fewshot_stack4`). Metric coefficients are unchanged: codec FD 1.0, pair divergence 0.8, speaker FD 0.6.

The few-shot experiment matters more than 0.3 says, for three reasons:
- It measures the paper's own stated weakness: robustness for poorly covered speakers at 4x.
- It is the only experiment with headroom above simple variants.
- Its test speakers' distributions are not well estimable from public data, so it is the experiment least exposed to private search.

The 4x points still carry 0.75 in total, keeping the paper's emphasis on aggressive stacking. All cohort numbers above use the new weights.

## 7. Packaging, and what is not fixed

**Run script.** `scripts/run.sh` is restored to the end-to-end reference reproduction:
- `scripts/reproduce.sh` trains the main and few-shot reference and comparison models;
- `python -m eval.test.prepare_fixtures` freezes the fixtures;
- both slices are scored;
- `validate.py` runs.

**Fixtures.** The closing job's `prepare_fixtures` run reproduced every main-task fixture byte for byte ([fixtures_check.json](eval/test/hardening/rebuild/fixtures_check.json)), with one exception: `eval/train/fixed/reference_moments.pt` was rewritten with an added `groups` key, which the train evaluator requires, and its statistics are bit-identical. The few-shot fixtures are new.

**Score files and validation.**
- `scoring/scores_test.json`, `scores_train.json` and `scores.json` hold the reference rows alone. Shortcut and variant scores live only under `eval/test/hardening/`.
- `scoring/reference.json` results equal the hidden-test reference measurements.
- `python validate.py --post-design` passed in the job and again after this write-up.

**Method code.** `method/` gained an optional `--speaker-rank` flag, off by default. The reference's computation and checkpoints are unchanged.

**Not fixed, stated plainly:**
- **Private search** pays +1.10, as much as the best honest variant. Only the shortcut review can catch it.
- **The sub-threshold partial splice** on main rows pays +0.20, below training-seed noise.
- **False compute declarations** cannot be seen in tokens.
- **The main task is close to saturation** under simple training changes. Its headroom above those is small, and the few-shot experiment carries the room for improvement.
- **The paper's 2x-to-4x quality drop** appears only for converged variants, not for the reference recipe. MaskGIT's pair-divergence and 4x speaker claims are not reproduced.
- **No real-speech claim** (WER, TitaNet SSIM, UTMOSv2) is tested. The gym measures synthetic codec-token generation only.
