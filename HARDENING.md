# Hardening: final state of the test

**Better results now score higher, weaker ones lower, and every shortcut that REDTEAM.md reported fails.** The reference scores 0. The strongest honest learned variant scores **+0.247**, with paired 95% interval **[+0.229, +0.266]**. Its gain holds across three training seeds. Weaker variants fail the task and rank below every feasible run.

This iteration found one new paying shortcut: splicing visible training utterances into pieces that are shorter than the old 12-frame copy window. It now fails, through a memorization check added to both evaluators. Four shortcut families still pay. Generated tokens cannot reveal them, so they are left to the separate shortcut review (section 4):
- a hand-written sampler of the synthetic data process;
- repairing or filtering outputs with the public validity rule;
- false compute declarations;
- splices of 3-frame pieces, or of a small fraction of rows.

All numbers below come from the final replay job `9bb82593-d9d` (CPU, exit 0) unless marked otherwise. That job rescored every saved case through the no-argument `scripts/evaluate_test.sh` and `scripts/evaluate_train.sh`. Evidence:
- [summary.json](eval/test/hardening/last/summary.json) and [summary.txt](eval/test/hardening/last/summary.txt);
- per-case raw measurements in [scores/](eval/test/hardening/last/scores/);
- the before-fix replay of job `47df85f6-fd8`, in [summary_before_excess_check.json](eval/test/hardening/last/summary_before_excess_check.json);
- the runner, [run_last.py](eval/test/hardening/last/run_last.py).

Nothing was retrained in this iteration. Method code, training data, the reference, and the comparison outputs are unchanged since the earlier hardening jobs that trained them.

Scores are the rank's quantity: for each experiment and metric, `weight × |coefficient| × improvement in the higher_is_better direction ÷ the metric's standard deviation across the saved cohort` (54 cases, including failed runs at worst values). Absolute values therefore depend on the cohort. Scores from REDTEAM.md used raw, unnormalized utility on an older task and are only comparable in sign and order.

## 1. Do better results score higher?

Yes. The table shows the before/after view of the variants.

| Variant (same inputs, paired) | REDTEAM.md (original gym) | Final test | Final train | Final outcome |
|---|---:|---:|---:|---|
| Reference, AR LT, 10k steps | 0 | **0** | 0 | feasible |
| Same model, 20k steps, cosine LR decay | — | **+0.247** [+0.229, +0.266] | +0.262 | feasible |
| …same, training seeds 17 / 27 | — | +0.231 / +0.146 | +0.247 / +0.156 | feasible |
| Same model, 20k steps, constant LR | +0.031724 (10k vs 5k) | +0.091 [+0.071, +0.120] | +0.084 | feasible |
| …same, training seeds 17 / 27 | — | +0.029 / +0.033 | +0.042 / +0.042 | feasible |
| Same model, 10k steps, cosine LR (seeds 7/17/27) | — | +0.054 / −0.118 / +0.129 | +0.062 / −0.109 / +0.140 | feasible, seed-noisy |
| Reference weights, paper temperature 0.6 | — | −0.263 [−0.282, −0.241] | −0.237 | feasible |
| 2k steps (weaker) | −1.127834 (100 steps) | FAILED −3.957 | FAILED | validity 0.448 / 0.269 |
| Local transformer removed (weaker) | −0.883640 | FAILED −3.957 | FAILED | validity 0.141 |
| Reference weights, argmax decoding | — | FAILED −3.957 | FAILED | distribution no closer than uniform random tokens |
| Wider/deeper model, 10k / 20k steps | — | FAILED −3.957 | FAILED | memorizes: 30% / 67% of rows repeat a visible 12-frame half |

In REDTEAM.md, independent random tokens beat the reference by +0.424 even though they failed the task. That inversion is gone: random tokens now fail (validity 0.070), and every failed run receives the worst value of every scored metric in both experiments.

Noise:
- Three retrainings of the reference span **0.115** (seed 17 −0.074, seed 27 +0.041).
- Three generation seeds of the reference span **0.014**.
- The 20k cosine variant's worst seed (+0.146) is above the best reference seed (+0.041). Its mean gain over the reference seed group is +0.219.
- The 20k constant-LR variant gains +0.062 on average, but its worst seed overlaps the reference seed range. It is a real but small improvement.

The paired interval uses a block bootstrap with 16 blocks and 200 replicates. Every replicate uses identical held-out rows and identical hidden references for both methods, recomputes full-covariance FD from resampled sufficient statistics, and holds the cohort standard deviations fixed.

Train and test agree:
- The best variant scores +0.262 on train and +0.247 on test.
- Reference validity is 0.9737 / 0.9329 on train and 0.9743 / 0.9330 on test.

## 2. Shortcuts: fixes and scores before and after

"Before" for the REDTEAM rows is the score REDTEAM.md reported. For shortcuts found later, it is the replay before the fix that closes them. "FAILED" ranks below every feasible run. Its diagnostic utility is −3.957 here because every scored metric is replaced by its worst value.

| Shortcut | Before | Final | Fix |
|---|---:|---|---|
| Copy visible test answers | +0.586483 | FAILED | No test answer file; test references are drawn inside `eval/test/` from a hidden seed. The replay falls back to copying training utterances, which fails the half-copy check (100%) and the window-excess check (0.955). |
| Rebuild answers with the public `make_split` | +0.586483 | FAILED | The data generator is no longer public (only `eval/test/` holds it). The replay's import fails, and its empty fallback fails the shape check. |
| Rule sampler (independent frames) | +0.146181 | FAILED | Temporal task. Coarse-transition validity must be at least 0.5; this sampler reaches about 0.125. |
| Stack factor 24 / −1 / ∞ / 0 | +1.2475 / +7.485 / +1.497 / crash | FAILED (all four) | Exact evaluator-fixed operating points (2 and 4, positive integers). No speed reward. |
| Prewrite scores, add verdict fields | 0 (paying tie) | FAILED | Scores are recomputed from raw tokens, and extra top-level fields fail the artifact. |
| Replace baseline predictions | +1.612635 (aggregate) | FAILED | Comparison rows load from `eval/*/fixed`. Changing a reserved file fails the run. |
| Replace ground truth (test file / validation file) | +0.300536 (aggregate) | FAILED (both) | A method-side `test.pt` is forbidden, and train/validation are hash-checked. |
| Independent random tokens, one step | +0.424047 | FAILED | Validity threshold (0.070 < 0.5). Failed runs get worst values everywhere. |
| Constant zeros / vocabulary 1 / out-of-range IDs / empty output | −19.85 / −18.88 / −32.77 / crash | FAILED (all four) | Fixed vocabulary, range and shape checks. |
| Copy with local edit; splice of aligned 12-frame halves | +0.124 / +0.113 (earlier hardening replay) | FAILED | Aligned 12-frame-half copy check (> 1% of rows fails the run). |
| 2x outputs reused as the 4x row (whole or via unscored rows) | +0.060 / +0.059 | FAILED | Cross-operating-point reuse check over the full sibling file. |
| **Splice of training utterances at frames 8 and 16** (new) | **+0.271** (job `47df85f6-fd8`) | **FAILED** (excess 0.684) | **New this iteration: memorization check**, described below. |
| **Splice of 6-frame / 4-frame pieces** (new) | +0.237 / +0.152 (local rescoring of the same construction, job-1 cohort) | **FAILED** (excess 0.547 / 0.275) | Memorization check. |
| Train on the formerly public train-slice references | — | −0.058 (feasible) | Does not pay. The train fixture holds only the sufficient statistics of those references. |

**The memorization check** applies to both evaluators: `visible_window_excess` in `eval/test/evaluate.py` and `eval/train/evaluate.py`. For every scored utterance it counts the 4-frame windows (all codebooks, same frame positions) that occur in the frozen visible utterances, and subtracts the count that occurs in a fresh sample of the data process of the same size and speaker composition.
- **Why its expectation is 0:** a generator that does not reproduce visible segments matches both sets equally often in expectation. Only copied or memorized visible segments raise the visible count.
- **Threshold:** a run fails when the excess exceeds 0.05 of all windows.
- **Honest outputs:** the reference, all learned variants, the analytic sampler, rule repair and best-of-8 filtering stay at or below **0.012**. The failed argmax control reaches 0.018.
- **Copied outputs:** every splice of pieces of 4 frames or more is at or above **0.27**, and memorizing or copying runs score 0.09–0.99.
- **Control samples:** the hidden test draws its control at evaluation time from a seed only `eval/test/` holds. The public train evaluator holds only 64-bit hashes of its control windows (`eval/train/fixed/control_windows.bin`, written by `eval/test/prepare_fixtures.py`), so the control utterances cannot become extra training data.
- **Margin on train:** the train side uses only the train and validation utterances as its visible set, and its worst honest learned value is 0.012 (`better_more_training_seed17`). That leaves a fourfold margin.

## 3. Reference before and after

The reference-relative score is 0 before and after, by construction. Raw reference measurements did not change in this iteration. Its outputs and comparison rows are frozen, `scoring/scores_test.json` matches `scoring/reference.json` to 1e-12, and its window excess is 0.0009 at 2x and 0.0008 at 4x.

| Row (hidden test) | Validity ↑ | Codec FD ↓ | Speaker-conditioned FD ↓ |
|---|---:|---:|---:|
| REDTEAM.md AR 2x / 4x (original task, validity saturated) | 1.000 / 1.000 | 0.0430 / 0.0470 | 0.277 / 0.304 |
| **Final AR LT 2x** (`generator_2x`) | **0.9743** | **0.1850** | **0.5027** |
| **Final AR LT 4x** (`generator_4x`) | **0.9330** | **0.2681** | **0.6029** |
| Parallel 1x / 2x / 4x | 0.160 / 0.141 / 0.122 | 0.307 / 0.407 / 1.327 | 0.871 / 0.892 / 1.748 |
| MaskGIT LT 2x / 4x | 0.179 / 0.125 | 0.511 / 0.938 | 0.953 / 1.330 |

The paper's core claim is reproduced in its direction. The AR local transformer beats parallel prediction at every stack factor on all three measurements, and it degrades from 2x to 4x, as the paper reports for aggressive stacking.

The paper's MaskGIT claim is **not reproduced**. Its 4x row beats parallel 4x on both distances, but it fails the validity threshold at both points, and its 2x distances are worse than parallel 2x. With only 3 unmasking passes, tokens unmasked in the same pass are sampled independently, which hurts this strongly chained synthetic codec. MaskGIT follows the paper's settings (non-causal LT, 3 passes, purity selection); it was not altered to change this result. The comparison rows earn no rank.

The remaining deviations from the paper are documented in `briefing/method.md` and `scoring/reference.json`:
- a scaled synthetic task (width 96, 2+1 layers, 10k steps, instead of 768, 12+4 layers, 220k steps);
- temperature-1 sampling instead of Koel-TTS's temperature 0.6 / top-k 80 / CFG. On the reference weights, the paper's temperature scores −0.263, so restoring it would hand a future agent a trivial +0.26.

The paper's numbers are unchanged under `_paper_reported`.

## 4. Shortcuts left to the shortcut review, and why

These were measured on the hidden test in the final job. None can be told apart from honest generation using the tokens alone.

| Shortcut | Final test | Why it is not closed |
|---|---:|---|
| Hand-written sampler of the data process (affine residual rule, coarse walk and style rates fitted from training data) | **+0.390** [+0.375, +0.409] | The validity rule must be public, because the train evaluator scores it, and it is visible in the training data anyway. A perfect sampler of the data process produces the same distribution as a perfect learned generator. It sits at the task's bound: validity 1, codec FD 0.013, speaker FD 0.23, which is the finite-sample floor. |
| Rule repair: snap the reference's outputs onto the public rule | +0.202 [+0.191, +0.213] | This is private computation with the public train-side scorer. The scorer is the data's own deterministic structure, so it cannot be kept out of reach. |
| Best-of-8 per row by public validity | +0.041 [+0.009, +0.071] | Same reason as rule repair. |
| 2x model's second sample submitted as the 4x row | +0.099 [+0.085, +0.119] | Tokens cannot reveal how many primary steps produced them. Reusing the same samples is caught; an independent sample is not. |
| Splice of 3-frame pieces | +0.030 [+0.018, +0.049] | A 3-frame window check would flag honest samplers too: the analytic sampler already matches visible 6-frame windows in 4% of rows. The gain is far below the 0.115 training-seed range. |
| 6% of rows replaced by 8-frame splices | +0.037 [+0.032, +0.044] | Sized to stay under the 0.05 threshold (excess 0.042 / 0.044). The residual gain is below training-seed noise. |

`scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` name all of these as shortcuts. Each needs code that indexes training data, hard-codes or applies the validity rule, or misdeclares the operating point. The hand-written sampler and rule repair are the major shortcuts that still pay, and only code review can catch them.

## 5. Headroom

The reference sits at **no scored bound**:
- Validity is below 1 (0.974 at 2x, 0.933 at 4x).
- Both distances are well above the finite-sample floor of a perfect sampler (codec FD about 0.013, speaker-conditioned FD about 0.23).

The 4x experiment, weight 0.6, is the experiment from where the paper's method falls short (aggressive stacking). Earlier iterations added it with a temporal, utterance-persistent data process, and it keeps the most headroom. The best honest variant leaves 4x validity unchanged (0.933) and reaches codec FD 0.039 and speaker FD 0.311. A better method still has room in validity at 4x, in residual persistence across primary steps (0.79–0.80 against 1.0 in the data), and in speaker-conditioned distance.

Simple baselines are clearly below the reference: every parallel and MaskGIT row fails validity. Brute force is infeasible: there are 32^8 joint tuples per frame, and about 2.9e17 sequences per speaker.

**Weights are unchanged:** codec_stack2 0.4 and codec_stack4 0.6; validity 1.0, codec FD 0.6, speaker FD 0.4. The evidence did not show any metric mattering more than its weight already says.

## 6. Packaging and anything not fixed

**Run script:** `scripts/run.sh` is restored to the end-to-end reference reproduction:
- train the reference and comparison models;
- run `python -m eval.test.prepare_fixtures`;
- score both slices;
- run `validate.py`.

**Fixture reproduction:** the final job ran the packaging step on the frozen reference outputs.
- Every committed fixture was reproduced bit for bit, including the new control hashes, with one exception: `eval/train/fixed/reference_moments.pt` was reproduced only to within 8.9e-16 (|cov| ≤ 2.8). The committed file was computed with 2 BLAS threads and the job ran with 1. A 2-thread recomputation is identical.
- This file feeds only the public train evaluator.
- Retraining with `run.sh` would also reproduce the reference only up to GPU nondeterminism.

**Score files:** `scoring/scores_test.json`, `scores_train.json` and `scores.json` hold the reference run alone. Audit cases live only under `eval/test/hardening/`.

**Validation:** `python validate.py --post-design` passed in the job ([validation.log](eval/test/hardening/last/logs/validation.log)) and again after packaging.

**Not fixed:**
- the shortcut families in section 4;
- the MaskGIT claim;
- any real-speech claim (WER, TitaNet SSIM, UTMOSv2): the gym measures synthetic token generation only.
