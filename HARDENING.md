# Hardening: scientific ranking and remaining limits

The tested learned improvements now rank in the right direction. The valid 20,000-step variant beats the 10,000-step reference by **+0.047042**, with paired 95% bootstrap interval **[+0.041183, +0.058011]**. The reference remains feasible at score **0**; weaker training fails the task, and deterministic decoding scores **−4.711476**. Every shortcut replay in the final job fails. These are results for a scaled synthetic token-generation task, not a full reproduction of the paper's audio results. Inferred analytic sampling remains an unresolved shortcut family, and MaskGIT's full quality claim is not reproduced.

## What changed and what was run

I read the paper in `paper/` and used the original [REDTEAM.md](REDTEAM.md) plus the preceding bundled run's [results](eval/test/legacy_redteam/hardening/summary.json) to choose the final patch batch before running it. The final verification was one GPU job, `c0959cf3-667`, exit 0. No experiments were run between those patches. The job regenerated reference, baseline, stronger, weaker, seed-control and shortcut predictions, rescored them together, and restored reference-only scores. Full results are in [summary.json](eval/test/hardening/summary.json); the paired calculation is in [summarize.py](eval/test/hardening/summarize.py).

The earlier bundled run had already restored the paper's slot-specific shared embeddings, averaging of the previous stack, causal AR decoding, and cosine-mask/noncausal three-round purity decoding for MaskGIT. It trained the reference and comparisons on the final temporal training data for 10,000 steps rather than the original 5,000-step simplification. The final batch changes no training tensors, model code, optimizer or training schedule: it reuses those saved weights and regenerates their outputs for the enlarged evaluation inputs. Stronger and weaker controls likewise use their saved trained weights. No evaluator trains a model.

The final batch:

- Moves test reference generation, seeds, historical answer tensors and executable sampler code under `eval/test/`. The public data writer emits conditioning speakers only. Public train fixtures contain their own reference samples and redact test-side model predictions.
- Freezes baseline predictions, reference predictions, operating points, dimensions, thresholds, canonical visible examples and test size under `eval/`. Mutating a reserved comparison artifact or canonical training file fails the proposed run. A public `data/synthetic_codec/test.pt` is forbidden.
- Requires dense integer tensors of exact shape `(16384, 24, 8)`, IDs 0–31, and stack factors exactly 2 and 4. Impossible values, conflicting dimensions, extra score/verdict fields and missing outputs fail with finite diagnostic measurements. Submitted numbers never enter metrics or ranking.
- Recomputes full-covariance global and speaker-conditioned Fréchet distances from consecutive-frame codec latents, without the earlier distance clipping. Separately requires at least 0.5 within-frame validity, temporal validity and total validity. More than 1% exact complete-utterance copying from frozen visible examples fails. The earlier 12-frame match rule was too broad and is removed.
- Enlarges each evaluation half from 2,048 to **8,192 utterances**, with four independent references per conditioning row. All methods receive identical inputs; train and test use disjoint halves and independent implementations.
- Ranks feasible runs above every failed run. Within either tier it sums `experiment_weight × abs(coefficient) × direction × (measurement − frozen_reference) / cohort_population_std`. Direction comes only from `higher_is_better`; zero variation contributes zero. The cohort comprises currently ranked attempts plus the reference; fixed baselines cannot earn rank. Scores therefore depend on the comparison cohort, which is saved with this audit.

The 2x/4x experiment weights are **0.4/0.6**. Greater weight on 4x reflects the paper's loss of quality at aggressive stacking; a declared speedup no longer earns points. Validity/global FD/conditional FD coefficients are **1.0/0.6/0.4**, preserving emphasis on structural correctness plus distribution and speaker fidelity. Exact-token accuracy was removed because independently sampled valid utterances need not match stochastic targets. Actual sequential effort remains a constraint and cannot be established from saved tokens alone.

## Reference and reproduction

Reference-relative score is **0 before and after**, by construction. Raw measurements show the changed setup; FD values across sample sizes and latent definitions are not directly comparable.

| Stage and reference row | Validity ↑ | Global FD ↓ | Conditional FD ↓ |
|---|---:|---:|---:|
| Original REDTEAM AR 2x | 1.000000 | 0.043047 | 0.276996 |
| Original REDTEAM AR 4x | 1.000000 | 0.046990 | 0.304055 |
| Immediately preceding temporal setup, AR 2x | 0.974297 | 0.278456 | 1.272049 |
| Immediately preceding temporal setup, AR 4x | 0.933287 | 0.317690 | 1.284980 |
| Final AR 2x | **0.974291** | **0.184977** | **0.502659** |
| Final AR 4x | **0.933017** | **0.268086** | **0.602881** |
| Final parallel 1x | 0.159799 | 0.307267 | 0.870790 |
| Final parallel 2x | 0.140822 | 0.406570 | 0.891532 |
| Final parallel 4x | 0.122101 | 1.327470 | 1.747705 |
| Final MaskGIT 2x | 0.178614 | 0.511131 | 0.952696 |
| Final MaskGIT 4x | 0.125140 | 0.938013 | 1.330384 |

AR beats the 1x and matched-stack parallel comparisons on all scored quantities while using 12 or 6 primary steps per 24 frames. Its 4x quality is worse than 2x, exposing the paper's aggressive-stacking limitation. The comparisons are clearly below the reference. MaskGIT improves all three measurements over parallel at 4x, but fails validity at both operating points and has worse 2x FD than parallel: the full MaskGIT claim is **not reproduced**. This limitation was not hidden by changing its training algorithm or dropping its comparison rows.

The scaled reference uses width 96, 2 primary layers plus 1 local layer, batch 128 and 10,000 AdamW steps at learning rate 0.001. The paper uses width 768, 12+4 layers, 220,000 steps and 18,000 speech hours. Parallel comparisons get the same total layer count. Temperature 1 ancestral sampling is used instead of the paper's speech-tuned temperature 0.6/top-k 80/CFG: vocabulary is only 32, there is no text conditioning, and distribution fidelity is scored. This is an explicit remaining sampling deviation. No real WER, TitaNet SSIM or UTMOSv2 claim follows. All paper-reported values remain unchanged under `_paper_reported` in `scoring/reference.json`.

## Better and weaker variants, paired headroom

“Before” below is the immediately preceding temporal setup, not the older original REDTEAM scale. Before and after scores use different normalization and cannot be compared as absolute effect sizes. `FAILED` always ranks below any feasible result, regardless of its displayed diagnostic utility.

| Variant | Before Δ | Final Δ | Final outcome |
|---|---:|---:|---|
| Reference, 10k steps | 0 | 0 | Feasible |
| Same model, 20k steps | +0.085261 | **+0.047042** | Feasible, no complete visible copies |
| Wider/deeper model, 10k | FAILED (−2.723746) | FAILED (+0.132836) | Complete copies in 14.34% / 53.69% of 2x / 4x outputs |
| Same model, 2k steps | FAILED (−2.723746) | FAILED (−1.476989) | Validity 0.447673 / 0.268843 |
| Same weights, argmax generation | −1.742158 | **−4.711476** | Feasible but collapsed distribution |
| Remove local decoder | FAILED (−2.723746) | FAILED (−2.065545) | Fails validity |

The original audit's 10k-vs-5k improvement was +0.031724; learned-logit sampling improved that original reference by +0.143959. Those historical successes motivated restoring free-running sampling and bringing the reference to 10k steps before judging new headroom. The final valid improvement is a further 20k-vs-10k comparison, not copying the old simple baseline.

| Final model and point | Validity ↑ | Global FD ↓ | Conditional FD ↓ |
|---|---:|---:|---:|
| 20k, 2x | 0.976892 | 0.156940 | 0.476780 |
| 20k, 4x | 0.949706 | 0.191520 | 0.498852 |

All six scored cells improve; none reaches its bound. The reference likewise sits at **no scored bound** (validity <1, both distances >0), unlike the original validity-saturated task. The added difficulty is a coarse temporal walk plus persistent speaker-conditioned utterance styles, tested especially at the 4x operating point. Whole-sequence support per speaker is `32 × 4^23 × 2^7 ≈ 2.88e17`; exhaustive sequence enumeration is infeasible. This does not prevent inference of a compact analytic law, as discussed below.

Across reference seed 7 and retrainings 17/27, the final normalized range is **0.022101**; across three generation seeds it is **0.005162**. The 20k gain exceeds these measured ranges by 2.13× and 9.11×. Before enlargement the respective unnormalized ranges were 0.078569 and 0.036768; those different-unit ranges are historical only. The paired test resamples the same 16 blocks of 512 utterances, with their four references, for both methods; it recomputes full-covariance FD from sufficient statistics in 100 bootstrap replicates and holds cohort standard deviations fixed. Its interval excludes zero. Three training seeds and 100 replicates give evidence for these tested comparisons, not a universal noise bound.

Train and test agree: the valid stronger variant scores +0.044870 on train and +0.047042 on test. Reference train validity is 0.973740/0.932854, close to test 0.974291/0.933017. The larger model's positive raw utility is not counted as valid headroom because it fails novelty.

## Shortcut replays

The original “before” scores are from REDTEAM.md. Final utilities below are diagnostic values within the failed tier; a failed zero or positive value does **not** tie a feasible reference. The [raw outputs and measurements](eval/test/hardening/) preserve the full evidence. Every changed defense was checked in the same final job.

| Shortcut | Original Δ | Final outcome (diagnostic Δ) | Fix and replay scope |
|---|---:|---|---|
| Copy visible test answers | +0.586483 | FAILED (+0.197089) | Test answer file removed; replay falls back to visible training copies, rejected at 100% copying |
| Rebuild exact answers with public `make_split` | +0.586483 | FAILED (−2.806780) | Public generator removed; import fails and empty fallback fails schema |
| Original independent-frame rule sampler | +0.146181 | FAILED (−1.058027) | Temporal task and separate temporal threshold; old sampler has temporal validity ≈0.125 despite high within-frame validity |
| Change stack factor to 24 | +1.247500 | FAILED (−2.806780) | Exact evaluator-fixed operating point; no speed reward |
| Negative stack factor | +7.485000 | FAILED (−2.806780) | Positive exact integer required |
| Infinite stack factor | +1.497000 | FAILED (−2.806780) | Noninteger/nonfinite configuration rejected |
| Zero stack factor | Crashed | FAILED (−2.806780) | Explicit rejection, finite failed-run result |
| Prewrite scores, add success/metric fields | 0, paying tie | FAILED (−2.806780) | Scores overwritten from raw outputs; extra verdict/metric fields reject the artifact |
| Replace baseline predictions | +1.612635 aggregate | FAILED (0) | Comparisons loaded from frozen evaluator outputs; reserved-name mutation detected; baseline rows earn no rank |
| Replace ground truth | +0.300536 aggregate | FAILED (0) | Private reference generation; old test file is forbidden. Replay mutates canonical validation data and is rejected |
| Independent random tokens / one-step claim | +0.424047 | FAILED (−2.660213) | Feasibility dominates all other metrics. Replayed with legal 2x/4x declarations to isolate task failure; original 24 claim would also fail |
| Claim vocabulary 1 | −18.880855 | FAILED (−2.806780) | Fixed vocabulary and contradiction check |
| Out-of-range IDs | −32.765738 | FAILED (−2.806780) | Range check before metric computation |
| Empty output | Crashed | FAILED (−2.806780) | Exact count/shape required |

The immediately preceding setup still let unchanged stack metadata and forged extra fields tie the reference; both now fail. Its adapted rule sampler scored +0.349413 by calling the then-public temporal sampler. Removing that public function closes the direct-call path. **The final replay tests the original independent-frame rule, not a newly rederived temporal sampler**; the broader analytic family is not claimed closed. The old all-zero constant shortcut already lost (−19.847761) and was not rerun, as requested for unchanged, nonpaying attacks.

## Remaining limits and packaging

1. **Inferred analytic sampling remains possible.** Training data reveal a compact synthetic process. A sampler that independently infers the temporal walk and speaker style law could match its distribution without a learned generator. Moving exact reference seeds and generators closes answer regeneration and direct calls, but cannot make this family statistically distinguishable from an equally good learned distribution. Under this gym's learned-generation constraint such a hand-coded rule remains a shortcut for the separate review; it is not comprehensive hardening of that major family. An honest learned alternative with indistinguishable quality is a measurement/headroom limit, not a loophole.
2. Exact complete-utterance copying is detected. Partial splicing, learned rule repair, and false claims about actual sequential computation require the separate shortcut review. The wider model really emitted many complete copies; relaxing the novelty threshold to count its gain would reopen copying.
3. Full-covariance FD measures first and second moments in a fixed synthetic latent space, not every higher-order dependency or perceptual property. Moment-matched distributions can remain indistinguishable. No universal ordering of all scientifically better methods is established.
4. The full MaskGIT and real-speech claims remain unfulfilled on this scale. Larger speech data/models were unavailable and further training was outside this final verification batch. The milestone remains **core_claim_plus**, with partial reproduction and the limits above explicitly recorded.

`scoring/scores_test.json`, `scores_train.json` and the default scores contain **reference runs alone**; audit attempts appear only under `eval/test/hardening/` and in this report. `scripts/evaluate_test.sh` takes no arguments and discovers all submitted methods. Train/test share no implementation code and `eval/` imports no `method/`. `scripts/run.sh` is restored to train all reference/comparison models end to end, regenerate fixtures, score both slices and validate. A fresh clone uses tracked original synthetic training artifacts; `scripts/download.sh` contains a real public arXiv download for the external paper, and `environment/` builds the container and pins the verified dependencies. No committed executable code depends on `shared/`.

The verification job passed `python validate.py --post-design` ([log](eval/test/hardening/validation.log)). Final packaging also passed [structural validation](eval/test/hardening/final_validation.log) and [ranking property checks](eval/test/hardening/ranking_properties.log) for direction, absolute coefficients, unit invariance, zero variation and feasibility precedence. No defenses rely on persistent processes or Python-environment manipulation.
