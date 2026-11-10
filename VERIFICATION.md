# Verification report

Milestone: **core_claim_plus**. The scaled AR local-transformer result and measurable learned headroom are verified. The full MaskGIT and real-speech claims are not reproduced. See [HARDENING.md](HARDENING.md) for before/after results, shortcut replays and remaining limitations.

Final jobs:
- **`47df85f6-fd8` (GPU).** Checked fixture reproduction. Built the validity-filter, rule-repair and three-piece-splice cases, and replayed every saved case.
- **`9bb82593-d9d` (CPU).** Ran after the memorization check was added. Replayed all 54 cases through the no-argument `scripts/evaluate_test.sh` and `scripts/evaluate_train.sh`, computed the paired bootstrap, restored reference-only scores and validated.

No model was retrained in this iteration.

**Learned headroom.** The 20k-step cosine-LR variant improves all six scored measurements:
- normalized test gain +0.247, paired 95% interval [+0.229, +0.266];
- train gain +0.262;
- seeds 17/27 score +0.231/+0.146.

Reference noise: training-seed range 0.115, generation-seed range 0.014.

**Weaker variants and memorizing models fail.** Weaker training, no local transformer and argmax decoding all fail. Larger models that memorize training utterances also fail, and are not counted as headroom.

**Shortcuts.** Every REDTEAM.md shortcut and every copy, splice and operating-point-reuse case fails. The shortcuts that still pay are listed with measured scores in HARDENING.md section 4:
- the analytic sampler, +0.390;
- rule repair, +0.202;
- the 2x model submitted as the 4x row, +0.099;
- best-of-8 filtering, +0.041;
- the sub-threshold partial splice, +0.037;
- the 3-frame splice, +0.030.

**Fixtures.** Running `python -m eval.test.prepare_fixtures` on the frozen reference outputs reproduced every committed fixture, with one exception: the train-slice moment file matched only to within 8.9e-16, because of a different thread count ([fixtures_check.json](eval/test/hardening/last/fixtures_check.json)).

**Score files.** `scoring/scores_test.json` matches `scoring/reference.json` to 1e-12. The score files contain reference runs alone.

**Paper numbers.** The paper's reported results are unchanged under `_paper_reported` in `scoring/reference.json`.
