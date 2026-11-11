# Progress

Current milestone: **core_claim_plus**, final hardening verified on 2026-11-11. The current evidence is in [HARDENING.md](HARDENING.md) and [VERIFICATION.md](VERIFICATION.md). Older stages are recorded in [MILESTONES.md](MILESTONES.md).

The paper's AR local-transformer claim is reproduced at reduced synthetic scale, in the paper's direction. AR LT beats parallel prediction at the same stacking factor on codec FD, codebook-pair divergence and speaker-conditioned FD, at 2x and 4x on the main task and at 4x on the few-shot task.

The few-shot task reproduces the paper's loss of speaker robustness at 4x. The same model has speaker FD 0.68 on well-covered speakers and 1.53 on few-shot speakers.

Not reproduced:
- MaskGIT's pair-divergence and 4x speaker claims;
- the 2x-to-4x quality drop, under the reference recipe (converged variants do show it);
- real-speech metrics.

The closing job `bbe268c5-f24` exited 0 and passed `python validate.py --post-design`:
- **Few-shot experiment completed.** The job trained the few-shot reference and comparisons, froze the fixtures, and replayed all 49 cases with all three experiments.
- **Better variants score higher.** The 20k-step cosine-LR variant scores +1.070 [+1.011, +1.121] in the feasible cohort. A larger model scores +1.005 to +1.122 over 3 seeds. The reference's training-seed range is 0.267.
- **Weaker variants score lower or fail.** 2k steps scores −1.68 and no local transformer −2.98. Argmax decoding and the few-shot memorizer fail.
- **REDTEAM shortcuts fail.** Every REDTEAM.md shortcut and every whole-run copy or splice fails.
- **Shortcut left: private search.** Choosing one of 3 samples per row to match the public data's moments pays +1.10. It cannot be detected from tokens, so it is named as a shortcut in `scoring/CONSTRAINTS.md` and left to the shortcut review.
- **Weights.** The experiment weights are now 0.25 / 0.35 / 0.4, with the few-shot experiment weighing most.

`scripts/run.sh` reproduces reference and comparison training (both tasks), predictions, fixtures, both evaluations and validation, end to end. Score files contain the reference run alone.
