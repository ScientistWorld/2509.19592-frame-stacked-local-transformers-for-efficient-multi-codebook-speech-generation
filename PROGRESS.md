# Progress

Current milestone: **core_claim_plus**, final hardening verified on 2026-11-10. The current evidence is in [HARDENING.md](HARDENING.md) and [VERIFICATION.md](VERIFICATION.md). Older stages are recorded in [MILESTONES.md](MILESTONES.md).

The paper's AR local-transformer claim is reproduced at reduced synthetic scale, in the paper's direction:
- AR LT beats parallel prediction at 1x/2x/4x on dependency validity, codec-latent FD, and speaker-conditioned FD;
- its quality degrades from 2x to 4x.

The MaskGIT claim is not reproduced, and neither are real-speech metrics.

The final replay job `9bb82593-d9d` exited 0 and passed `python validate.py --post-design`:
- **Better variants score higher.** The same model trained 20k steps with cosine LR decay gains +0.247 over the reference (paired 95% interval [+0.229, +0.266]). Its worst training seed (+0.146) is above the best reference retraining (+0.041).
- **Weaker variants fail.** Weaker training, no local transformer, argmax decoding, and memorizing larger models all fail.
- **REDTEAM shortcuts fail.** Every shortcut REDTEAM.md reported fails.
- **New fix this iteration.** A memorization check (4-frame windows shared with visible utterances, minus those shared with a fresh data-process sample, at most 0.05) closes splices of visible training segments of 4 or more frames. The three-piece splice scored +0.271 before this fix.

Four shortcut families still pay and are left to the shortcut review; see HARDENING.md section 4:
- hand-written samplers of the data process;
- repair or filtering with the public validity rule;
- false 4x compute declarations;
- 3-frame or sub-threshold partial splices.

`scripts/run.sh` reproduces reference and comparison training, predictions, fixtures, both evaluations and validation, end to end. Score files contain the reference run alone.
