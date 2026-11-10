# Targets

Improve learned joint codec generation under the fixed 12-step and 6-step primary schedules. `codec_stack2` has weight 0.4 and `codec_stack4` weight 0.6. Every feasible run outranks every failed run.

Increase dependency validity (coefficient 1.0), decrease full-covariance consecutive-frame codec FD (0.6), and decrease speaker-conditioned codec FD (0.4). Direction comes from `higher_is_better`; coefficient magnitudes determine relative importance after normalization across the ranked attempts. See `briefing/evaluation.md` for the complete formula.

The fixed reference's reproduced values are in `reference.json`. They are below the quality bound on all scored metrics at both operating points. The aggressive point tests temporal transitions and persistent utterance characteristics that are difficult at a reduced primary frame rate. Headroom must be assessed with learned variants and paired uncertainty, not by an algebraic bound alone.

A valid learned variant (same model, 20000 steps with cosine learning-rate decay) improves every scored measurement, with normalized test gain +0.247 and paired 95% interval [+0.229, +0.266] in the saved audit cohort; its gain holds across three training seeds (worst seed +0.146, best reference retraining +0.041). The reference is not at a bound on any scored metric, and this measured gain establishes usable headroom, largest at 4x validity and in speaker-conditioned distance. See `HARDENING.md` for seed variation and limitations.
