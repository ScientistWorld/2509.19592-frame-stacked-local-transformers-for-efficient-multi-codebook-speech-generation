# Targets

Improve learned joint codec generation under fixed primary schedules. There are three experiments:
- `codec_stack2` (weight 0.25): 12 primary steps per 24 frames on the main task;
- `codec_stack4` (weight 0.35): 6 primary steps on the main task;
- `codec_fewshot_stack4` (weight 0.4): 6 primary steps for speakers with 8 training utterances.

Every feasible run outranks every failed run.

In every experiment, decrease codec FD (coefficient 1.0), codebook-pair divergence (0.8) and speaker-conditioned codec FD (0.6). Direction comes from `higher_is_better`. Coefficient magnitudes set relative importance once each improvement has been divided by its variation across the ranked attempts; see `briefing/evaluation.md`.

The fixed reference's hidden-test values are in `reference.json`:

| Row | Codec FD | Pair divergence | Speaker FD |
|---|---:|---:|---:|
| 2x | 0.221 | 0.0162 | 0.550 |
| 4x | 0.151 | 0.0116 | 0.433 |
| Few-shot 4x | 0.212 | 0.0185 | 1.534 |

None of these is at a bound. A perfect sampler of the data process reaches about 0.004 codec FD and 0.002 pair divergence. Its speaker FD is 0.045 on the main task and 0.086 on the few-shot task.

**Headroom on the main task.** Better training of the same method improves every measurement there. The same model trained for 20000 steps with cosine decay gains +1.070 in the saved feasible cohort, paired 95% interval [+1.011, +1.121], against a reference training-seed range of 0.267.

**Headroom on the few-shot task.** The largest open target is few-shot speaker FD. The same models that reach 0.25–0.34 for well-covered speakers stay at 1.39–1.52 for few-shot speakers. Sharing speaker information across speakers and adapting from the 8 utterances are in scope. Selecting rows toward data statistics is not; see `CONSTRAINTS.md`.

See `HARDENING.md` for seed variation, paired intervals and limitations.
