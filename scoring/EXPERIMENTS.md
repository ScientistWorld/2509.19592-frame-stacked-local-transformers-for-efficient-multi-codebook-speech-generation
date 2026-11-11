# Experiments

Two synthetic codec tasks share one interface. Each utterance has 24 frames with 8 codebooks per frame and vocabulary 32. The utterances come from hidden speech-like processes and are quantized by a public FSQ codec grid.

- **Main task** (`data/synthetic_codec`): 16 speakers, with 32768 training and 2048 validation utterances.
- **Few-shot task** (`data/synthetic_codec_fewshot`): an independent process with 256 speakers in one shared speaker space.
  - Speakers 0–191 have 160 training utterances each.
  - Speakers 192–255 have 8 each.
  - Validation (2048 utterances) covers speakers 0–191 only.

Each task has 16384 conditioning rows (speaker IDs). The visible train evaluator scores one half of them and the hidden test scores the other:
- main task: a fixed random half of the rows;
- few-shot task: disjoint speaker halves (speakers 192–223 on train, 224–255 on test).

Each row is compared with four reference utterances drawn from its task's hidden process. Every candidate sees the same inputs and references. Both evaluators recompute every measurement from raw generated tokens.

| Experiment | Weight | Row | Speakers | Primary-step budget |
|---|---:|---|---|---:|
| `codec_stack2` | 0.25 | `generator_2x` | main, well covered | 12 per 24 frames |
| `codec_stack4` | 0.35 | `generator_4x` | main, well covered | 6 per 24 frames |
| `codec_fewshot_stack4` | 0.4 | `generator_fewshot_4x` | few-shot, 8 training utterances each | 6 per 24 frames |

Each experiment scores three measurements, all better lower:
- **Codec FD** (coefficient 1.0): codec-space Fréchet distance, with full covariance on consecutive-frame pairs of the public FSQ latent.
- **Codebook-pair divergence** (0.8): the mean Jensen–Shannon divergence of the token distributions of the 28 within-frame codebook pairs and the 64 frame-to-frame codebook pairs.
- **Speaker-conditioned codec FD** (0.6): the codec FD for each conditioning speaker, averaged over speakers.

No measurement is clipped. None measures agreement with published numbers or wall-clock time.

A run fails on any of the following:
- malformed or contradictory artifacts;
- copied 12-frame halves of visible utterances;
- excess 4-frame windows shared with visible utterances (more than 0.05 above a fresh process sample);
- reuse of one main operating point's samples for the other;
- any measurement not below half of its uniform-random-token value;
- changed canonical data or comparison files.

A failed or missing row fails the whole run. That run then receives the uniform-random-token value of every scored measurement in all three experiments, so it ranks below every feasible run.

Within feasible runs, the rank sums experiment weight × |coefficient| × improvement over the fixed reference. Each improvement is divided by that measurement's variation across the ranked attempts. See `briefing/evaluation.md`.

**Weights.** The few-shot experiment weighs most, for three reasons:
- It is the scaled counterpart of the paper's reported weakness: lost speaker robustness at 4x stacking for poorly covered speakers.
- It keeps the most headroom.
- Its test speakers' distributions cannot be estimated precisely from public data.

The two 4x experiments together weigh 0.75, following the paper's focus on aggressive stacking.

**The reference.** It is the paper's AR local transformer at reduced scale. On the hidden test:

| Row | Codec FD | Pair divergence | Speaker FD |
|---|---:|---:|---:|
| 2x | 0.221 | 0.0162 | 0.550 |
| 4x | 0.151 | 0.0116 | 0.433 |
| Few-shot 4x | 0.212 | 0.0185 | 1.534 |

It beats parallel prediction at the same stacking factor on every measurement of every experiment. The perfect-sampler floors are:
- codec FD 0.004;
- speaker FD 0.045 on the main task and 0.086 on the few-shot task;
- pair divergence about 0.002.

**Headroom.** Measured with learned variants on identical inputs (`HARDENING.md`):
- The same model trained 20000 steps with cosine learning-rate decay gains **+1.070**, paired 95% interval [+1.011, +1.121], in the saved feasible cohort.
- A larger model with the same schedule gains +1.005 to +1.122 across three seeds.
- The reference's own training-seed range is 0.267.

These variants come close to the floor on the main task. On the few-shot task the same models reach speaker FD 0.25–0.34 for well-covered speakers but 1.39–1.52 for the few-shot speakers. Speaker modeling from few utterances is where the remaining room is.

**Comparison rows.** These are frozen by each evaluator:
- main task: parallel heads at 1x/2x/4x and the MaskGIT LT at 2x/4x;
- few-shot task: parallel and MaskGIT at 4x.

Their raw measurements remain visible; methods cannot change them.

**Limits.** This scaled experiment establishes token-generation results, not real-speech intelligibility or perceptual quality. Tokens cannot reveal three things, which are named as shortcuts in `CONSTRAINTS.md` and measured in `HARDENING.md`:
- actual sequential compute;
- private search, that is, choosing among several samples per row to match data statistics;
- very short or sparse splices.

Paper-reported WER, SSIM, FD, UTMOSv2, confidence intervals and throughput are preserved unchanged under `_paper_reported` in `reference.json`. They are not active targets on this scale.
