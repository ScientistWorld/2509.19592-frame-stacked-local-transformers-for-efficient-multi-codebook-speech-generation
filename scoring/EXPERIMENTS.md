# Experiments

The fixed dataset contains 24-frame utterances with 8 codebooks, vocabulary 32, and 16 speakers. Training has 8192 utterances and validation 1024. Evaluation has 16384 conditioning rows, split into disjoint train and test halves of 8192; each half has four reference samples per row. Every candidate uses the same inputs and references. Both evaluation slices independently compute the same measurements from raw generated tensors.

| Experiment | Weight | Required output | Primary-step budget |
|---|---:|---|---:|
| `codec_stack2` | 0.4 | `generator_2x` | 12 per 24 frames |
| `codec_stack4` | 0.6 | `generator_4x` | 6 per 24 frames |

Both score dependency validity (coefficient 1.0), codec-space Fréchet distance (0.6), and speaker-conditioned codec-space Fréchet distance (0.4). Validity is better higher; both distances are better lower. Full covariance and consecutive-frame pairs measure joint distribution and temporal structure. No distance is clipped, and no metric measures agreement with published results or wall-clock time.

Every feasible attempt outranks every failed attempt. Within each tier, coefficient-weighted improvements over the fixed reference are standardized by variation across the attempts being ranked, then summed with the experiment weights. See `briefing/evaluation.md` for the precise interface and failure conditions.

The harder 4x point receives more weight because the scientific problem is preserving quality at aggressive compression of the primary generation schedule. The temporal and utterance-level variation keeps this point below its quality ceiling. The final hardening evidence compares genuinely learned stronger and weaker variants on identical test inputs, including paired bootstrap uncertainty.

Comparison rows at 1x, 2x and 4x are frozen by each evaluator. Their raw metrics remain visible even when they fail the task; methods cannot change them. The reference has room above it on every scored metric. This scaled experiment establishes token-generation results, not real-speech intelligibility or perceptual quality.

Paper-reported WER, SSIM, FD, UTMOSv2, confidence intervals and throughput are preserved unchanged under `_paper_reported` in `reference.json` and are not active targets on this scale.
