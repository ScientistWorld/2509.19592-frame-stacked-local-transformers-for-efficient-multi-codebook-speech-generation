# Experiments

Both scored experiments use the synthetic multi-codebook codec described in `briefing/problem.md` (16 speakers, 24 frames, 8 codebooks, vocabulary 32; within-frame residual chains, a coarse random walk across frames, and residual style bits drawn per speaker that persist through the utterance). A generator writes one utterance per conditioning speaker (4096 rows); the hidden test scores 2048 of them against 8192 reference utterances it draws itself, so every generator sees the same conditions and references.

Metrics per row: `dependency_validity` (primary, higher is better), `codec_fd` and `speaker_conditioned_codec_fd` (lower is better); see `briefing/evaluation.md`. A row with validity below 0.5, copied visible data, an impossible or over-budget `stack_factor`, or a malformed artifact is failed and gets worst-case values on every metric (validity 0, distances at the random-token cap).

## codec_stack2 (weight 0.4)

Operating point: at most 12 primary (coarse) generation steps per 24-frame utterance, i.e. each coarse step emits 2 frames (16 tokens). Ranked row: `generator_2x`. Comparison rows (frozen in the hidden test): `baseline_parallel_1x` (one frame per coarse step, every codebook sampled independently; 24 steps), `baseline_parallel_2x`, `baseline_maskgit_2x`.

## codec_stack4 (weight 0.6)

Operating point: at most 6 coarse steps per utterance, i.e. 4 frames (32 tokens) per coarse step. Ranked row: `generator_4x`. Comparison rows: `baseline_parallel_4x`, `baseline_maskgit_4x`.

Weighting: the 4x point is weighted higher because it is where the efficiency gain is largest and where the paper reports its own method losing quality, so improvements there are the most informative.

## Paper context

`scoring/reference.json` keeps the paper's LibriTTS numbers (WER, SSIM, FD, UTMOSv2 with 95% CIs at stack factors 1/2/4, and relative throughput) under `_paper_reported`. They are not on this gym's scale and are not targets.
