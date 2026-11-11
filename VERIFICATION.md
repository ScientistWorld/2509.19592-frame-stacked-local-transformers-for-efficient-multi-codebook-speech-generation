# Verification report

Milestone: **core_claim_plus**. Verified:
- the scaled AR local-transformer result on both tasks;
- the few-shot speaker-robustness shortfall at 4x;
- measurable learned headroom.

The full MaskGIT claim and the real-speech claims are not reproduced. See [HARDENING.md](HARDENING.md) for before/after results, shortcut replays and remaining limitations.

**Final job `bbe268c5-f24` (1 GPU, exit 0):**
- trained the few-shot reference (AR LT 4x) and its parallel and MaskGIT comparisons, 10k steps each;
- froze the fixtures of both tasks;
- added few-shot rows to every saved case, training each variant's few-shot model;
- built the private-search, few-shot-memorizer and main-only splice cases;
- scored all 49 cases through the no-argument `scripts/evaluate_test.sh` and `scripts/evaluate_train.sh`;
- computed the paired block bootstrap and the few-shot shortfall diagnostic;
- restored reference-only scores and passed `python validate.py --post-design`.

The main-task models were trained in job `4a264730-2a8` from the same data and method code, and were not retrained.

**Reference (hidden test).**

| Row | Codec FD | Pair divergence | Speaker FD |
|---|---:|---:|---:|
| 2x | 0.2208 | 0.01616 | 0.5503 |
| 4x | 0.1506 | 0.01162 | 0.4332 |
| Few-shot 4x | 0.2122 | 0.01852 | 1.5339 |

It passes every check on both slices. Train-slice values are close to the test values.

**Learned headroom** (feasible cohort, paired 95% intervals):
- 20k cosine: +1.070 [+1.011, +1.121];
- larger model: +1.041 / +1.005 / +1.122 across 3 seeds;
- rank-16 speaker table: +1.013.

Reference noise: training-seed range 0.267, generation-seed range 0.057. On few-shot speakers, speaker FD stays at 1.39–1.52 even for models that reach 0.25–0.34 on well-covered speakers.

**Shortcuts.** Every REDTEAM.md shortcut and every whole-run copy, splice or memorization case fails. Still paying, and left to the shortcut review:
- private search, +1.104 (main rows only +0.776);
- the sub-threshold 12% main-row splice, +0.200, below training-seed noise.

**Fixtures.** Every main-task fixture was reproduced byte for byte. `eval/train/fixed/reference_moments.pt` was rewritten with its `groups` key; its statistics are bit-identical ([fixtures_check.json](eval/test/hardening/rebuild/fixtures_check.json)).

**Score files.** `scoring/scores_test.json`, `scores_train.json` and `scores.json` hold the reference runs alone. `scoring/reference.json` results equal the hidden-test reference measurements.

**Paper numbers.** The paper's reported results are unchanged under `_paper_reported`.
