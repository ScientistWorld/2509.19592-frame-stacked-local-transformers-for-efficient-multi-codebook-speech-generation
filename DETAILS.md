# Gym details

This gym scales the paper’s frame-stacked multi-codebook generation problem to synthetic codec tokens. The learned reference uses the paper’s autoregressive local transformer, shared slot embeddings and averaged previous-stack input. The parallel comparisons receive the same total transformer depth; the MaskGIT comparisons use cosine masking and three purity-based refinement rounds. See `briefing/method.md` for the implementation and explicit deviations from the paper.

Training has 8192 utterances and validation 1024. Evaluation has 16384 speaker conditions, split into two comparable halves of 8192. Each evaluator measures structural validity and full-covariance consecutive-frame codec Fréchet distances against four reference samples per condition. All references, baselines, operating points, dimensions and thresholds are fixed by the evaluators. The test sampler and historical answer artifacts live under `eval/test/`.

`briefing/evaluation.md` specifies the raw-output interface and feasibility-first, cohort-standardized ranking. `scoring/EXPERIMENTS.md` records active measurements and weights. Paper-reported speech results remain unchanged and inactive under `_paper_reported` in `scoring/reference.json`.

Run `bash scripts/run.sh` to reproduce all reference and comparison models, freeze evaluator fixtures and regenerate reference-only scores. `HARDENING.md` records scientific headroom, shortcut replays and remaining limitations.
