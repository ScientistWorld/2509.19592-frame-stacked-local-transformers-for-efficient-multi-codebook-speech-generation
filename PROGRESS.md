# Progress

Current milestone: **core_claim_plus**, final hardening verified on 2026-11-09. The current evidence is [HARDENING.md](HARDENING.md) and [VERIFICATION.md](VERIFICATION.md); older stages are recorded in [MILESTONES.md](MILESTONES.md).

The paper's AR local-transformer mechanism is reproduced at reduced synthetic scale with slot-specific shared embeddings, averaged previous-stack inputs, free-running generation and matched-depth parallel comparisons. The final training data include temporal structure and persistent speaker-conditioned styles. The reference and comparisons were trained on these data before the final evaluator-only patch batch. That batch regenerated all outputs from saved weights for the enlarged paired evaluation rather than retraining.

Job `c0959cf3-667` exited 0 and passed `python validate.py --post-design`. The reference beats parallel comparisons on all scored measurements at 2x and 4x. A valid longer-trained variant improves all six scored cells and scores +0.047042 above the reference; its paired 95% interval [+0.041183, +0.058011] excludes zero and exceeds observed seed ranges. Weaker controls lose. No scored reference metric is saturated.

Test answers, their generator, old shortcut outputs and test seeds are restricted to `eval/test/`. Public `data/generate_synthetic.py` creates only conditioning inputs; original training and validation tensors are tracked. Independent train/test evaluators freeze the comparison outputs, references, dimensions, operating points and thresholds. They recompute all measurements from raw tokens, fail invalid artifacts, and rank failed runs below feasible ones before adding standardized, direction-aware metric improvements. All replayed shortcut cases fail. Final score files contain reference runs alone; audit outputs are under `eval/test/hardening/`.

The complete MaskGIT claim and real-speech evaluation remain unreproduced. Analytic inference of the synthetic distribution, partial splicing and false compute declarations cannot all be rejected from saved tensors and remain subject to the separate shortcut review. These limitations prevent a claim of universal scientific ordering.

`scripts/run.sh` reproduces reference and comparison training, outputs, fixtures, both evaluations and structural validation. The paper's reported metrics remain unchanged under `_paper_reported` in `scoring/reference.json`. The container builds from `environment/`; no committed executable code references `shared/`.
