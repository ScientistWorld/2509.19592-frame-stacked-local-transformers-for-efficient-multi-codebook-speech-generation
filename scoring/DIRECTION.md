# Direction

Develop learned generators that preserve joint codebook structure, temporal coherence, diversity and speaker conditioning while reducing sequential primary generation steps. In scope:
- improvements to models, local decoding and the information passed between primary steps;
- training schedules, regularization and sampling;
- model scaling within the available compute, provided the model generalizes rather than reproducing training utterances.

The data come from a hidden speech-like process. It has nonlinear dynamics, phone-like regimes and persistent utterance style, and its parameters are not public, so the training data are the only way to learn it. The larger-weight 4x operating point studies the high-stacking regime, where the local decoder must produce four temporally dependent frames from one primary state. A solution must improve the generated distribution itself, rather than optimize agreement with reference token IDs or exploit reported metadata.

A learned alternative that produces a better distribution is a scientific improvement, even when its architecture differs from the reference. That includes models with continuous latents that use the public FSQ codec grid.

These are shortcuts:
- calling or copying a hidden reference sampler;
- copying, editing or splicing visible examples, or assembling rows from looked-up training frames or segments;
- false compute declarations, including using a 2x model for the 4x row;
- modifying canonical data or fixed comparisons;
- substituting scores.

Generated tokens cannot reveal some of these, so a separate shortcut review checks them:
- actual sequential computation;
- splices of very short pieces;
- lookups of single training frames;
- copies of a small fraction of rows.
