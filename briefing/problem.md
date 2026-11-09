# Problem Definition

Speech generation systems that operate on neural audio codec tokens must predict a matrix of discrete tokens with shape `(time, codebook)`. At every acoustic frame, several codebooks jointly describe the same audio frame, and they are not independent of each other. A generator must produce codebook combinations that are internally consistent within each frame, coherent from frame to frame, faithful to the conditioning speaker, and as diverse as real speech, not just plausible marginal tokens.

Generation cost matters as much as quality. A generator that runs a large sequence model once per frame is slow; the gym therefore fixes **operating points**: the number of coarse (primary) sequential generation steps the generator may spend per 24-frame utterance. The scored operating points are **12 steps** and **6 steps** per 24 frames, i.e. every coarse step has to produce 2 or 4 complete frames (16 or 32 codebook tokens). Any additional per-step computation inside a coarse step is allowed. Quality is compared at a fixed operating point.

The gym uses a reproducible synthetic codec because the paper's 18k-hour training mixture, codec tokenizer and trained models are not released. Each utterance has a speaker id (16 speakers, all seen in training) and 24 frames of 8 codebooks with vocabulary 32. Its structure (implemented in `data/generate_synthetic.py`):

- Within a frame, each residual codebook is determined by the previous codebook, the speaker, the codebook index, the frame position and one residual "style" bit.
- Across frames, the coarse (first) codebook follows a random walk with four allowed increments, and the seven residual style bits are fixed for the whole utterance (drawn with speaker-specific probabilities).

So within-frame codebooks are strongly dependent, frames depend on the previous frame, utterance-level style must be carried through the whole utterance, and a coarse step that produces several frames must keep them mutually consistent. Producing every codebook independently matches the marginals but breaks the joint structure.

**Task.** Learn a generator from the training split (`data/synthetic_codec/train.pt`, optionally `val.pt`: 8192 and 1024 utterances with speaker ids) that samples new utterances conditioned on a speaker id. For each of the 4096 conditioning speakers in `data/synthetic_codec/test_conditions.pt`, generate one 24-frame utterance at each operating point. Generated utterances are compared, as a distribution, against reference utterances of the same speakers that the evaluator draws itself; there is no per-example target to predict.

Hard requirements:

- Generate; do not copy. Outputs that reproduce long stretches of visible utterances are failed.
- The generator must be learned from data. Writing the dataset's generating rule (or calling the data generator) into the method to produce outputs is out of scope (see `scoring/DIRECTION.md`).
- Respect the operating point you declare (`stack_factor` in the artifact): the number of coarse sequential steps per 24 frames must not exceed 24 / `stack_factor`.
