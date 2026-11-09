# Direction

Work within the paper's problem family: learned generation of multi-codebook codec tokens that models the dependencies between codebooks inside a frame and between frames, at a reduced number of primary (coarse) generation steps.

The gym ranks generators at two fixed operating points: 12 and 6 primary steps per 24-frame utterance (2x and 4x frame stacking). The 6-step point is where the paper's own method loses quality, and it carries the larger weight. Quality there depends on how well a single coarse step can produce several mutually consistent, temporally coherent frames and carry utterance-level properties across coarse steps.

Acceptable directions: better primary or local sequence models, better ways to pass information between coarse steps, iterative or hybrid decoding inside a coarse step, better training objectives and schedules, better sampling procedures, scaling the model or the training budget, regularization and data use within the provided splits.

Out of scope (treated as shortcuts by the grader):

- generating outputs from, or repairing/filtering outputs with, the dataset's known generating rule or `data/generate_synthetic.py`;
- copying, splicing or lightly perturbing visible utterances;
- declaring a larger `stack_factor` than the generator actually uses;
- changing the data, the conditions, the evaluators or the comparison rows.
