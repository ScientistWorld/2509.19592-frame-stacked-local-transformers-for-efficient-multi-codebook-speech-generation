# Progress

## What Works

- Official arXiv paper files are saved under `paper/`.
- `scoring/reference.json` records paper-reported LibriTTS unseen-speaker metrics for the no-LT baseline and LT variants, plus a clearly labeled synthetic replacement target group used by this gym's evaluator.
- `data/generate_synthetic.py` creates a reproducible multi-codebook token dataset with intra-frame residual-style dependencies.
- `method/models.py` implements the paper's core algorithmic variants at reduced scale: parallel no-LT prediction, AR local transformer, MaskGIT local transformer, and frame stacking.
- `eval/evaluate_predictions.py` scores saved prediction artifacts without importing method code.

## Results Achieved

Job `e194b6d3-0a8` generated the synthetic dataset, trained `baseline_parallel_1x` for 500 CPU test steps, and wrote `checkpoints/predictions/baseline_parallel_1x.pt`. It failed in evaluation before any LT model ran because `codec_features` had a feature/codebook broadcast bug.

Job `27ab7f98-e2c` completed the full CPU smoke run. It trained and evaluated `baseline_parallel_1x`, `ar_lt_2x`, and `maskgit_lt_2x`. This reaches `method_runs`, but not `core_claim`: AR LT dependency validity was 0.032 versus baseline 0.038, indicating broken sampling. The AR sampling alignment is now fixed and will be re-evaluated from the existing checkpoints.

Job `30c8fb22-0d5` re-evaluated the existing checkpoints after the AR sampling fix. AR LT improved to dependency validity 0.055 versus baseline 0.038 and codec FD 0.106, but this is still too close to the trivial baseline to claim `core_claim`. The AR checkpoint was only trained for 500 steps and its NLL remained 1.157, above the expected deterministic-slot lower bound, so the next run retrains AR LT longer.

## Deviations From Paper

- Original training data: 18k hours from Koel-TTS. Replacement: synthetic codec-token sequences generated locally.
- Original codec: 21.5 fps, 8-codebook NanoCodec with audio reconstruction. Replacement: 8-codebook synthetic codec tokens with deterministic dependency rules and fixed codec-frame embeddings.
- Original metrics: WER, TitaNet speaker similarity, NanoCodec FD, UTMOSv2, and relative throughput. Replacement metrics: dependency validity, token accuracy, deterministic codec FD, NLL, and primary decoder steps.
- Original scale: 16 total transformer layers, dimension 768, 220k steps. Replacement scale: small transformer models for a reusable gym.

## What Remains

- Run `scripts/reproduce.sh` on a compute node.
- Confirm `baseline_parallel_1x` produces lower dependency validity than `ar_lt_2x` and `maskgit_lt_2x`.
- Update milestones and scores after the job completes.

## Issues Encountered

- `/home/user/paper/2509.19592.md` and `.pdf` were not present initially; the official arXiv files were fetched instead.
- No paper-specific code release was found.
- Job `e194b6d3-0a8` exposed a tensor broadcasting bug in `data/codec_dataset.py`; fixed by giving the codebook index tensor an explicit feature singleton dimension.
- Job `27ab7f98-e2c` exposed an AR local-transformer sampling bug: generated previous slots were not shifted into the same positions used during teacher-forced training.
- Job `30c8fb22-0d5` showed the fixed 500-step AR checkpoint was undertrained; next job retrains only AR LT for more steps while reusing the baseline.
