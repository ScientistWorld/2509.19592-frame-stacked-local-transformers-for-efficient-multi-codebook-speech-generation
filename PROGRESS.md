# Progress

## What Works

- Official arXiv paper files are saved under `paper/`.
- `scoring/reference.json` records paper-reported LibriTTS unseen-speaker metrics for the no-LT baseline and LT variants.
- `data/generate_synthetic.py` creates a reproducible multi-codebook token dataset with intra-frame residual-style dependencies.
- `method/models.py` implements the paper's core algorithmic variants at reduced scale: parallel no-LT prediction, AR local transformer, MaskGIT local transformer, and frame stacking.
- `eval/evaluate_predictions.py` scores saved prediction artifacts without importing method code.

## Results Achieved

No compute job has completed yet. Current milestone remains `none`.

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
