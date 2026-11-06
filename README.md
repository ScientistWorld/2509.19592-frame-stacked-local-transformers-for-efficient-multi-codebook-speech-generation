# Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.19592"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.19592-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2509.19592](https://arxiv.org/abs/2509.19592) |

## Summary

This reproduction implements the core computation from "Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation": a primary autoregressive decoder, a no-LT parallel codebook baseline, autoregressive and MaskGIT local transformers, and frame stacking.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +104.0 |
| `gpt-5.6-sol-xhigh` | +101.4 |
| `gpt-5.6-terra-xhigh` | +100.9 |
| `gpt-5.6-luna-xhigh` | +93.7 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2509.19592 --mode innovate --agent claude-code
sw evaluate 2509.19592
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

