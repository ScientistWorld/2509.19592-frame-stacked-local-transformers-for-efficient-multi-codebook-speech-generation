#!/usr/bin/env python3
"""Development scorer: the visible train-slice evaluation, written to scoring/scores.json."""
import sys

from eval.train.evaluate import main

if __name__ == "__main__":
    if not any(arg.startswith("--out") for arg in sys.argv[1:]):
        sys.argv += ["--out", "scoring/scores.json"]
    main()
