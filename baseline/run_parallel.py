#!/usr/bin/env python3
"""Train the paper's no-LT parallel multi-codebook baseline."""

import subprocess
import sys


if __name__ == "__main__":
    args = ["python", "-m", "method.train", "--variant", "none", *sys.argv[1:]]
    raise SystemExit(subprocess.call(args))
