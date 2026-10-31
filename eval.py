"""Compatibility entry point: python eval.py [new-output.json]."""

import sys
from eval_lab.__main__ import main

if __name__ == "__main__":
    arguments = ["run", "--suite", "smoke", "--prompt", "clarified"]
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python eval.py [new-output.json]; see python -m eval_lab --help")
    if len(sys.argv) == 2:
        arguments += ["--output", sys.argv[1]]
    main(arguments)
