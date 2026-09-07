"""Deprecated path. The summary now lives in cli.summary.

This file used to hold the implementation while cli/summary.py held a fork of it,
and the fork had drifted to the schema audit findings A4 and A19 were about while
still defaulting its output to the live CSV. Two parsers whose failure mode is a
plausible wrong label is the hazard this project keeps paying for.

The implementation moved to cli/summary.py because that is the destination the
port map declares and because cli/ may not import from scripts/. This shim stays
so the older invocation keeps working.

Example
    python -m cli.summary
"""

import sys

from cli.summary import main, parse_args

if __name__ == "__main__":
    sys.exit(main())

# Re-exported so the flag-parity check in tests/test_cli.py still compares the
# 2 invocation paths rather than skipping.
__all__ = ["main", "parse_args"]
