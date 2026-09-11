"""Deprecated: use `python -m cli.compare operating-points`. Forwards argv there.

Every function and constant this module used to define now lives in cli.compare,
prefixed `lowfpr_` (bare for the module-level constant). Re-exported here under
the old spelling so a script that imports from here directly keeps working.

Example
    python -m cli.operating_points --poison-rate 0.01
"""

import sys

from cli.compare import LOWFPR_DEFAULT_FPRS as DEFAULT_FPRS
from cli.compare import lowfpr_best_placement as best_placement
from cli.compare import lowfpr_build_header as build_header
from cli.compare import lowfpr_discover_folders as discover_folders
from cli.compare import lowfpr_operating_points as operating_points
from cli.compare import lowfpr_render_row as render_row
from cli.compare import lowfpr_scores_at as scores_at
from cli.compare import lowfpr_tpr_at as tpr_at
from cli.compare import main

__all__ = [
    "DEFAULT_FPRS",
    "scores_at",
    "tpr_at",
    "operating_points",
    "best_placement",
    "discover_folders",
    "build_header",
    "render_row",
]

if __name__ == "__main__":
    print(
        "cli.operating_points is deprecated, use `python -m cli.compare "
        "operating-points`",
        file=sys.stderr,
    )
    sys.exit(main(["operating-points", *sys.argv[1:]]))
