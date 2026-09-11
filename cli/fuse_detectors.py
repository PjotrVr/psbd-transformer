"""Deprecated: use `python -m cli.compare fused`. Forwards argv there.

Every function and constant this module used to define now lives in cli.compare,
prefixed `fused_` (bare for the module-level constants). Re-exported here under
the old spelling so a script that imports from here directly keeps working:
`fuse` and `tpr_at_fpr` are exercised directly by tests/test_fuse_detectors.py.

Example
    python -m cli.fuse_detectors --detector strip --fpr 0.01 0.05 0.25
"""

import sys

from cli.compare import FUSED_COLUMNS as COLUMNS
from cli.compare import FUSED_DEFAULT_FPRS as DEFAULT_FPRS
from cli.compare import fused_column_label as column_label
from cli.compare import fused_detector_scores as detector_scores
from cli.compare import fused_discover_folders as discover_folders
from cli.compare import fused_folder_row as folder_row
from cli.compare import fused_fuse as fuse
from cli.compare import fused_psbd_scores as psbd_scores
from cli.compare import fused_render_table as render_table
from cli.compare import fused_tpr_at_fpr as tpr_at_fpr
from cli.compare import main

__all__ = [
    "COLUMNS",
    "DEFAULT_FPRS",
    "psbd_scores",
    "detector_scores",
    "fuse",
    "tpr_at_fpr",
    "discover_folders",
    "folder_row",
    "column_label",
    "render_table",
]

if __name__ == "__main__":
    print(
        "cli.fuse_detectors is deprecated, use `python -m cli.compare fused`",
        file=sys.stderr,
    )
    sys.exit(main(["fused", *sys.argv[1:]]))
