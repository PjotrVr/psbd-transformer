"""Deprecated: use `python -m cli.compare placements tables`. Forwards argv there.

Every function and constant this module used to define now lives in cli.compare,
prefixed `tables_` (bare for the module-level constants). Re-exported here under
the old spelling so a script that imports from here directly keeps working.

Example
    python -m cli.tables --coverage-only
"""

import sys

from cli.compare import main
from cli.compare import tables_cell_score as cell_score
from cli.compare import tables_is_mismatched as is_mismatched
from cli.compare import tables_placement_name as placement_name
from cli.compare import tables_print_coverage as print_coverage
from cli.compare import tables_print_dataset_table as print_dataset_table
from cli.compare import tables_read_asr as read_asr
from cli.compare import tables_read_cell_metadata as read_cell_metadata
from cli.compare import tables_render_row as render_row
from cli.compare import tables_required_cells as required_cells
from cli.compare import tables_score_every_cell as score_every_cell
from cli.compare import tables_table_columns as table_columns
from cli.compare import TABLES_DATASETS as DATASETS
from cli.compare import TABLES_DEFAULT_FPRS as DEFAULT_FPRS
from cli.compare import TABLES_DEFAULT_RATE_RULE as DEFAULT_RATE_RULE
from cli.compare import TABLES_DEFAULT_SIGMA as DEFAULT_SIGMA
from cli.compare import TABLES_MAX_MISSING_SHOWN as MAX_MISSING_SHOWN
from cli.compare import TABLES_PANEL as PANEL
from cli.compare import TABLES_POISON_TAGS as POISON_TAGS
from cli.compare import TABLES_RATE_RULES as RATE_RULES
from cli.compare import TABLES_SIGMA_MISMATCH_MARKER as SIGMA_MISMATCH_MARKER
from cli.compare import TABLES_WEAK_ASR as WEAK_ASR

__all__ = [
    "is_mismatched",
    "read_cell_metadata",
    "read_asr",
    "required_cells",
    "cell_score",
    "placement_name",
    "table_columns",
    "score_every_cell",
    "print_coverage",
    "render_row",
    "print_dataset_table",
    "DATASETS",
    "POISON_TAGS",
    "PANEL",
    "DEFAULT_FPRS",
    "RATE_RULES",
    "DEFAULT_RATE_RULE",
    "DEFAULT_SIGMA",
    "SIGMA_MISMATCH_MARKER",
    "WEAK_ASR",
    "MAX_MISSING_SHOWN",
]

if __name__ == "__main__":
    print(
        "cli.tables is deprecated, use `python -m cli.compare placements tables`",
        file=sys.stderr,
    )
    sys.exit(main(["placements", "tables", *sys.argv[1:]]))
