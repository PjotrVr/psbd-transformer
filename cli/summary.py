"""Deprecated: use `python -m cli.compare placements summary`. Forwards argv there.

Every function and constant this module used to define now lives in cli.compare,
the placement-name-parsing vocabulary under its original bare name and the rest
prefixed `summary_`. Re-exported here under the old spelling so a script that
imports from here directly keeps working.

Example
    python -m cli.summary --output results/detection_summary.csv
"""

import sys

from cli.compare import KNOWN_OPERATORS
from cli.compare import main
from cli.compare import MASK_SEED_SUFFIX
from cli.compare import MODEL_DROPOUT_SUFFIX
from cli.compare import BLOCK_RANGE_SUFFIX
from cli.compare import PASS_COUNT_SUFFIX
from cli.compare import placement_is_cache_backed
from cli.compare import PSU_VARIANTS
from cli.compare import QUALIFIER_SUFFIXES
from cli.compare import read_json
from cli.compare import SELECTION_RULES
from cli.compare import split_operator
from cli.compare import summary_operating_point_rows as operating_point_rows
from cli.compare import summary_rows_for_checkpoint as rows_for_checkpoint
from cli.compare import SUMMARY_FIELDS as FIELDS
from cli.compare import SUMMARY_OPERATING_POINT_FIELDS as OPERATING_POINT_FIELDS
from cli.compare import UNKNOWN_OPERATOR
from cli.compare import VALID_POSITION_NAMES
from cli.compare import VARIANT_SUFFIXES

__all__ = [
    "read_json",
    "split_operator",
    "placement_is_cache_backed",
    "operating_point_rows",
    "rows_for_checkpoint",
    "SELECTION_RULES",
    "VALID_POSITION_NAMES",
    "VARIANT_SUFFIXES",
    "PASS_COUNT_SUFFIX",
    "MODEL_DROPOUT_SUFFIX",
    "BLOCK_RANGE_SUFFIX",
    "MASK_SEED_SUFFIX",
    "KNOWN_OPERATORS",
    "UNKNOWN_OPERATOR",
    "QUALIFIER_SUFFIXES",
    "PSU_VARIANTS",
    "OPERATING_POINT_FIELDS",
    "FIELDS",
]

if __name__ == "__main__":
    print(
        "cli.summary is deprecated, use `python -m cli.compare placements summary`",
        file=sys.stderr,
    )
    sys.exit(main(["placements", "summary", *sys.argv[1:]]))
