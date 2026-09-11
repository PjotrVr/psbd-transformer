"""Deprecated: use `python -m cli.compare placements variants`. Forwards argv there.

Every function and constant this module used to define now lives in cli.compare,
prefixed `variants_` (bare for the module-level constants). Re-exported here
under the old spelling so a script that imports from here directly keeps working.

Example
    python -m cli.variants --held-out vit_cifar10_sig_0_1 vit_cifar10_lc_0_1
"""

import sys

from cli.compare import main
from cli.compare import PLACEMENT_VARIANTS as VARIANTS
from cli.compare import variants_collect_rows as collect_rows
from cli.compare import variants_evaluate as evaluate
from cli.compare import variants_score_split as score_split
from cli.compare import variants_select_rate as select_rate
from cli.compare import variants_show as show
from cli.compare import variants_variant_cells as variant_cells
from cli.compare import VARIANTS_TABLE_WIDTH as TABLE_WIDTH

__all__ = [
    "VARIANTS",
    "TABLE_WIDTH",
    "score_split",
    "select_rate",
    "evaluate",
    "collect_rows",
    "variant_cells",
    "show",
]

if __name__ == "__main__":
    print(
        "cli.variants is deprecated, use `python -m cli.compare placements variants`",
        file=sys.stderr,
    )
    sys.exit(main(["placements", "variants", *sys.argv[1:]]))
