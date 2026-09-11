"""Deprecated: use `python -m cli.compare detectors`. Forwards argv there.

Every function and constant this module used to define now lives in cli.compare,
prefixed `detectors_` (bare for the module-level constants). Re-exported here
under the old spelling so a script that imports from here directly keeps
working. `psbd_values` and `psbd_rate` are read by several generators under
scripts/paper/. `active_columns`, `common_coverage` and `metric` are exercised
directly by tests/test_compare_detectors.py.

Example
    python -m cli.compare_detectors --fpr 0.01 0.05
"""

import sys

from cli.compare import detectors_active_columns as active_columns
from cli.compare import detectors_benign_table as benign_table
from cli.compare import detectors_bootstrap_interval as bootstrap_interval
from cli.compare import detectors_cell_text as cell_text
from cli.compare import detectors_collect_columns as collect_columns
from cli.compare import detectors_common_coverage as common_coverage
from cli.compare import detectors_cost_table as cost_table
from cli.compare import detectors_detector_values as detector_values
from cli.compare import detectors_footer as footer
from cli.compare import detectors_metric as metric
from cli.compare import detectors_metric_table as metric_table
from cli.compare import detectors_paired_deltas as paired_deltas
from cli.compare import detectors_panel_cells as panel_cells
from cli.compare import detectors_per_detector_blocks as per_detector_blocks
from cli.compare import detectors_psbd_rate as psbd_rate
from cli.compare import detectors_psbd_values as psbd_values
from cli.compare import detectors_render as render
from cli.compare import detectors_rewrite_results_block as rewrite_results_block
from cli.compare import detectors_subsets as subsets
from cli.compare import detectors_write_csv as write_csv
from cli.compare import DETECTORS_DEFAULT_BOOTSTRAP as DEFAULT_BOOTSTRAP
from cli.compare import DETECTORS_DEFAULT_FPRS as DEFAULT_FPRS
from cli.compare import DETECTORS_DOC_OF as DOC_OF
from cli.compare import DETECTORS_PSBD_COLUMNS as PSBD_COLUMNS
from cli.compare import DETECTORS_RESULTS_BLOCK_BEGIN as RESULTS_BLOCK_BEGIN
from cli.compare import DETECTORS_RESULTS_BLOCK_END as RESULTS_BLOCK_END
from cli.compare import DETECTORS_SMALL_POSITIVE_SET as SMALL_POSITIVE_SET
from cli.compare import main
from cli.compare import read_json

__all__ = [
    "read_json",
    "panel_cells",
    "psbd_rate",
    "psbd_values",
    "detector_values",
    "collect_columns",
    "metric",
    "cell_text",
    "bootstrap_interval",
    "subsets",
    "active_columns",
    "common_coverage",
    "metric_table",
    "paired_deltas",
    "benign_table",
    "cost_table",
    "footer",
    "render",
    "write_csv",
    "rewrite_results_block",
    "per_detector_blocks",
    "PSBD_COLUMNS",
    "DEFAULT_FPRS",
    "SMALL_POSITIVE_SET",
    "DEFAULT_BOOTSTRAP",
    "RESULTS_BLOCK_BEGIN",
    "RESULTS_BLOCK_END",
    "DOC_OF",
]

if __name__ == "__main__":
    print(
        "cli.compare_detectors is deprecated, use `python -m cli.compare detectors`",
        file=sys.stderr,
    )
    sys.exit(main(["detectors", *sys.argv[1:]]))
