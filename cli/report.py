"""Deprecated: use `python -m cli.compare placements report`. Forwards argv there.

Every function this module used to define now lives in cli.compare, prefixed
`report_`. The names below are re-exported under their old, unprefixed spelling
so a script that imports from here directly keeps working.

Example
    python -m cli.report --format markdown
"""

import sys

from cli.compare import main
from cli.compare import read_json
from cli.compare import report_cell as cell
from cli.compare import report_collect_rows as collect_rows
from cli.compare import report_print_negative_control as print_negative_control
from cli.compare import report_render_csv as render_csv
from cli.compare import report_render_markdown as render_markdown
from cli.compare import report_summarize_by_placement as summarize_by_placement
from cli.compare import REPORT_HEADLINE_QUANTILE_KEY as HEADLINE
from cli.compare import REPORT_MARKDOWN_COLUMN_COUNT as MARKDOWN_COLUMNS

__all__ = [
    "read_json",
    "cell",
    "collect_rows",
    "print_negative_control",
    "render_csv",
    "render_markdown",
    "summarize_by_placement",
    "HEADLINE",
    "MARKDOWN_COLUMNS",
]

if __name__ == "__main__":
    print(
        "cli.report is deprecated, use `python -m cli.compare placements report`",
        file=sys.stderr,
    )
    sys.exit(main(["placements", "report", *sys.argv[1:]]))
