"""Every headline macro with its value, its generator and the grade of the finding citing it.

Runs after headline.py, so it reads the folded headline.json. A macro cited by a
finding in findings.md inherits that finding's grade and status. A macro no
finding cites is listed as unregistered, which is the honest state of a number
that appears in a table but carries no claim. The table is the appendix a
reader uses to find where any number came from and how much it is trusted.

    PYTHONPATH=. python scripts/paper/ledger_macros.py --paper-dir paper
"""

import os
import re
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import build_parser, load_json, provenance_comment, tex_escape  # noqa: E402
from scripts.paper.findings_index import registry_rows  # noqa: E402

GENERATOR = "scripts/paper/ledger_macros.py"


def citing_findings(rows: list[dict]) -> dict[str, list[dict]]:
    """macro name to the findings whose claim cites it."""
    cited: dict[str, list[dict]] = {}
    for row in rows:
        for name in set(re.findall(r"\\([A-Za-z]+)", row["claim"])):
            cited.setdefault(name, []).append(row)
    return cited


def write_ledger(path: str, macros: dict, cited: dict[str, list[dict]]) -> None:
    lines = [
        provenance_comment(GENERATOR, ["paper/headline.json", "paper/findings.md"]),
        r"\begin{longtable}{p{0.30\linewidth}p{0.12\linewidth}p{0.24\linewidth}p{0.10\linewidth}p{0.14\linewidth}}",
        r"\caption{Every headline macro: its value, the generator that wrote it, and the "
        r"grade and status of the finding that cites it. Unregistered means a table "
        r"number no finding rests on.}\label{tab:macro-ledger}\\",
        r"\toprule",
        r"macro & value & generator & grade & status \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"macro & value & generator & grade & status \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for name in sorted(macros):
        entry = macros[name]
        findings = cited.get(name, [])
        grade = ", ".join(sorted({f["grade"] for f in findings})) if findings else "unregistered"
        status = ", ".join(sorted({f["status"] for f in findings})) if findings else "--"
        generator = entry["generator"].replace("scripts/paper/", "")
        lines.append(
            f"\\texttt{{{tex_escape(name)}}} & {tex_escape(str(entry['value']))} & "
            f"\\texttt{{{tex_escape(generator)}}} & {tex_escape(grade)} & {tex_escape(status)} \\\\"
        )
    lines += [r"\end{longtable}", ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = build_parser(__doc__).parse_args()
    macros = load_json(os.path.join(args.paper_dir, "headline.json")) or {}
    rows = registry_rows(os.path.join(args.paper_dir, "findings.md"))
    cited = citing_findings(rows)
    write_ledger(os.path.join(args.paper_dir, "tables", "macro_ledger.tex"), macros, cited)
    registered = sum(1 for name in macros if name in cited)
    print(f"macro ledger: {len(macros)} macros, {registered} cited by a finding")


if __name__ == "__main__":
    main()
