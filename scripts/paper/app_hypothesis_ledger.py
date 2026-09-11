"""The hypothesis ledger as an appendix table, read from docs/hypothesis/README.md.

The ledger index lists every hypothesis with its verdict, and the paper's
appendix reprints that index rather than a retyped copy of it, so a verdict
changed in the ledger changes in the paper on the next build. Each row keeps the
id, the claim and the leading verdict word of the status cell. The rest of the
status cell is the ledger's own commentary and stays in the ledger.

    PYTHONPATH=. python scripts/paper/app_hypothesis_ledger.py --paper-dir paper
"""

import collections
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    build_parser,
    provenance_comment,
    tex_escape,
    write_macros,
)

GENERATOR = "scripts/paper/app_hypothesis_ledger.py"
LEDGER = os.path.join("docs", "hypothesis", "README.md")
# Checked in this order, so a partial verdict is read as partial and never as
# the bare word that follows it.
VERDICT_WORDS = (
    "PARTIALLY",
    "SUPPORTED",
    "REFUTED",
    "CONFIRMED",
    "INCONCLUSIVE",
    "DROPPED",
    "RETIRED",
    "OPEN",
    "PRE-REGISTERED",
)


def index_rows(path: str) -> list[dict[str, str]]:
    """id, claim and status cell of every row in the ledger's index table."""
    rows = []
    with open(path) as handle:
        for line in handle:
            match = re.match(r"\|\s*\[(H\d+)\]\([^)]*\)\s*\|(.*)\|(.*)\|\s*$", line)
            if match is None:
                continue
            rows.append(
                {
                    "id": match.group(1),
                    "claim": match.group(2).strip(),
                    "status": match.group(3).strip(),
                }
            )
    return rows


def leading_verdict(status: str) -> str:
    """The first verdict word in a status cell, `Partially supported` style."""
    stripped = status.replace("*", "")
    for word in VERDICT_WORDS:
        if word in stripped.upper():
            index = stripped.upper().find(word)
            fragment = stripped[index : index + 40].split(",")[0].split("(")[0]
            fragment = fragment.split(".")[0]
            fragment = fragment.split(":")[0].split(" as ")[0].strip(" .;")
            verdict = fragment.lower().capitalize()
            return verdict
    return stripped.split(",")[0][:30]


def clean_claim(claim: str) -> str:
    """Markdown emphasis and backticks removed, so the cell sets as plain text."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", claim)
    text = text.replace("**", "").replace("`", "")
    return text


def write_longtable(path: str, rows: list[dict[str, str]]) -> None:
    """A longtable, since the ledger runs past 1 page."""
    lines = [
        provenance_comment(GENERATOR, [LEDGER]),
        r"\begin{longtable}{p{0.06\linewidth}p{0.62\linewidth}p{0.26\linewidth}}",
        r"\caption{The hypothesis ledger, 1 row per hypothesis file under "
        r"docs/hypothesis/, with the leading verdict of its status cell. The "
        r"ledger's fuller commentary stays in the ledger.}"
        r"\label{tab:hypothesis-ledger}\\",
        r"\toprule",
        r"id & claim & verdict \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"id & claim & verdict \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in rows:
        lines.append(
            f"{row['id']} & {tex_escape(clean_claim(row['claim']))} & "
            f"{tex_escape(leading_verdict(row['status']))} \\\\"
        )
    lines += [r"\end{longtable}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = build_parser(__doc__).parse_args()
    rows = index_rows(LEDGER)
    if not rows:
        raise SystemExit(f"no index rows found in {LEDGER}")
    write_longtable(os.path.join(args.paper_dir, "tables", "hypothesis_ledger.tex"), rows)

    verdicts = collections.Counter(
        leading_verdict(row["status"]).split()[0].lower() for row in rows
    )
    macros = {
        "hypothesis_count": (str(len(rows)), "hypotheses in the ledger index"),
        "hypothesis_refuted": (
            str(verdicts.get("refuted", 0)),
            "hypotheses whose leading verdict is refuted",
        ),
        "hypothesis_supported": (
            str(verdicts.get("supported", 0) + verdicts.get("confirmed", 0)),
            "hypotheses whose leading verdict is supported or confirmed",
        ),
        "hypothesis_partial": (
            str(verdicts.get("partially", 0)),
            "hypotheses whose leading verdict is partial",
        ),
        "hypothesis_inconclusive": (
            str(verdicts.get("inconclusive", 0)),
            "hypotheses whose leading verdict is inconclusive",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "hypothesis_ledger.macros.json"),
        GENERATOR,
        [LEDGER],
        macros,
    )
    print(f"hypothesis ledger: {len(rows)} rows, verdicts {dict(verdicts)}")


if __name__ == "__main__":
    main()
