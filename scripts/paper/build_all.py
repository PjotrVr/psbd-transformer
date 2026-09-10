"""Build paper/ from results/ end to end, and refuse a chapter that carries a bare number.

Runs every generator under scripts/paper/ (tab_*, fig_*, mech_*, app_*), folds
their macro sidecars into headline.tex, renders the findings registry, then reads
every chapter and fails on a digit that is not inside a macro, a citation, a
reference, a label, an input path or a year, and on a macro headline.tex does not
define. A chapter that passes quotes only numbers a script produced.

    PYTHONPATH=. python scripts/paper/build_all.py --results-dir results
    PYTHONPATH=. python scripts/paper/build_all.py --check-only
"""

import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import build_parser, load_json  # noqa: E402

GENERATOR_PATTERNS = ("tab_*.py", "fig_*.py", "mech_*.py", "app_*.py")
# What a digit may legitimately sit inside. Everything else is a typed number.
ALLOWED = (
    r"\\(?:cite|citep|citet|ref|cref|Cref|label|input|include|includegraphics(?:\[[^\]]*\])?|eqref|pageref|bibliography)\{[^}]*\}",
    r"\\[A-Za-z]+",
    r"%[^\n]*",
    r"\b(?:19|20)\d\d\b",
    r"\$[^$]*\$",
)


def generator_scripts() -> list[str]:
    """Every generator, in name order, so a build is reproducible."""
    here = os.path.dirname(os.path.abspath(__file__))
    scripts = sorted(
        path
        for pattern in GENERATOR_PATTERNS
        for path in glob.glob(os.path.join(here, pattern))
    )
    return scripts


def run_script(path: str, args) -> bool:
    """Run 1 generator with the shared flags, printing its tail on failure."""
    command = [
        sys.executable,
        path,
        "--results-dir",
        args.results_dir,
        "--declaration",
        args.declaration,
        "--paper-dir",
        args.paper_dir,
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."}
    )
    name = os.path.relpath(path)
    if completed.returncode != 0:
        print(f"FAILED {name}\n{completed.stderr[-2000:]}")
        return False
    print(f"ok     {name}")
    return True


def strip_allowed(text: str) -> str:
    """The chapter text with every legitimate digit carrier removed."""
    stripped = text
    for pattern in ALLOWED:
        stripped = re.sub(pattern, " ", stripped)
    return stripped


def chapter_problems(path: str, defined: set[str]) -> list[str]:
    """Bare digits and undefined macros in 1 chapter, as messages with line numbers."""
    problems = []
    with open(path) as handle:
        lines = handle.read().split("\n")
    for number, line in enumerate(lines, start=1):
        for macro in re.findall(r"\\([A-Z][A-Za-z]+)", line):
            if macro not in defined:
                problems.append(
                    f"{path}:{number}: macro \\{macro} is not in headline.tex"
                )
        if re.search(r"\d", strip_allowed(line)):
            problems.append(f"{path}:{number}: a typed number: {line.strip()[:80]}")
    return problems


def check_chapters(paper_dir: str) -> list[str]:
    """Every problem in every chapter, or an empty list."""
    headline = load_json(os.path.join(paper_dir, "headline.json")) or {}
    preamble = os.path.join(paper_dir, "draft", "preamble.tex")
    defined = set(headline)
    if os.path.exists(preamble):
        with open(preamble) as handle:
            defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", handle.read()))
    problems = []
    for path in sorted(glob.glob(os.path.join(paper_dir, "chapters", "*.tex"))):
        problems += chapter_problems(path, defined)
    return problems


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--check-only", action="store_true", help="skip the generators")
    args = parser.parse_args()

    failed = []
    if not args.check_only:
        for path in generator_scripts():
            if not run_script(path, args):
                failed.append(path)
        here = os.path.dirname(os.path.abspath(__file__))
        for helper in ("headline.py", "findings_index.py"):
            if not run_script(os.path.join(here, helper), args):
                failed.append(helper)

    problems = check_chapters(args.paper_dir)
    for problem in problems:
        print(problem)
    print(f"{len(failed)} generators failed, {len(problems)} chapter problems")
    status = 1 if failed or problems else 0
    return status


if __name__ == "__main__":
    sys.exit(main())
