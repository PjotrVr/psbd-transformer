"""Build paper/ from results/ end to end, and refuse a section that carries a bare number.

Runs every generator under scripts/paper/ (tab_*, fig_*, mech_*, app_*), folds
their macro sidecars into headline.tex, renders the findings registry, runs the
ledgers (ledger_*) that read the folded headline, then reads every section and
reports a hand-typed measurement or a macro headline.tex does not define.

A measurement is what has to come from a script: a decimal, a signed decimal or
an integer of 3 digits or more, and a bare count immediately followed by a word
naming a population we measure. A bare 1 or 2 digit integer on its own is a count
in prose such as 2 axes, and it is left alone. Forcing
every such digit into a macro produced 147 reports nobody could act on and hid
the real ones among them.

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
# A ledger reads headline.json, so it runs after headline.py has folded the sidecars.
LEDGER_PATTERNS = ("ledger_*.py",)
# Generators the paper deliberately reads nothing from. The related-work section
# declares sharpness-aware training out of scope, so its 2 generators stay
# runnable by hand and are not part of a paper build. The ledgers are audit
# records of the macro and hypothesis registries rather than paper content.
NOT_IN_PAPER = {
    "tab_sam.py": "SAM is declared out of scope in the related-work section",
    "tab_sam_low_rate.py": "SAM is declared out of scope in the related-work section",
}
LEDGERS = {"app_hypothesis_ledger.py", "ledger_macros.py"}
# What a digit may legitimately sit inside. Everything else is a typed number.
ALLOWED = (
    r"\\(?:cite|citep|citet|cited|ref|cref|Cref|label|input|include|includegraphics(?:\[[^\]]*\])?|eqref|pageref|bibliography)\{[^}]*\}",
    r"\\[A-Za-z]+",
    r"%[^\n]*",
    r"\b(?:19|20)\d\d\b",
    r"\$[^$]*\$",
    # Proper names that carry a digit are names, not measurements: architectures,
    # datasets, ported detectors and the hypothesis and finding ids.
    r"\b(?:ViT-B/16|ResNet-18|VGG16|CIFAR-10|CIFAR-100|H\d{1,2}|F\d{2})\b",
)
# A digit that survives ALLOWED is a measurement when it carries a decimal point,
# or is long enough to be a population size, or is a count of something the panel
# measures. Anything else is an ordinary count in prose.
MEASUREMENT = re.compile(
    r"[-+\u2212]?\d*\.\d+"
    r"|\b\d{3,}\b"
    # 2 digits and up, because a single-digit count before one of these words is
    # structural, as in 2 placements or 3 seeds, while 65 models is a population.
    r"|\b\d{2}\s+(?:models?|cells?|checkpoints?|datasets?|attacks?|placements?|"
    r"seeds?|probes?|detectors?)\b"
)
# Standard LaTeX commands whose name starts with a capital letter, so the
# undefined-macro check does not read them as headline macros.
LATEX_COMMANDS = {
    "Cref",
    "IfFileExists",
    "FloatBarrier",
    "Phi",
    "Sigma",
    "Delta",
    "Gamma",
    "Lambda",
    "Omega",
    "Theta",
    "Pi",
    "Psi",
    "Xi",
    "LaTeX",
    "TeX",
}


def generator_scripts(patterns: tuple[str, ...] = GENERATOR_PATTERNS) -> list[str]:
    """Every generator matching the patterns, in name order, so a build is reproducible."""
    here = os.path.dirname(os.path.abspath(__file__))
    scripts = sorted(
        path
        for pattern in patterns
        for path in glob.glob(os.path.join(here, pattern))
        if os.path.basename(path) not in NOT_IN_PAPER
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
    """The section text with every legitimate digit carrier removed."""
    stripped = text
    for pattern in ALLOWED:
        stripped = re.sub(pattern, " ", stripped)
    return stripped


def section_problems(path: str, defined: set[str]) -> list[str]:
    """Bare digits and undefined macros in 1 section, as messages with line numbers."""
    problems = []
    with open(path) as handle:
        lines = handle.read().split("\n")
    # A display equation is notation, so its lines are skipped the way inline
    # math is stripped. The environment is tracked line by line.
    in_display_math = False
    for number, line in enumerate(lines, start=1):
        for macro in re.findall(r"\\([A-Z][A-Za-z]+)", line):
            if macro not in defined and macro not in LATEX_COMMANDS:
                problems.append(
                    f"{path}:{number}: macro \\{macro} is not in headline.tex"
                )
        if re.search(r"\\begin\{(?:equation|align|gather|split)\*?\}|\\\[", line):
            in_display_math = True
        if in_display_math:
            if re.search(r"\\end\{(?:equation|align|gather|split)\*?\}|\\\]", line):
                in_display_math = False
            continue
        typed = MEASUREMENT.findall(strip_allowed(line))
        if typed:
            problems.append(f"{path}:{number}: hand-typed {typed}: {line.strip()[:70]}")
    return problems


def check_sections(paper_dir: str) -> list[str]:
    """Every problem in every section, or an empty list."""
    headline = load_json(os.path.join(paper_dir, "headline.json")) or {}
    defined = set(headline)
    # The preamble's own commands and environments are legitimate macro names too.
    for preamble in (os.path.join(paper_dir, "preamble.tex"),):
        if not os.path.exists(preamble):
            continue
        with open(preamble) as handle:
            text = handle.read()
        defined |= set(re.findall(r"\\newcommand\*?\{\\([A-Za-z]+)\}", text))
        defined |= set(re.findall(r"\\newenvironment\{([A-Za-z]+)\}", text))
        defined |= set(re.findall(r"\\newif\\if([A-Za-z]+)", text))
    problems = []
    for path in sorted(glob.glob(os.path.join(paper_dir, "sections", "*.tex"))):
        problems += section_problems(path, defined)
    return problems


def generator_outputs(paper_dir: str) -> dict[str, dict[str, set[str]]]:
    """What each generator wrote into paper/, keyed by the generator's file name.

    A table carries its generator in its first-line provenance comment and a
    macro sidecar or a figure sidecar carries it as a field, so no generator has
    to register what it writes.
    """
    outputs: dict[str, dict[str, set[str]]] = {}

    def entry(generator: str) -> dict[str, set[str]]:
        key = os.path.basename(generator)
        return outputs.setdefault(
            key, {"tables": set(), "figures": set(), "macros": set()}
        )

    for path in glob.glob(os.path.join(paper_dir, "tables", "*.tex")):
        with open(path) as handle:
            match = re.search(r"generated by (\S+)", handle.readline())
        if match:
            entry(match.group(1))["tables"].add(os.path.basename(path)[:-4])
    for path in glob.glob(os.path.join(paper_dir, "tables", "*.macros.json")):
        payload = load_json(path) or {}
        entry(payload.get("generator", "?"))["macros"].update(payload.get("macros", {}))
    for path in glob.glob(os.path.join(paper_dir, "figures", "*.json")):
        payload = load_json(path) or {}
        entry(payload.get("generator", "?"))["figures"].add(os.path.basename(path)[:-5])
    return outputs


def orphan_problems(paper_dir: str) -> list[str]:
    """Generators whose every output the paper ignores.

    10 generators ran on every build for months while the paper read nothing
    they wrote, and 2 of them carried captions that did not compile, which
    nobody saw because nothing input them. A generator is either read, listed in
    NOT_IN_PAPER with its reason, or a ledger.
    """
    text = ""
    for path in glob.glob(os.path.join(paper_dir, "sections", "*.tex")):
        with open(path) as handle:
            text += handle.read()
    with open(os.path.join(paper_dir, "main.tex")) as handle:
        text += handle.read()
    cited = set(re.findall(r"\\([A-Z][A-Za-z]+)", text))
    tables = set(re.findall(r"input\{tables/([A-Za-z0-9_]+)\}", text))
    figures = set(
        re.findall(r"includegraphics(?:\[[^\]]*\])?\{([A-Za-z0-9_]+)(?:\.pdf)?\}", text)
    )
    figures |= set(re.findall(r"input\{figures/([A-Za-z0-9_]+)\}", text))

    problems = []
    for generator, produced in sorted(generator_outputs(paper_dir).items()):
        if generator in NOT_IN_PAPER or generator in LEDGERS:
            continue
        read = (
            produced["macros"] & cited
            or produced["tables"] & tables
            or produced["figures"] & figures
        )
        if not read:
            problems.append(
                f"{generator}: the paper reads none of its output, input it or list "
                "it in NOT_IN_PAPER with a reason"
            )
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
        for path in generator_scripts(LEDGER_PATTERNS):
            if not run_script(path, args):
                failed.append(path)

    problems = check_sections(args.paper_dir) + orphan_problems(args.paper_dir)
    for problem in problems:
        print(problem)
    print(f"{len(failed)} generators failed, {len(problems)} section problems")
    status = 1 if failed or problems else 0
    return status


if __name__ == "__main__":
    sys.exit(main())
