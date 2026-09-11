"""What every paper generator shares: where things are, how numbers are read and how tex is written.

A generator reads results/ and the declaration, computes 1 table, figure or set
of macros and writes it under paper/ with a first-line comment naming itself,
its inputs, the commit and the time. Nothing under paper/ is typed by hand. The
macros a generator contributes go to paper/tables/<name>.macros.json, and
scripts/paper/headline.py folds every sidecar into paper/headline.tex, so a
number that appears in prose has exactly 1 source.

    from scripts.paper._common import build_parser, write_table, write_macros
"""

import argparse
import json
import math
import os
import random
import statistics

from defences.decision import HEADLINE_QUANTILE
from utils.provenance import current_git_commit, utc_timestamp

PAPER_DIR = "paper"
TABLES_DIR = os.path.join(PAPER_DIR, "tables")
FIGURES_DIR = os.path.join(PAPER_DIR, "figures")
CHAPTERS_DIR = os.path.join(PAPER_DIR, "chapters")
DEFAULT_RESULTS_DIR = "results"
DEFAULT_DECLARATION = os.path.join("configs", "psbd_basis.json")
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 0
HEADLINE_KEY = f"q{HEADLINE_QUANTILE:.2f}"

# The criticality scale, defined once here and in paper/README.md in the same words.
GRADES = ("CRITICAL", "STRONG", "SUPPORTING", "WEAK", "NEGATIVE")


def build_parser(description: str) -> argparse.ArgumentParser:
    """The flags every generator takes: where results are and where paper/ is."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--declaration", default=DEFAULT_DECLARATION)
    parser.add_argument("--paper-dir", default=PAPER_DIR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def load_json(path: str) -> dict | None:
    """A JSON file, or None when it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        loaded = json.load(handle)
    return loaded


def load_coverage(results_dir: str) -> dict:
    """The coverage ledger, the single source of which cells the panel holds."""
    path = os.path.join(results_dir, "coverage", "coverage.json")
    coverage = load_json(path)
    if coverage is None:
        raise SystemExit(f"{path} does not exist, run scripts/coverage_ledger.py first")
    return coverage


def load_declaration(path: str) -> dict:
    declaration = load_json(path)
    if declaration is None:
        raise SystemExit(f"{path} does not exist")
    return declaration


def clearing_cells(coverage: dict) -> list[dict]:
    """The panel cells whose attack implanted, the only cells a detection number spans."""
    cells = [cell for cell in coverage["cells"] if cell.get("asr_class") == "clears"]
    return cells


def load_psbd_metrics(results_dir: str, folder: str) -> dict | None:
    """A checkpoint's psbd_metrics.json, or None when the sweep has not reached it."""
    report = load_json(os.path.join(results_dir, folder, "psbd_metrics.json"))
    return report


def rate_row(placement_block: dict, rate: float) -> dict | None:
    """The per-rate entry of a placement block at exactly this rate, or None."""
    for row in placement_block.get("rates", []):
        if row.get("rate") == rate:
            return row
    return None


def bootstrap_ci(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    """A 95% interval on the mean by resampling with replacement, seeded.

    Below 3 values there is nothing to resample and the interval is undefined.
    """
    if len(values) < 3 or resamples <= 0:
        return float("nan"), float("nan")
    generator = random.Random(seed)
    draws = sorted(
        statistics.mean(generator.choices(values, k=len(values)))
        for _ in range(resamples)
    )
    interval = (draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1])
    return interval


def mean_or_none(values: list[float]) -> float | None:
    """The mean, or None for an empty list so an empty aggregate prints as a dash."""
    if not values:
        return None
    mean = statistics.mean(values)
    return mean


def ci_text(low: float, high: float) -> str:
    """A bootstrap interval as [low, high] with signs, a dash when it is undefined."""
    if math.isnan(low) or math.isnan(high):
        return "--"
    text = f"[{fmt(low, signed=True)}, {fmt(high, signed=True)}]"
    return text


def fmt(value: float | None, places: int = 3, signed: bool = False) -> str:
    """A number for a table cell, `--` where there is none."""
    if value is None or value != value:
        return "--"
    text = f"{value:+.{places}f}" if signed else f"{value:.{places}f}"
    return text


def provenance_comment(generator: str, inputs: list[str]) -> str:
    """The first line of every generated file: who made it, from what, when, at which commit."""
    comment = (
        f"% generated by {generator} from {', '.join(inputs)} "
        f"at {utc_timestamp()}, commit {current_git_commit()}. Do not edit by hand."
    )
    return comment


def tex_escape(text: str) -> str:
    """Underscores and percent signs, the 2 characters a folder or attack name brings."""
    escaped = text.replace("_", r"\_").replace("%", r"\%")
    return escaped


def write_table(
    path: str,
    generator: str,
    inputs: list[str],
    caption: str,
    label: str,
    header: list[str],
    rows: list[list[str]],
    align: str | None = None,
) -> None:
    """A booktabs table as a complete .tex file, ready for \\input."""
    columns = align or "l" + "r" * (len(header) - 1)
    lines = [
        provenance_comment(generator, inputs),
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(tex_escape(cell) for cell in header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(cell) for cell in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def macro_name(text: str) -> str:
    """A LaTeX-legal macro name from a snake_case or spaced key, letters only."""
    words = text.replace("_", " ").replace("-", " ").split()
    name = "".join(word[:1].upper() + word[1:] for word in words if word)
    for digit, word in zip(
        "0123456789",
        (
            "Zero",
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
            "Nine",
        ),
    ):
        name = name.replace(digit, word)
    return name


def write_macros(
    sidecar_path: str,
    generator: str,
    inputs: list[str],
    macros: dict[str, tuple[str, str]],
) -> None:
    """A generator's macros as a JSON sidecar: name to (value text, what it is).

    headline.py folds every sidecar into paper/headline.tex and headline.json.
    """
    payload = {
        "generator": generator,
        "inputs": inputs,
        "written_at": utc_timestamp(),
        "git_commit": current_git_commit(),
        "macros": {
            macro_name(name): {"value": value, "meaning": meaning}
            for name, (value, meaning) in macros.items()
        },
    }
    os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
    with open(sidecar_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def figure_sidecar(path: str, generator: str, inputs: list[str], plotted: dict) -> None:
    """The numbers a figure shows, beside the figure, so a plot is never the only record."""
    payload = {
        "generator": generator,
        "inputs": inputs,
        "written_at": utc_timestamp(),
        "git_commit": current_git_commit(),
        "plotted": plotted,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


# Okabe-Ito, colourblind safe, in 1 fixed order so every figure colours the same
# attack the same way.
OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
DEFAULT_CHECKPOINTS_DIR = "checkpoints"
# Folder tokens that mark a checkpoint as outside the ViT panel: SAM ablations,
# adaptive-attacker evasions, seed replicates, strength and trigger sweeps.
NON_PANEL_TOKENS = ("sam_rho", "evade", "_ep", "a2m", "seed_", "_trig", "_pilot")


def build_parser_with_checkpoints(description: str) -> argparse.ArgumentParser:
    """build_parser plus --checkpoints-dir, for generators reading args.json sidecars."""
    parser = build_parser(description)
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR)
    return parser


def load_args_json(checkpoints_dir: str, folder: str) -> dict | None:
    """A checkpoint's training-provenance sidecar, or None when the folder has none."""
    sidecar = load_json(os.path.join(checkpoints_dir, folder, "args.json"))
    return sidecar


def is_panel_folder(folder: str) -> bool:
    """Whether a folder name carries none of the tokens that exclude it from the panel."""
    excluded = any(token in folder for token in NON_PANEL_TOKENS)
    return not excluded


def save_figure(figure, path: str) -> None:
    """Write a figure as PDF, creating the directory, and release it."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def markdown_table_rows(path: str, first_header: str) -> list[dict[str, str]]:
    """The rows of the markdown table whose header starts with first_header, as dicts.

    A run entry under docs/runs/ is sometimes the only surviving record of a
    measurement, so a generator reading it names the entry as its input the same
    way it would name a JSON file. Cell text is returned verbatim, backticks
    stripped, and the caller parses numbers.
    """
    with open(path) as handle:
        lines = handle.read().split("\n")
    rows = []
    header = None
    for line in lines:
        if not line.startswith("|"):
            header = None if header is not None and rows else header
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if header is None:
            if cells and cells[0] == first_header:
                header = cells
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def std_or_none(values: list[float]) -> float | None:
    """The sample standard deviation, or None below 2 values."""
    if len(values) < 2:
        return None
    deviation = statistics.stdev(values)
    return deviation
