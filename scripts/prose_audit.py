"""Count the prose-style violations in comments and docstrings, file by file.

Prose here means comments and docstrings, extracted with the tokenizer and the
parser rather than by regex over raw lines, so a semicolon in code is never
counted and a semicolon in a comment always is. The rules are the project's
(CLAUDE.md, Code style) plus one more: no Oxford comma.

    python scripts/prose_audit.py                  # totals for the library and cli
    python scripts/prose_audit.py attacks cli      # only these directories
    python scripts/prose_audit.py --show attacks   # every hit with file:line

Each rule is a heuristic and the number-word rule in particular over-reports
idioms such as "one another"; the listing exists so a person can judge each hit.
Exit status is always 0, since this reports rather than gates.
"""

import argparse
import ast
import io
import os
import re
import tokenize

DEFAULT_ROOTS = (
    "attacks",
    "analysis",
    "cli",
    "data",
    "defences",
    "detectors",
    "evaluation",
    "models",
    "training",
    "utils",
    "scripts",
)

NUMBER_WORDS = (
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)\b"
)
# Idiomatic uses of a number word that are not counts and must not be rewritten.
NUMBER_IDIOMS = re.compile(
    r"\b(one another|no one|one of\b|every one|not one|the one\b|each one|which one|"
    r"this one|that one|other one|any one|a single one|clean one|confident one|"
    r"same one|first one|last one|wrong one|right one|new one|old one|one-|two-|"
    r"three-|one\'s|someone|anyone|everyone|none)",
    re.IGNORECASE,
)
AI_TELLS = (
    "note that",
    "it is worth noting",
    "in order to",
    "is responsible for",
    "leverage",
    "seamless",
    "ensure that",
    "simply ",
    "essentially",
    "utilize",
    "this module provides",
    "this function ",
    "this class ",
    "it should be noted",
    "in this module",
    "in this file",
    "for the purpose of",
)


def prose_of(path: str) -> list[tuple[int, str, str]]:
    """Every comment and docstring in a file as (line, kind, text)."""
    source = open(path).read()
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            found.append((token.start[0], "comment", token.string.lstrip("#").strip()))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            text = ast.get_docstring(node, clean=False)
            if text:
                line = node.body[0].lineno
                pieces = text.splitlines()
                # A line indented past the docstring's own body is a formula or
                # pseudocode block, which the style rules do not apply to.
                body_indent = min(
                    (len(p) - len(p.lstrip()) for p in pieces[1:] if p.strip()),
                    default=0,
                )
                for offset, piece in enumerate(pieces):
                    indent = len(piece) - len(piece.lstrip())
                    formula = offset > 0 and indent >= body_indent + 4
                    found.append(
                        (line + offset, "formula" if formula else "docstring", piece)
                    )
    return found


def strip_code_spans(text: str) -> str:
    """Prose with backticked spans and bracketed math removed, since those are code."""
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\([^()]*\)", "", text)
    text = re.sub(r"\[[^\[\]]*\]", "", text)
    return text


def violations(line: int, kind: str, text: str) -> list[tuple[str, int, str]]:
    if kind == "formula":
        return []
    plain = strip_code_spans(text)
    hits = []
    if ";" in plain and not re.search(r"\w;\w", plain):
        hits.append(("semicolon", line, text))
    if re.search(r"(-->|->|=>|<-)", plain):
        hits.append(("arrow", line, text))
    if re.search(r"(—|–| -- )", text):
        hits.append(("em dash", line, text))
    # On the raw text, since stripping a parenthesis can manufacture ", and".
    without_code = re.sub(r"`[^`]*`", "", text)
    if re.search(r",\s+(and|or)\s", without_code) and without_code.count(",") >= 2:
        hits.append(("oxford comma", line, text))
    for match in re.finditer(NUMBER_WORDS, plain, re.IGNORECASE):
        window = plain[max(0, match.start() - 12) : match.end() + 10]
        if not NUMBER_IDIOMS.search(window):
            hits.append(("number word", line, text))
            break
    if kind == "comment" and re.match(r"^[-=*#~_]{4,}", text):
        hits.append(("banner", line, text))
    lowered = plain.lower()
    for tell in AI_TELLS:
        if tell in lowered:
            hits.append(("ai tell", line, text))
            break
    return hits


def python_files(roots) -> list[str]:
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in sorted(filenames):
                if name.endswith(".py"):
                    yield os.path.join(directory, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="directories or files to audit",
    )
    parser.add_argument(
        "--show", action="store_true", help="print every hit with file:line"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots, show = args.roots, args.show

    totals: dict[str, int] = {}
    per_file: dict[str, dict[str, int]] = {}
    for path in python_files(roots):
        for line, kind, text in prose_of(path):
            for rule, at, snippet in violations(line, kind, text):
                totals[rule] = totals.get(rule, 0) + 1
                per_file.setdefault(path, {}).setdefault(rule, 0)
                per_file[path][rule] += 1
                if show:
                    print(f"{path}:{at}: [{rule}] {snippet.strip()[:100]}")

    print("\nrule              hits")
    for rule in sorted(totals, key=totals.get, reverse=True):
        print(f"  {rule:16} {totals[rule]:5}")
    print(f"  {'files with hits':16} {len(per_file):5}")
    if not show:
        worst = sorted(per_file.items(), key=lambda item: -sum(item[1].values()))[:10]
        print("\nworst files")
        for path, counts in worst:
            print(f"  {sum(counts.values()):4}  {path}  {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
