"""Count the prose-style violations in 1 file or 1 tree, whatever the language.

Prose means the text a person reads: comments and docstrings in Python, the body
of a markdown document, the body of a LaTeX section. 1 extractor per language
hands the shared rule engine a list of (line, kind, text), so the rules are
written once and every language gets all of them. Python is read with the
tokenizer and the parser rather than by regex, so a semicolon in code is never
counted and a semicolon in a comment always is. Markdown and LaTeX have their
code, math and generated tables removed the same way.

The rules are the project's, from CLAUDE.md and .claude/styles/, plus 2 more: no
Oxford comma and American English spelling.

    python scripts/prose_audit.py                       # the library and cli
    python scripts/prose_audit.py attacks cli           # only these directories
    python scripts/prose_audit.py --show paper/sections # every hit with file:line
    python scripts/prose_audit.py --gate README.md      # exit 1 if anything hits

Each rule is a heuristic. The number-word rule over-reports idiomatic uses that
NUMBER_IDIOMS does not list, and the Oxford-comma rule cannot tell a list peer
from an appositive, so
it counts `, which we call PSBD-TM, and` alongside a real serial comma. The
listing exists for a person to judge each hit. --gate is for a commit hook, where
a nonzero exit is the point.
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
    r"three-|one\'s|someone|anyone|everyone|none|all-to-one|to-one\b|one-sided|"
    r"one another|"
    # "one" as a pronoun standing in for a noun already named, as in "the
    # BadNets one" or "a token-level one". A count always names its unit, so
    # "one" followed by punctuation or by a verb or preposition is a pronoun.
    r"(?:the|a|an|that|this|next|another)\s+(?:[\w-]+\s+){0,2}one\b|"
    r"one\s*[.,;:)\]]|"
    r"one\s+(?:would|will|is|was|were|wins?|means?|has|have|had|does|do|did|"
    r"can|cannot|could|should|may|might|sits?|reads?|gives?|shows?|in|of|to|for|and|"
    r"or|but|that|which)\b)",
    re.IGNORECASE,
)
# British spellings and their American forms. The project writes American English,
# and a rename that misses a spelling leaves 2 conventions in 1 file. The -ise family
# is matched on the stem so every inflection is caught by 1 entry.
BRITISH_STEMS = (
    "optimis",
    "normalis",
    "visualis",
    "standardis",
    "initialis",
    "regularis",
    "summaris",
    "organis",
    "recognis",
    "utilis",
    "penalis",
    "minimis",
    "maximis",
    "quantis",
    "serialis",
    "parameteris",
    "characteris",
    "marginalis",
    "discretis",
    "crystallis",
    "orthogonalis",
    "localis",
    "generalis",
    "randomis",
    "prioritis",
)
BRITISH_WORDS = (
    "behaviour",
    "colour",
    "favour",
    "honour",
    "labour",
    "centre",
    "fibre",
    "licence",
    "defence",
    "offence",
    "pretence",
    "grey",
    "artefact",
    "analyse",
    "analysed",
    "analysing",
    "catalogue",
    "modelling",
    "modelled",
    "labelling",
    "labelled",
    "signalling",
    "travelling",
    "cancelled",
    "fulfil",
    "enrol",
    "skilful",
)
# An -ise stem only counts with a real inflection behind it. Without the suffix
# group the optimis stem matches "optimistic", which is correct American English.
BRITISH_SPELLING = re.compile(
    r"\b(?:"
    + "|".join(
        rf"{stem}(?:e|es|ed|ing|ation|ations|er|ers|able)" for stem in BRITISH_STEMS
    )
    + r"|"
    + "|".join(BRITISH_WORDS)
    + r")\b"
)

# Words whose frequency in academic writing jumped with the arrival of language
# models (Kobak et al., https://arxiv.org/html/2406.07016v4). A hit is not proof of
# anything, it is a prompt to pick the specific word instead.
EXCESS_VOCABULARY = (
    "delve",
    "delves",
    "delving",
    "underscores",
    "underscored",
    "underscoring",
    "intricate",
    "pivotal",
    "crucial",
    "comprehensive",
    "meticulous",
    "meticulously",
    "showcase",
    "showcases",
    "showcasing",
    "realm",
    "tapestry",
    "paradigm shift",
    "moreover",
    "furthermore",
    "notably",
    "additionally",
)
# The signature of a serial comma is a list item closed by a comma and followed by
# the conjunction, as in `a, b, and c`. A comma before a conjunction that joins 2
# independent clauses is
# ordinary punctuation, so the middle item must itself be short and comma-free.
SERIAL_COMMA = re.compile(r",\s+[^,;:]{1,40},\s+(?:and|or)\s")

# Verbs that name no actor and no action (Black,
# https://perceiving-systems.blog/en/post/writing-a-good-scientific-paper).
VAGUE_VERBS = (
    "allows to",
    "allows us to",
    "provides a way",
    "enables the",
    "facilitates",
    "is responsible for",
)
AI_TELLS = (
    "note that",
    "it is worth noting",
    "in order to",
    "is responsible for",
    "leverage the",
    "leverage a",
    "leverages",
    "leveraging",
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


def python_prose(path: str) -> list[tuple[int, str, str]]:
    """Every comment and docstring in a Python file as (line, kind, text)."""
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


def markdown_prose(path: str) -> list[tuple[int, str, str]]:
    """Every prose line of a markdown file as (line, kind, text).

    A fenced block is code or output and carries none of the prose rules. A table
    row is usually generated, and its pipes and digits would swamp every count, so
    it is skipped too. An indented block counts as code only where it opens after
    a blank line, which is CommonMark's rule. Treating every 4-space line as code
    hid the wrapped continuation of a list item, and that is 230 lines of docs/.
    """
    found = []
    in_fence = False
    in_math = False
    after_blank = True
    with open(path) as handle:
        for number, line in enumerate(handle.read().split("\n"), start=1):
            stripped = line.strip()
            blank = not stripped
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence
                after_blank = False
                continue
            if in_fence:
                continue
            if stripped.startswith("$$"):
                in_math = not in_math
                after_blank = False
                continue
            if in_math:
                continue
            if line.startswith("    ") and after_blank:
                after_blank = blank
                continue
            if stripped.startswith("|") or set(stripped) <= set("|-: "):
                after_blank = blank
                continue
            if stripped:
                found.append((number, "markdown", line))
            after_blank = blank
    return found


TEX_COMMENT = re.compile(r"(?<!\\)%")
# Commands whose brace argument is prose a reader sees, so the argument stays and
# only the command name goes. A caption carries as much prose as a paragraph.
TEX_PROSE_COMMANDS = (
    "caption",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "title",
    "footnote",
    "emph",
    "textbf",
    "textit",
    "item",
)
# What a LaTeX line may hold that is notation or markup rather than prose. Order
# matters: the prose-carrying commands are unwrapped before the general command
# pattern would eat their arguments.
TEX_NOISE = (
    r"\\begin\{[^}]*\}(\[[^\]]*\])?(\{[^}]*\})*",
    r"\\end\{[^}]*\}",
    r"\$[^$]*\$",
    r"\\[A-Za-z@]+\*?(\[[^\]]*\])?(\{[^}]*\})*",
    r"\\[^A-Za-z]",
)
TEX_UNWRAP = re.compile(
    r"\\(?:" + "|".join(TEX_PROSE_COMMANDS) + r")\*?(?:\[[^\]]*\])?\{"
)
# Environments whose body is notation, a generated table or verbatim code.
TEX_SKIPPED_ENVIRONMENTS = (
    "equation",
    "align",
    "gather",
    "split",
    "tabular",
    "tabularx",
    "verbatim",
    "lstlisting",
    "tikzpicture",
)


def tex_prose(path: str) -> list[tuple[int, str, str]]:
    """Every prose line of a LaTeX file as (line, kind, text).

    A comment line is the author talking to themselves and still carries the rules.
    A math or tabular environment is notation, so its body is skipped the way a
    docstring's indented formula block is. A caption is prose, so the commands
    that wrap prose are unwrapped rather than removed with their argument.
    """
    found = []
    depth = 0
    opening = re.compile(r"\\begin\{(" + "|".join(TEX_SKIPPED_ENVIRONMENTS) + r")\*?\}")
    closing = re.compile(r"\\end\{(" + "|".join(TEX_SKIPPED_ENVIRONMENTS) + r")\*?\}")
    # A short environment that opens and closes on 1 line is cut out in place, so
    # the sentence either side of it survives. Only a multi-line one sets depth.
    inline = re.compile(
        r"\\begin\{(" + "|".join(TEX_SKIPPED_ENVIRONMENTS) + r")\*?\}"
        r".*?\\end\{\1\*?\}"
    )
    with open(path) as handle:
        for number, raw in enumerate(handle.read().split("\n"), start=1):
            line = inline.sub(" ", raw)
            if opening.search(line):
                depth += 1
            if depth:
                if closing.search(line):
                    depth -= 1
                continue
            # A LaTeX comment starts at an UNESCAPED %. \% is a percent sign, and
            # splitting on the literal character truncated every sentence that
            # quoted a rate. A whole-line comment is still prose and is kept.
            if line.lstrip().startswith("%"):
                body = line
            else:
                body = TEX_COMMENT.split(line, maxsplit=1)[0]
            body = TEX_UNWRAP.sub(" ", body)
            for pattern in TEX_NOISE:
                body = re.sub(pattern, " ", body)
            body = body.replace("{", " ").replace("}", " ")
            if body.strip():
                found.append((number, "latex", body))
    return found


# A sentence ends at a full stop, question mark or colon followed by a space and a
# capital or a macro. The negative lookbehind keeps "et al." attached to what
# follows it, so a citation is never split down the middle.
SENTENCE_END = re.compile(r"(?<!\bet al)(?<=[.?:])\s+(?=[A-Z\\])")


def as_sentences(
    fragments: list[tuple[int, str, str]],
) -> list[tuple[int, str, str]]:
    """The same fragments split at sentence boundaries, keeping each one's line."""
    split = []
    for line, kind, body in fragments:
        for sentence in SENTENCE_END.split(body):
            if sentence.strip():
                split.append((line, kind, sentence.strip()))
    return split


EXTRACTORS = {
    ".py": python_prose,
    ".md": lambda path: as_sentences(markdown_prose(path)),
    ".tex": lambda path: as_sentences(tex_prose(path)),
}


def prose_of(path: str) -> list[tuple[int, str, str]]:
    """Every prose fragment of a file, whichever language it is written in."""
    extractor = EXTRACTORS.get(os.path.splitext(path)[1])
    if extractor is None:
        return []
    return extractor(path)


MARKDOWN_MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$", re.DOTALL)


def strip_code_spans(text: str) -> str:
    """Prose with backticked spans and bracketed math removed, since those are code."""
    text = MARKDOWN_MATH.sub(" code ", text)
    text = re.sub(r"`[^`]*`", "code", text)
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
    if SERIAL_COMMA.search(without_code):
        hits.append(("oxford comma", line, text))
    for match in re.finditer(NUMBER_WORDS, plain, re.IGNORECASE):
        window = plain[max(0, match.start() - 12) : match.end() + 10]
        if not NUMBER_IDIOMS.search(window):
            hits.append(("number word", line, text))
            break
    if kind == "comment" and re.match(r"^[-=*#~_]{4,}", text.strip()):
        hits.append(("banner", line, text))
    lowered = plain.lower()
    for tell in AI_TELLS:
        if tell in lowered:
            hits.append(("ai tell", line, text))
            break
    if BRITISH_SPELLING.search(lowered):
        hits.append(("british spelling", line, text))
    for word in EXCESS_VOCABULARY:
        if re.search(rf"\b{word}\b", lowered):
            hits.append(("excess vocabulary", line, text))
            break
    for phrase in VAGUE_VERBS:
        if phrase in lowered:
            hits.append(("vague verb", line, text))
            break
    return hits


SKIPPED_DIRECTORIES = {"__pycache__", ".git", ".venv", "third_party", "node_modules"}


def audited_files(roots) -> list[str]:
    """Every file under the roots that an extractor knows how to read."""
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIPPED_DIRECTORIES]
            for name in sorted(filenames):
                if os.path.splitext(name)[1] in EXTRACTORS:
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
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit 1 when anything hits, for a commit hook",
    )
    parser.add_argument(
        "--rule",
        action="append",
        default=None,
        help="report only this rule, repeatable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots, show = args.roots, args.show
    wanted = set(args.rule) if args.rule else None

    totals: dict[str, int] = {}
    per_file: dict[str, dict[str, int]] = {}
    for path in audited_files(roots):
        for line, kind, text in prose_of(path):
            for rule, at, snippet in violations(line, kind, text):
                if wanted is not None and rule not in wanted:
                    continue
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
    failed = args.gate and bool(totals)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
