"""Prove that a set of edits changed prose and nothing else.

A docstring or comment rewrite must leave behaviour untouched, and the test suite
cannot promise that on its own, since it does not cover every line. This compares
the abstract syntax tree of each changed Python file against a git revision with
every docstring removed from both sides. Comments never reach the tree at all. If
the two trees are identical the edit was prose only, whatever the diff looks like.

    python scripts/check_prose_only.py            # working tree against HEAD
    python scripts/check_prose_only.py HEAD~3     # working tree against 3 commits back
    python scripts/check_prose_only.py A B        # revision B against revision A

Exit status is 0 only when every changed file is prose only. A new or deleted file
cannot be, and is reported as a code change.
"""

import argparse
import ast
import subprocess
import sys


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    ).stdout


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """The same tree with every module, class and function docstring removed."""
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            if isinstance(body[0].value.value, str):
                # A body of only a docstring needs a placeholder to stay valid.
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_signature(source: str) -> str:
    return ast.dump(strip_docstrings(ast.parse(source)), include_attributes=False)


def source_at(revision: str | None, path: str) -> str | None:
    """File contents at a revision, or from the working tree when revision is None."""
    try:
        if revision is None:
            with open(path) as handle:
                return handle.read()
        return git("show", f"{revision}:{path}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def changed_python_files(base: str, target: str | None) -> list[str]:
    arguments = (
        ["diff", "--name-only", base] + ([target] if target else []) + ["--", "*.py"]
    )
    return [line for line in git(*arguments).splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "base", nargs="?", default="HEAD", help="revision to compare against"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="revision to compare, default the working tree",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base, target = args.base, args.target

    code_changes = []
    for path in changed_python_files(base, target):
        before = source_at(base, path)
        after = source_at(target, path)
        if before is None or after is None:
            code_changes.append((path, "added or deleted"))
            continue
        try:
            same = code_signature(before) == code_signature(after)
        except SyntaxError as error:
            code_changes.append((path, f"does not parse: {error}"))
            continue
        if not same:
            code_changes.append((path, "code differs"))
        print(f"{'code changed' if not same else 'prose only  '}  {path}")

    if code_changes:
        print(f"\n{len(code_changes)} file(s) changed code, not only prose:")
        for path, reason in code_changes:
            print(f"  {path}: {reason}")
        sys.exit(1)
    print("\nevery changed file is prose only")


if __name__ == "__main__":
    main()
