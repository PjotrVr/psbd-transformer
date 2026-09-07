"""Rewrite pre-rewrite imports to their psbd equivalents.

Most of the mapping is 1 old module to 1 new module and a plain textual swap is
enough. Two are not, and they are the reason this is a script rather than a sed
command:

  defences.psbd_metrics  split into psbd.scores (what a sample's number is) and
                         psbd.decision (how a number becomes a verdict)
  defences.dropout       split into psbd.positions (where a probe attaches) and
                         psbd.operators (what it does)

For those, the destination depends on which SYMBOL is being imported, so the
script resolves each name against the real modules rather than guessing. A name
that resolves to neither half is reported instead of being rewritten, because a
silently dropped import is worse than a failed rewrite.
"""

import argparse
import ast
import importlib
import os
import re

DIRECT = {
    "poison": "psbd.poisoning",
    "models": "psbd.models",
    "train": "psbd.training",
    "evaluate": "psbd.evaluation",
    "loaders": "psbd.eval_loaders",
    "stealth": "psbd.stealth",
    "sam": "psbd.sam",
    "backdoor_data": "psbd.backdoorbench",
    "adaptive_evasion": "psbd.evasion",
    "attacks": "psbd.attacks",
    "utils.config": "psbd.config",
    "utils.datasets": "psbd.data",
    "analysis.cka": "psbd.analysis.cka",
    "analysis.direction": "psbd.analysis.direction",
    "analysis.features": "psbd.analysis.features",
    "analysis.lipschitz": "psbd.analysis.lipschitz",
    "analysis.embedding": "psbd.analysis.embedding",
    "analysis.analyze_latent": "psbd.analysis.latent",
    "defences.psbd_cache": "psbd.cache",
    "defences.checkpoint_eval": "psbd.splits",
    "defences.inference": "psbd.inference",
    "defences.perturbations": "psbd.operators",
    "defences.baselines": "psbd.baselines",
    "defences.detection": "psbd.evaluation",
}

# Symbols the rewrite renamed. Two privates became public because they are now
# imported across a module boundary, which a leading underscore says they should
# not be. Resolving these here keeps the rename out of every call site.
SYMBOL_RENAMES = {
    "_resolve_targets": "resolve_targets",
    "_ForwardRestore": "ForwardRestore",
    "vit_core": "network_core",
}


# Old module -> the new modules its symbols were split across, in lookup order.
SPLIT = {
    "defences.psbd_metrics": ("psbd.scores", "psbd.decision"),
    "defences.dropout": ("psbd.positions", "psbd.operators"),
}


def resolve_split(old_module: str, name: str) -> tuple[str, str] | None:
    """Which half of a split module owns this symbol, and its current name."""
    current = SYMBOL_RENAMES.get(name, name)
    for candidate in SPLIT[old_module]:
        if hasattr(importlib.import_module(candidate), current):
            return candidate, current
    return None


def rewrite_source(source: str, path: str, unresolved: list) -> str:
    """Rewrite every old import in one file, leaving everything else alone."""
    tree = ast.parse(source)
    replacements = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            old = node.module
            if old in DIRECT:
                replacements.append((node.lineno, old, DIRECT[old]))
            elif old in SPLIT:
                # One import statement can pull symbols that now live apart, so
                # each name is resolved and the statement may become 2 lines.
                by_target: dict[str, list[str]] = {}
                for alias in node.names:
                    resolved = resolve_split(old, alias.name)
                    if resolved is None:
                        unresolved.append((path, node.lineno, old, alias.name))
                        continue
                    target, current = resolved
                    # A renamed symbol keeps working at the call site by aliasing
                    # it back to the name the file already uses.
                    if current != alias.name and not alias.asname:
                        label = f"{current} as {alias.name}"
                    else:
                        label = current + (f" as {alias.asname}" if alias.asname else "")
                    by_target.setdefault(target, []).append(label)
                if by_target:
                    replacements.append((node.lineno, old, by_target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in DIRECT:
                    replacements.append((node.lineno, alias.name, DIRECT[alias.name]))

    if not replacements:
        return source

    lines = source.splitlines(keepends=True)
    for lineno, old, target in replacements:
        index = lineno - 1
        if isinstance(target, str):
            lines[index] = re.sub(rf"\b{re.escape(old)}\b", target, lines[index])
        else:
            indent = re.match(r"\s*", lines[index]).group()
            rendered = [
                f"{indent}from {module} import {', '.join(sorted(names))}\n"
                for module, names in sorted(target.items())
            ]
            lines[index] = "".join(rendered)
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="directories to rewrite in place")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unresolved: list = []
    changed = 0

    for root in args.roots:
        for directory, _, files in os.walk(root):
            if "__pycache__" in directory:
                continue
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(directory, name)
                source = open(path).read()
                rewritten = rewrite_source(source, path, unresolved)
                if rewritten == source:
                    continue
                changed += 1
                if args.dry_run:
                    print(f"  would rewrite {path}")
                else:
                    open(path, "w").write(rewritten)

    print(f"\nfiles {'to rewrite' if args.dry_run else 'rewritten'}: {changed}")
    if unresolved:
        print(f"\nUNRESOLVED, left untouched and needing a human ({len(unresolved)}):")
        for path, lineno, module, symbol in unresolved:
            print(f"  {path}:{lineno}  {module}.{symbol}")


if __name__ == "__main__":
    main()
