"""Every module in every package still imports.

A reorganization's real failure mode is a broken import nobody happens to exercise
in another test, so this is deliberately exhaustive rather than relying on the rest
of the suite to catch it incidentally. One test case per module, parametrized, so a
break names exactly which one broke.

The list is walked from the filesystem rather than written out. A hand-maintained
list is the thing that goes stale: the previous one still named 45 modules that had
been deleted, which turned a reorganization into 45 confusing failures instead of
1 clear one. EXPECTED_MINIMUM guards the other direction, since a walk that finds
nothing would pass silently.

Run with pytest from the repo root: pytest tests/test_imports.py.
"""

import importlib
import os

import pytest

PACKAGES = (
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
)

# A floor, not a count, so adding a module needs no edit here.
EXPECTED_MINIMUM = 60

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def discover_modules() -> list[str]:
    """Every importable module under the packages, as dotted names."""
    found = []
    for package in PACKAGES:
        directory = os.path.join(REPO_ROOT, package)
        found.append(package)
        for name in sorted(os.listdir(directory)):
            if name.endswith(".py") and name != "__init__.py":
                found.append(f"{package}.{name[:-3]}")
    return found


MODULES = discover_modules()


def test_the_walk_found_the_tree():
    """A walk that found nothing would make every case below vacuous."""
    assert len(MODULES) >= EXPECTED_MINIMUM, (
        f"found only {len(MODULES)} modules, so the walk is not seeing the tree"
    )


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
