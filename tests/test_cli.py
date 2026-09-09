"""The cli/ package: it imports, it keeps its flags, it runs, and it stays thin.

Four properties, each of which has a specific way of going wrong during a port.

  1. Every module imports. A dependency that moved packages breaks at import
     time, and an entrypoint nobody exercised until a job started is the worst
     place to find that out.
  2. Flag parity. Hundreds of generated PBS scripts and every doc example spell
     these flags by name, so a rename is a silently queued broken job. Each new
     parser's option strings are compared against its original's programmatically,
     and the new set must be a superset: additions are allowed, removals are not.
  3. Real runs. Four entrypoints are executed as subprocesses against the real
     results/ and checkpoints/ trees, at the smallest scope on disk, and their
     output is asserted. Writes are redirected into tmp_path, and the stage-1
     cache is symlinked in read-only, so no test can modify a results folder.
  4. Import discipline. cli/ may import from psbd and from cli. Importing a
     leftover root module or one of the pre-rewrite packages would keep the old
     tree alive, so the check is on the AST rather than on a successful import,
     which would pass for as long as those files happen to still exist.

The real-run tests skip when no complete stage-1 cache is on disk, so the suite
still passes on a fresh checkout.
"""

import argparse
import ast
import csv
import importlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI_DIR = os.path.join(REPO_ROOT, "cli")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")

CLI_MODULES = (
    "cli.analyze",
    "cli.backfill",
    "cli.baselines",
    "cli.compare_detectors",
    "cli.evaluate",
    "cli.fuse_detectors",
    "cli.head_profile",
    "cli.operating_points",
    "cli.report",
    "cli.summary",
    "cli.sweep",
    "cli.tables",
    "cli.train_backdoor",
    "cli.train_benign",
    "cli.variants",
)

# Every entrypoint and the module it was ported from. The old module stays the
# source of truth for the flag set until it is deleted.
PORTED_FROM = {
    "cli.analyze": "psbd_analyze",
    "cli.backfill": "scripts.backfill_metadata",
    "cli.baselines": "baseline_detect",
    "cli.compare_detectors": "detector_comparison",
    "cli.evaluate": "metrics",
    "cli.fuse_detectors": "detector_fusion",
    "cli.head_profile": "psbd_head_profile",
    "cli.operating_points": "psbd_operating_points",
    "cli.report": "psbd_report",
    "cli.summary": "scripts.detection_summary",
    "cli.sweep": "psbd_dropout_sweep",
    "cli.tables": "defence_tables",
    "cli.train_backdoor": "train_backdoor",
    "cli.train_benign": "train_benign",
    "cli.variants": "psbd_variants",
}

# Top-level names cli/ must never import. The pre-rewrite packages, plus every
# root-level module whose contents now live in psbd/. Spelled out rather than
# derived from the filesystem so the check keeps its meaning once those files are
# deleted, which is the entire point of the rewrite.
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "analysis",
        "attacks",
        "defences",
        "scripts",
        "utils",
        "adaptive_evasion",
        "backdoor_data",
        "baseline_detect",
        "defence_tables",
        "detector_comparison",
        "detector_fusion",
        "evaluate",
        "loaders",
        "metrics",
        "models",
        "pbs_grid",
        "poison",
        "psbd_analyze",
        "psbd_dropout_sweep",
        "psbd_head_profile",
        "psbd_operating_points",
        "psbd_report",
        "psbd_variants",
        "sam",
        "stealth",
        "train",
        "train_backdoor",
        "train_benign",
    }
)

STAGE_ONE_BASELINES = (
    "baseline_validation.pt",
    "baseline_clean.pt",
    "baseline_backdoor.pt",
)


class _CapturedParser(Exception):
    """Carries the parser out of parse_args before it can read sys.argv."""

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser


def build_parser(module_name: str) -> argparse.ArgumentParser:
    """The ArgumentParser a module's parse_args builds, without parsing anything.

    parse_args is intercepted rather than called with a crafted argv, because
    several of these parsers run post-validation on the parsed namespace and a
    required argument would abort before the parser is visible. Raising from the
    interception point returns the parser and skips both.
    """
    module = importlib.import_module(module_name)
    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        raise _CapturedParser(self)

    argparse.ArgumentParser.parse_args = capture
    try:
        module.parse_args()
    except _CapturedParser as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_args = original

    raise AssertionError(
        f"{module_name}.parse_args never called ArgumentParser.parse_args"
    )


def option_strings(parser: argparse.ArgumentParser) -> set[str]:
    """Every flag spelling the parser accepts, including -h."""
    flags: set[str] = set()
    for action in parser._actions:
        flags.update(action.option_strings)
    return flags


def imported_roots(path: str) -> set[str]:
    """The top-level package of every absolute import in one source file.

    Relative imports are excluded because they can only reach inside cli/ itself.
    """
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative, so it resolves inside cli/
                continue
            if node.module:
                roots.add(node.module.split(".")[0])

    return roots


def cli_source_files() -> list[str]:
    """Every .py file in cli/, including __init__.py."""
    return sorted(
        os.path.join(CLI_DIR, name)
        for name in os.listdir(CLI_DIR)
        if name.endswith(".py")
    )


def smallest_cached_checkpoint() -> str | None:
    """The folder with a complete stage-1 cache and the fewest placements.

    Fewest placements because stage 2 is linear in them and this runs inside the
    ordinary test suite. None when nothing on disk is complete enough to analyze.
    """
    if not os.path.isdir(RESULTS_DIR):
        return None

    candidates = []
    for folder in sorted(os.listdir(RESULTS_DIR)):
        psbd_dir = os.path.join(RESULTS_DIR, folder, "psbd")
        if not os.path.isdir(psbd_dir):
            continue
        if not os.path.exists(os.path.join(CHECKPOINTS_DIR, folder, "args.json")):
            continue
        if not os.path.exists(os.path.join(psbd_dir, "split_manifest.json")):
            continue
        if not all(
            os.path.exists(os.path.join(psbd_dir, name)) for name in STAGE_ONE_BASELINES
        ):
            continue
        placements = [
            name
            for name in os.listdir(psbd_dir)
            if os.path.isdir(os.path.join(psbd_dir, name))
        ]
        if placements:
            candidates.append((len(placements), folder))

    if not candidates:
        return None
    return min(candidates)[1]


def run_cli(module_name: str, *arguments: str) -> str:
    """Run one entrypoint as a subprocess from the repository root, returning stdout.

    A subprocess rather than a direct main() call, because these are entrypoints
    and "python -m cli.x" is the thing that has to work, argv parsing and all.
    """
    completed = subprocess.run(
        [sys.executable, "-m", module_name, *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"{module_name} exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed.stdout


@pytest.fixture(scope="session")
def cached_checkpoint() -> str:
    """A real checkpoint folder whose stage-1 cache is complete."""
    folder = smallest_cached_checkpoint()
    if folder is None:
        pytest.skip("no complete stage-1 PSBD cache under results/")
    return folder


@pytest.fixture(scope="session")
def analyzed_results(tmp_path_factory, cached_checkpoint: str) -> tuple[str, str]:
    """A scratch results tree holding one real checkpoint's stage-2 output.

    The stage-1 cache is symlinked in read-only and every write lands in the
    scratch tree, so running the suite can never touch results/.
    """
    results_dir = tmp_path_factory.mktemp("results")
    folder_dir = results_dir / cached_checkpoint
    folder_dir.mkdir()
    (folder_dir / "psbd").symlink_to(
        os.path.join(RESULTS_DIR, cached_checkpoint, "psbd")
    )

    run_cli(
        "cli.analyze",
        "--checkpoint-folder",
        cached_checkpoint,
        "--results-dir",
        str(results_dir),
    )
    return str(results_dir), cached_checkpoint


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_module_imports(module_name: str) -> None:
    """Every entrypoint imports, so a moved dependency fails here and not in a job."""
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_module_has_docstring(module_name: str) -> None:
    """Several of these docstrings explain what the output means and are load bearing."""
    module = importlib.import_module(module_name)
    assert module.__doc__ and module.__doc__.strip()


@pytest.mark.parametrize(("new_module", "old_module"), sorted(PORTED_FROM.items()))
def test_flag_parity_with_original(new_module: str, old_module: str) -> None:
    """No flag spelling was dropped or renamed by the port.

    A superset check rather than equality: adding a flag is a normal change, and
    removing one silently breaks every queued PBS script that spells it.

    Skips once the original is deleted, which is the intended end state of the
    rewrite. Until then this is the only mechanical guard on the flag set.
    """
    if importlib.util.find_spec(old_module) is None:
        pytest.skip(f"{old_module} has been removed, so there is nothing to compare")

    new_flags = option_strings(build_parser(new_module))
    old_flags = option_strings(build_parser(old_module))

    removed = old_flags - new_flags
    assert not removed, f"{new_module} dropped {sorted(removed)} from {old_module}"


@pytest.mark.parametrize("path", cli_source_files(), ids=os.path.basename)
def test_imports_only_psbd_and_cli(path: str) -> None:
    """cli/ never reaches back into a root module or a pre-rewrite package."""
    forbidden = imported_roots(path) & FORBIDDEN_IMPORT_ROOTS
    assert not forbidden, f"{os.path.basename(path)} imports {sorted(forbidden)}"


def test_analyze_writes_psbd_metrics(analyzed_results: tuple[str, str]) -> None:
    """Stage 2 turns a real cache into a psbd_metrics.json with real detection numbers."""
    results_dir, folder = analyzed_results
    with open(os.path.join(results_dir, folder, "psbd_metrics.json")) as handle:
        report = json.load(handle)

    assert report["folder_name"] == folder
    assert report["placements"], "at least one placement must be scored"
    assert report["split_sizes"]["backdoor"] > 0

    for placement, block in report["placements"].items():
        assert block["rates"], f"{placement} produced no complete rate"
        for row in block["rates"]:
            auroc = row["detection"]["q0.25"]["auroc"]
            assert 0.0 <= auroc <= 1.0


def test_report_renders_a_table(analyzed_results: tuple[str, str]) -> None:
    """The aggregator renders one markdown row per (checkpoint, placement)."""
    results_dir, folder = analyzed_results
    output = run_cli("cli.report", "--results-dir", results_dir, "--format", "markdown")

    assert "| checkpoint | attack |" in output
    assert folder in output
    assert "Mean AUROC by placement" in output


def test_report_renders_csv(analyzed_results: tuple[str, str]) -> None:
    """The same rows as CSV, so a downstream join gets the values and not the layout."""
    results_dir, _folder = analyzed_results
    output = run_cli("cli.report", "--results-dir", results_dir, "--format", "csv")

    rows = list(csv.DictReader(output.splitlines()))
    assert rows
    assert "adaptive_auroc" in rows[0]


def test_summary_writes_a_versionable_csv(tmp_path) -> None:
    """The compact summary collapses the whole tree into one CSV with SAM excluded."""
    if not os.path.isdir(RESULTS_DIR):
        pytest.skip("no results/ tree")

    output_path = tmp_path / "detection_summary.csv"
    stdout = run_cli("cli.summary", "--output", str(output_path))

    assert "wrote" in stdout
    with open(output_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        pytest.skip("no psbd_metrics.json on disk to summarize")

    # A 4-column subset was the whole assertion here, which is how cli/summary.py
    # drifted into a fork missing every fix audit findings A4 and A19 were about
    # while still passing. The columns that mark a row unusable are the ones worth
    # requiring, since their absence is what makes a bad row invisible.
    required = {
        "folder",
        "placement",
        "position",
        "operator",
        "rule",
        "auroc",
        "variant",
        "cache_backed",
    }
    missing = required - set(rows[0])
    assert not missing, f"the summary lost columns that mark unusable rows: {missing}"

    # --include-sam defaults off, so a SAM checkpoint must not appear in the table.
    assert all(row["optimizer"] != "sam" for row in rows)

    # A4 was an unrecognized suffix silently attributed to the paper's own
    # baseline. Nothing may parse as an operator the registry does not have.
    # KNOWN_OPERATORS rather than psbd.operators.PERTURBATIONS, because scale_up
    # is a legitimate operator that the perturbation registry deliberately omits:
    # it needs the dataset's normalization constants and cannot be built from a
    # rate alone. The summary's own vocabulary is the right authority here.
    from cli.summary import KNOWN_OPERATORS

    known = set(KNOWN_OPERATORS) | {"dropout", "unknown"}
    seen = {row["operator"] for row in rows}
    assert seen <= known, f"summary invented operators: {sorted(seen - known)}"
    assert "unknown" not in seen, (
        "a placement parsed as 'unknown', so the parser does not cover the tree"
    )


def test_tables_coverage_only_reports_the_bar() -> None:
    """--coverage-only reports how much of the bar is met and emits no table."""
    if not os.path.isdir(CHECKPOINTS_DIR):
        pytest.skip("no checkpoints/ tree")

    output = run_cli("cli.tables", "--coverage-only")

    assert "coverage:" in output
    assert "per dataset:" in output
    assert "| poison |" not in output, "--coverage-only must not print a table"


@pytest.mark.parametrize("module_name", ["train_backdoor", "cli.train_backdoor"])
def test_repeated_attack_override_accumulates(module_name: str) -> None:
    """A repeated --attack-override must keep every key, not just the last one.

    Declared `nargs="*"` this silently kept only the final occurrence, so
    `--attack-override adversarial_dir=... --attack-override adversarial_epsilon=...`
    dropped the directory and turned Label-Consistent back into its patch-only
    variant while args.json advertised the adversarial one. Nothing was out of
    range, so nothing complained.
    """
    module = importlib.import_module(module_name)
    argv = [
        "--dataset",
        "cifar100",
        "--attack",
        "lc",
        "--poison-rate",
        "0.01",
        "--output",
        "unused",
        "--attack-override",
        "adversarial_dir=bases/",
        "--attack-override",
        "adversarial_epsilon=0.0627",
    ]
    old_argv = sys.argv
    try:
        sys.argv = [module_name, *argv]
        args = module.parse_args()
    finally:
        sys.argv = old_argv
    assert module.parse_attack_overrides(args.attack_override) == {
        "adversarial_dir": "bases/",
        "adversarial_epsilon": "0.0627",
    }
