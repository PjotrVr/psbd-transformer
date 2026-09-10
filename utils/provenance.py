"""Git and clock stamps every artefact writer records so a number traces to a commit."""

import subprocess
from datetime import datetime, timezone


def _git(arguments: list[str]) -> str:
    """Stdout of 1 git command in the current checkout, stripped."""
    result = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    )
    output = result.stdout.strip()
    return output


def current_git_commit() -> str | None:
    """The commit the caller ran from, with -dirty appended when the tree had edits.

    A bare sha claims that checking it out reproduces the run, which is false when
    uncommitted changes were present, so a dirty tree records <sha>-dirty. Dirty
    means a tracked file is modified or an untracked Python file exists, since
    either can change what ran. Untracked data, tables and caches do not count,
    otherwise every generated artefact would mark the tree dirty. None outside a
    git checkout, so provenance never blocks a run.
    """
    try:
        commit = _git(["rev-parse", "HEAD"])
        modified = _git(["status", "--porcelain", "--untracked-files=no"])
        untracked_code = _git(
            ["ls-files", "--others", "--exclude-standard", "--", "*.py"]
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    dirty = bool(modified) or bool(untracked_code)
    label = f"{commit}-dirty" if dirty else commit
    return label


def utc_timestamp() -> str:
    """Wall-clock time in UTC ISO 8601, for start and end stamps."""
    stamp = datetime.now(timezone.utc).isoformat()
    return stamp
