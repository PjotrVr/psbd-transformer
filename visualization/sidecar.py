"""The JSON sidecar every figure writes beside its PDF, so a plotted number traces back.

A figure without its numbers cannot be checked, regenerated or quoted, so the
writer records the tool, the checkpoint, the git commit, the layer, view and
sample count the figure was made from, and every array it plotted.
"""

import json
import os

from utils.provenance import current_git_commit, utc_timestamp


def sidecar_path(figure_path: str) -> str:
    """The .json path next to a figure's .pdf or .png."""
    stem, _ = os.path.splitext(figure_path)
    path = f"{stem}.json"
    return path


def write_sidecar(
    figure_path: str, tool: str, folder: str, settings: dict, plotted: dict
) -> None:
    """Write the numbers a figure plotted, with provenance, next to the figure.

    settings holds what the tool was run with (layer, view, samples, seed and
    any tool-specific knob), plotted holds every array in the figure as lists.
    """
    payload = {
        "tool": tool,
        "folder": folder,
        "figure": os.path.basename(figure_path),
        "git_commit": current_git_commit(),
        "written_at": utc_timestamp(),
        "settings": settings,
        "plotted": plotted,
    }
    with open(sidecar_path(figure_path), "w") as handle:
        json.dump(payload, handle, indent=2)
