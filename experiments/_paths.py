"""Where a cross-checkpoint experiment writes and reads its outputs.

A number in docs/ must trace to a commit and to a path. Before this helper 26
root-level results/*.json files and 17 CSV files scattered under experiments/
carried the cross-checkpoint records, with no rule saying which experiment had
written which file, so a reader who found a number in a table could not tell
which script to rerun. The convention paper/README.md states is that
per-checkpoint artefacts stay at results/<folder>/... and every cross-checkpoint
artefact lives under results/_experiments/<slug>/, where <slug> is the
experiments/<slug>/ directory that produced it. Every experiment builds its
output paths through these 2 functions so the slug in the path and the directory
holding the writer can never drift apart.
"""

import os


def experiment_results_dir(slug: str, results_dir: str = "results") -> str:
    """Return results/_experiments/<slug> and create it if it is missing.

    `slug` is the name of the experiments/<slug>/ directory that owns the
    artefacts. `results_dir` is the repo-level results root, overridable so a
    script run against another checkout or a scratch tree keeps the same layout.
    """
    directory = os.path.join(results_dir, "_experiments", slug)
    os.makedirs(directory, exist_ok=True)
    return directory


def experiment_result_path(slug: str, filename: str, results_dir: str = "results") -> str:
    """Return results/_experiments/<slug>/<filename>, creating the directory."""
    directory = experiment_results_dir(slug, results_dir)
    path = os.path.join(directory, filename)
    return path
