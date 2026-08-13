"""Reading and writing sweep artifacts: PSU score arrays and metric rows.

All disk access for results lives here. The on-disk layout matches what the
analysis module reads: one metrics.json list plus one compressed npz of scores
per dropout rate, under experiments/<placement>/<attack_folder>/.
"""

import json
import os

import numpy as np
import torch


def _rate_tag(rate: float) -> str:
    return f"scores_{rate}".replace(".", "_") + ".npz"


def scores_path(experiment_dir: str, rate: float) -> str:
    return os.path.join(experiment_dir, _rate_tag(rate))


def save_scores(
    experiment_dir: str,
    rate: float,
    validation_scores: torch.Tensor,
    clean_scores: torch.Tensor,
    backdoor_scores: torch.Tensor,
) -> None:
    np.savez_compressed(
        scores_path(experiment_dir, rate),
        val=validation_scores.float().numpy(),
        clean=clean_scores.float().numpy(),
        backdoor=backdoor_scores.float().numpy(),
    )


def load_scores(
    experiment_dir: str, rate: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = np.load(scores_path(experiment_dir, rate))
    return (
        torch.tensor(arrays["val"]),
        torch.tensor(arrays["clean"]),
        torch.tensor(arrays["backdoor"]),
    )


def save_metrics(experiment_dir: str, rows: list[dict]) -> None:
    """Write the full metrics list once, rather than rewriting per row.

    The notebook reloaded and rewrote the whole file for every appended row,
    which is quadratic in the number of rows. Accumulating in memory and
    writing once removes that cost while keeping the same file format.
    """
    with open(os.path.join(experiment_dir, "metrics.json"), "w") as handle:
        json.dump(rows, handle, indent=2)


def load_metrics(experiment_dir: str) -> list[dict]:
    with open(os.path.join(experiment_dir, "metrics.json")) as handle:
        return json.load(handle)
