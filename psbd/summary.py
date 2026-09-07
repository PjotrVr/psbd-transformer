"""Loading the detection summary so the unsafe rows cannot be used by accident.

`results/detection_summary.csv` carries rows that must not be pooled with the
rest, and carrying them is deliberate: dropping them at write time would hide
that they exist. The hazard is that nothing was dropping them at read time
either, so every `groupby("operator").auroc.mean()` over the file silently
included them.

Two columns mark them:

`cache_backed` is False when the stage-1 tensors the row was computed from are no
longer under `results/`. Those rows cannot be recomputed or verified. All 3029 of
them are the batch-coupled Gaussian that audit finding A2 archived, and they
score about 0.087 higher than the records that survive, so they inflate whatever
they touch.

`variant` is non-null for a measurement that is not the plain one: the superseded
Gaussian, a different Monte Carlo budget (`passes_20`), a depth-band placement
(`blocks_5_8`), or a run with the model's own dropouts activated
(`model_dropout_0_1`). Each is a legitimate measurement of a different question.

So `load_detection_summary` defaults to the plain, verifiable rows and says what
it dropped. Asking for the rest requires saying so.
"""

import pandas as pd

DEFAULT_SUMMARY_PATH = "results/detection_summary.csv"


def load_detection_summary(
    path: str = DEFAULT_SUMMARY_PATH,
    plain_only: bool = True,
    require_cache_backed: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """The detection summary, filtered to rows that can be pooled and verified.

    plain_only drops every row whose `variant` is set, which are measurements of
    a different question rather than of the same question done differently.

    require_cache_backed drops every row whose stage-1 tensors are gone. A row
    that cannot be recomputed cannot be checked, and the ones in this file that
    cannot be recomputed are known to be a superseded operator.

    verbose prints what was dropped, because a silent filter is how a coverage
    difference becomes a conclusion. Pass False in a loop.
    """
    frame = pd.read_csv(path)
    total = len(frame)

    dropped = {}
    if plain_only and "variant" in frame:
        variant_rows = frame["variant"].notna()
        dropped["variant"] = int(variant_rows.sum())
        frame = frame[~variant_rows]
    if require_cache_backed and "cache_backed" in frame:
        unbacked = ~frame["cache_backed"].astype(bool)
        dropped["not cache-backed"] = int(unbacked.sum())
        frame = frame[~unbacked]

    if verbose and dropped:
        parts = ", ".join(f"{count} {reason}" for reason, count in dropped.items())
        print(f"detection summary: {len(frame)} of {total} rows kept, dropped {parts}")

    return frame.reset_index(drop=True)


def summary_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (architecture, dataset) with the checkpoint and cell counts.

    Coverage differences between groups have inverted conclusions in this project
    before, so the shape of what is being averaged is worth looking at before the
    average is.
    """
    coverage = (
        frame.groupby(["architecture", "dataset"])
        .agg(checkpoints=("folder", "nunique"), cells=("auroc", "size"))
        .reset_index()
    )
    return coverage
