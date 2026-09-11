"""Loading the detection summary so the unsafe rows cannot be pooled by accident.

results/detection_summary.csv keeps rows that must not be averaged with the rest,
because dropping them at write time would hide that they exist. 2 columns mark
them. cache_backed is False when the stage-1 tensors a row was computed from are
no longer under results/, so the row cannot be recomputed or verified. The rows
like that on disk come from a superseded operator that scores higher than what
survives. variant is non-null for a measurement of a different question: a
different Monte Carlo budget (passes_20), a depth-band placement (blocks_5_8) or
a run with the model's own dropouts on (model_dropout_0_1).

load_detection_summary defaults to the plain, verifiable rows and says what it
dropped. Asking for the rest requires saying so.
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

    plain_only drops every row whose variant is set, since those measure a
    different question. require_cache_backed drops every row whose stage-1 tensors
    are gone, since a row that cannot be recomputed cannot be checked. verbose
    prints what was dropped, because a silent filter is how a coverage difference
    becomes a conclusion. Pass False in a loop.
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

    kept = frame.reset_index(drop=True)
    return kept


def summary_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """A row per (architecture, dataset) with its checkpoint and cell counts.

    A coverage difference between groups can invert a conclusion, so the shape of
    what is being averaged is worth seeing before the average.
    """
    coverage = (
        frame.groupby(["architecture", "dataset"])
        .agg(checkpoints=("folder", "nunique"), cells=("auroc", "size"))
        .reset_index()
    )
    return coverage
