"""Division that fails loudly instead of quietly.

A ratio whose denominator approaches 0 is undefined. Clamping the denominator to
a small positive floor turns that undefined quantity into a large finite number,
which flows into a mean, survives a plot and reads as a result. Below the floor
the answer here is NaN instead. NaN propagates, prints as nan, cannot be
mistaken for a measurement and forces the caller to say what it wants done with
the degenerate case.

2 floors for 2 distinct failures:

    safe_ratio           for a denominator that may be signed. Guards on magnitude,
                         so a legitimately negative denominator keeps its sign and
                         only a near-zero one is rejected. Clamping a signed
                         denominator to a positive floor would also flip the sign
                         of the result.
    safe_ratio_positive  for a denominator that is a norm, a count or a probability
                         and cannot be negative. A negative value there is a bug in
                         the caller and is rejected rather than used.

Not every clamped denominator is this failure. Normalising a vector by its own
sum to form a distribution, or flooring a probability before a log for an
entropy, divides by a quantity that is non-negative by construction, and
clamping there correctly yields 0 for an all-zero row. The failure is specific
to a ratio of 2 measurements, where a near-zero denominator means the comparison
is undefined and must say so.
"""

import math

import torch

# Activation norms in this project are order 1 to 100 and attention masses order
# 1e-3 to 1, so 1e-9 sits far below any real measurement and far above float32
# noise.
DEFAULT_FLOOR = 1e-9


def safe_ratio(numerator, denominator, floor: float = DEFAULT_FLOOR):
    """numerator / denominator, or NaN where |denominator| < floor.

    Accepts floats and tensors. A tensor result is NaN only in the degenerate
    positions, so 1 bad sample does not destroy a batch.
    """
    if isinstance(denominator, torch.Tensor) or isinstance(numerator, torch.Tensor):
        numerator = torch.as_tensor(numerator)
        denominator = torch.as_tensor(denominator)
        safe = denominator.abs() >= floor
        out = torch.full_like(
            torch.broadcast_tensors(numerator, denominator)[0].float(), float("nan")
        )
        out[safe] = (numerator / denominator.masked_fill(~safe, 1.0)).float()[safe]
        return out
    if abs(denominator) < floor:
        return float("nan")
    return numerator / denominator


def safe_ratio_positive(numerator, denominator, floor: float = DEFAULT_FLOOR):
    """numerator / denominator for a denominator that must be non-negative.

    A negative denominator here means the caller computed the wrong thing, since a
    norm cannot be negative, so the result is NaN rather than a plausible negative
    ratio.
    """
    if isinstance(denominator, torch.Tensor) or isinstance(numerator, torch.Tensor):
        numerator = torch.as_tensor(numerator)
        denominator = torch.as_tensor(denominator)
        safe = denominator >= floor
        out = torch.full_like(
            torch.broadcast_tensors(numerator, denominator)[0].float(), float("nan")
        )
        out[safe] = (numerator / denominator.masked_fill(~safe, 1.0)).float()[safe]
        return out
    if not denominator >= floor:
        return float("nan")
    return numerator / denominator


def is_defined(value) -> bool:
    """Whether a scalar result is a real measurement rather than a degenerate value."""
    return isinstance(value, (int, float)) and math.isfinite(value)
