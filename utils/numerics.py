"""Division that fails loudly instead of quietly.

A ratio whose denominator approaches zero is not a large number, it is an undefined one, and
the difference matters. Clamping the denominator to a small positive floor turns an undefined
quantity into a large FINITE one that flows into a mean, survives a plot, and reads as a
result. Two such numbers were produced in this project before this module existed: a patching
"recovery" of -20,868,252 from clamping a signed denominator to +1e-6, and a sink takeover
ratio of 176 from a denominator that was near zero because the quantity genuinely was.

The rule here is the opposite: below the floor the answer is NaN. NaN propagates, prints as
nan, cannot be mistaken for a measurement, and forces the caller to say what it wants done
with the degenerate case.

Two distinct floors, because two distinct failures:

    safe_ratio          for denominators that may be SIGNED. Guards on magnitude, so a
                        legitimately negative denominator keeps its sign and only a near-zero
                        one is rejected. Clamping a signed denominator to a positive floor
                        also flips the sign of the result, which is the worse half of that bug.
    safe_ratio_positive for denominators that are norms, counts or probabilities and cannot be
                        negative. A negative value there is a bug in the caller, so it is
                        rejected rather than silently used.
"""

import math

import torch

# Relative to the scales in this project: activation norms are order 1 to 100, attention
# masses are order 1e-3 to 1, so 1e-9 is far below any real measurement and far above float32
# noise.
DEFAULT_FLOOR = 1e-9


def safe_ratio(numerator, denominator, floor: float = DEFAULT_FLOOR):
    """numerator / denominator, or NaN where |denominator| < floor.

    Works for floats and for tensors; a tensor result is NaN only in the degenerate positions,
    so one bad sample does not destroy a batch.
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

    A negative denominator here means the caller computed the wrong thing (a norm cannot be
    negative), so it returns NaN rather than a plausible-looking negative ratio.
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


# NOT every clamped denominator is this bug. Normalising a vector by its own sum to form a
# distribution, or flooring a probability before a log for an entropy, divides by a quantity
# that is non-negative by construction, and clamping there correctly yields 0 for an all-zero
# row rather than NaN. The bug is specific to a RATIO OF TWO MEASUREMENTS, where a near-zero
# denominator means the comparison is undefined and must say so.


def is_defined(value) -> bool:
    """Whether a scalar result is a real measurement rather than a degenerate one."""
    return isinstance(value, (int, float)) and math.isfinite(value)
