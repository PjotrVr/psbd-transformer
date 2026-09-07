"""The trace estimator reading of isotropic perturbation, checked directly.

docs/theory-perturbation-consistency.md derives

    psu(x) is approximately -0.5 * trace(H * Sigma)

and observes that for an isotropic Sigma this is Hutchinson's trace estimator
applied to the Hessian of the predicted class probability. RademacherNoise exists
because that reading predicts a specific, checkable improvement over
GaussianNoise.

These tests check the mathematical claim on a known matrix, with no model
involved, so a failure here means the theory is wrong rather than that the
network did something unexpected. The claim about detection AUROC is a separate,
empirical question measured in experiments/.
"""

import pytest
import torch

from psbd.operators import (
    DETERMINISTIC_PERTURBATIONS,
    PERTURBATIONS,
    build_perturbation,
)

PROBE_DIMENSION = 64
PROBE_REPEATS = 4000


def symmetric_matrix(seed: int, dimension: int = PROBE_DIMENSION) -> torch.Tensor:
    """A symmetric matrix with a non-trivial diagonal, standing in for a Hessian."""
    generator = torch.Generator().manual_seed(seed)
    root = torch.randn(dimension, dimension, generator=generator)

    return (root + root.T) / 2


def hutchinson_estimates(matrix: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
    """One estimate of trace(matrix) per probe row, as z^T A z."""
    estimates = ((probes @ matrix) * probes).sum(dim=1)
    return estimates


def test_both_probe_distributions_estimate_the_trace_without_bias():
    """Hutchinson's identity: the expectation is the trace for either probe."""
    matrix = symmetric_matrix(seed=0)
    generator = torch.Generator().manual_seed(1)

    gaussian = torch.randn(PROBE_REPEATS, PROBE_DIMENSION, generator=generator)
    rademacher = (
        torch.randint(0, 2, (PROBE_REPEATS, PROBE_DIMENSION), generator=generator) * 2
        - 1
    ).float()

    truth = matrix.trace().item()
    for name, probes in (("gaussian", gaussian), ("rademacher", rademacher)):
        estimate = hutchinson_estimates(matrix, probes).mean().item()
        # The standard error at this repeat count is a few percent of the
        # Frobenius norm, which is what the tolerance is scaled to.
        tolerance = 4 * matrix.norm().item() / PROBE_REPEATS**0.5
        assert estimate == pytest.approx(truth, abs=tolerance), f"{name} is biased"


@pytest.mark.parametrize("diagonal_weight", [0.0, 2.0, 8.0])
def test_the_variance_gap_is_exactly_the_diagonal_energy(diagonal_weight):
    """The prediction that justifies RademacherNoise existing, and its limit.

        gaussian    variance = 2 * frobenius(A)^2
        rademacher  variance = 2 * (frobenius(A)^2 - sum of squared diagonal)

    Rademacher is smaller by exactly the diagonal energy, which makes it the
    minimum variance probe among distributions with independent entries
    (Hutchinson 1990; see also Epperly, "Don't use Gaussians in stochastic trace
    estimation", 2024).

    The size of that advantage is the honest caveat. For a random symmetric
    matrix the diagonal carries only about 3 percent of the Frobenius energy, so
    the advantage is real but smaller than the sampling noise at any practical
    number of passes. It only becomes worth having when the matrix is
    diagonally dominant, which is why this is parametrized rather than asserted
    once: whether the Hessian of a softmax output is diagonally dominant is an
    empirical question about the network, not something the estimator theory
    settles.
    """
    matrix = symmetric_matrix(seed=2) + diagonal_weight * torch.eye(PROBE_DIMENSION)
    generator = torch.Generator().manual_seed(3)

    gaussian = torch.randn(PROBE_REPEATS, PROBE_DIMENSION, generator=generator)
    rademacher = (
        torch.randint(0, 2, (PROBE_REPEATS, PROBE_DIMENSION), generator=generator) * 2
        - 1
    ).float()

    gaussian_variance = hutchinson_estimates(matrix, gaussian).var().item()
    rademacher_variance = hutchinson_estimates(matrix, rademacher).var().item()

    frobenius_squared = matrix.pow(2).sum().item()
    diagonal_energy = matrix.diagonal().pow(2).sum().item()

    assert gaussian_variance == pytest.approx(2 * frobenius_squared, rel=0.12)
    assert rademacher_variance == pytest.approx(
        2 * (frobenius_squared - diagonal_energy), rel=0.12
    )

    # The ordering is only measurable once the predicted gap clears the noise on
    # a variance estimated from this many probes, which is about 2 / sqrt(n).
    predicted_gap = 2 * diagonal_energy
    sampling_noise = 3 * gaussian_variance * (2 / PROBE_REPEATS) ** 0.5
    if predicted_gap > sampling_noise:
        assert rademacher_variance < gaussian_variance


def test_the_rademacher_advantage_is_negligible_on_a_generic_matrix():
    """Stated as a measurement so the prediction is not oversold.

    A symmetric matrix with independent entries puts about 1 part in dim of its
    energy on the diagonal, so the variance reduction is about 3 percent at
    dimension 64 and smaller as the dimension grows. Any claimed improvement from
    Rademacher probes on such a matrix would be noise.
    """
    matrix = symmetric_matrix(seed=9)

    frobenius_squared = matrix.pow(2).sum().item()
    diagonal_energy = matrix.diagonal().pow(2).sum().item()
    relative_advantage = diagonal_energy / frobenius_squared

    assert relative_advantage < 0.05, (
        f"the generic-matrix advantage is {relative_advantage:.3f}, larger than expected"
    )


def test_rademacher_perturbs_by_exactly_the_requested_scale():
    """Every entry moves by the same magnitude, which is what makes it isotropic."""
    module = build_perturbation("rademacher")(0.3).train()
    generator = torch.Generator().manual_seed(4)
    activation = torch.randn(4, 6, 8, generator=generator)

    deviation = module(activation) - activation
    per_sample_scale = 0.3 * activation.std(dim=(1, 2), keepdim=True)

    assert torch.allclose(
        deviation.abs(), per_sample_scale.expand_as(deviation), atol=1e-6
    )
    assert set(deviation.sign().unique().tolist()) <= {-1.0, 1.0}


def test_rademacher_and_gaussian_share_the_same_covariance_scale():
    """Both are zero mean with the same per-entry variance, so Sigma matches.

    That equality is what makes the comparison between them a comparison of
    estimator variance rather than of perturbation magnitude.
    """
    generator = torch.Generator().manual_seed(5)
    activation = torch.randn(256, 12, 16, generator=generator)

    deviations = {}
    for name in ("gaussian", "rademacher"):
        torch.manual_seed(6)
        module = build_perturbation(name)(0.25).train()
        deviations[name] = module(activation) - activation

    for name, deviation in deviations.items():
        assert deviation.mean().abs().item() < 0.01, f"{name} is not zero mean"

    gaussian_scale = deviations["gaussian"].std().item()
    rademacher_scale = deviations["rademacher"].std().item()
    assert rademacher_scale == pytest.approx(gaussian_scale, rel=0.05)


def test_rademacher_is_registered_as_stochastic():
    """It draws fresh signs per pass, so k passes are k different estimates."""
    assert "rademacher" in PERTURBATIONS
    assert "rademacher" not in DETERMINISTIC_PERTURBATIONS

    module = build_perturbation("rademacher")(0.3).train()
    activation = torch.randn(2, 5, 7, generator=torch.Generator().manual_seed(7))
    assert not torch.equal(module(activation), module(activation))


def test_rademacher_is_inert_in_eval_mode():
    """Like every probe here, it must vanish when the model is not being probed."""
    module = build_perturbation("rademacher")(0.5).eval()
    activation = torch.randn(2, 5, 7, generator=torch.Generator().manual_seed(8))

    assert torch.equal(module(activation), activation)
