"""The Hessian estimators against the exact Hessian of a net small enough to form it.

A 2-layer tanh net of 197 parameters, so torch.autograd.functional.hessian can
build the full (197, 197) matrix and every estimator is checked against it: the
product against H v, power iteration against the largest-magnitude eigenvalue,
the Lanczos quadrature's nodes against the spectrum and its first moment against
the trace over the parameter count, and the density curve against its own
normalisation. float64 throughout, so a disagreement is the estimator's and not
the arithmetic's.
"""

import pytest
import torch
import torch.nn as nn
from lightning import seed_everything
from torch.func import functional_call

from analysis.curvature import (
    NORMALISATION_EPSILON,
    density_curve,
    hessian_density,
    hessian_parameters,
    hessian_vector_product,
    parameter_count,
    top_hessian_eigenpairs,
)

INPUT_DIM = 10
HIDDEN_DIM = 12
NUM_CLASSES = 5
BATCH = 32


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    """torch defaults to 64 threads on the 128-core login node, where the tiny CPU
    ops of a toy model run 1000 times slower than at 8 from OpenMP overhead."""
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 8))
    yield
    torch.set_num_threads(previous)


def build_case(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.Tanh(), nn.Linear(HIDDEN_DIM, NUM_CLASSES)
    ).double()
    images = torch.randn(BATCH, INPUT_DIM, dtype=torch.float64)
    labels = torch.randint(0, NUM_CLASSES, (BATCH,))
    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn, (images, labels)


def exact_hessian(model, loss_fn, batch) -> torch.Tensor:
    """The full Hessian of the batch loss over the flat parameter vector, (n, n)."""
    parameters = hessian_parameters(model)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    sizes = [p.numel() for p in parameters]
    images, labels = batch

    def loss_of_flat(flat):
        pieces = torch.split(flat, sizes)
        values = {
            name: piece.view_as(p) for name, piece, p in zip(names, pieces, parameters)
        }
        return loss_fn(functional_call(model, values, (images,)), labels)

    flat = torch.cat([p.detach().reshape(-1) for p in parameters])  # (n,)
    hessian = torch.autograd.functional.hessian(loss_of_flat, flat)  # (n, n)
    return hessian


@pytest.fixture(scope="module")
def case():
    model, loss_fn, batch = build_case()
    hessian = exact_hessian(model, loss_fn, batch)
    return model, loss_fn, batch, hessian


def test_the_net_has_about_200_parameters(case):
    model, _, _, hessian = case
    count = parameter_count(hessian_parameters(model))
    assert count == 197
    assert hessian.shape == (197, 197)
    assert torch.allclose(hessian, hessian.T, atol=1e-10), "a Hessian is symmetric"


def test_hessian_vector_product_matches_the_exact_product(case):
    model, loss_fn, batch, hessian = case
    torch.manual_seed(1)
    vector = torch.randn(197, dtype=torch.float64)

    product = hessian_vector_product(model, loss_fn, batch, vector)

    assert product.shape == (197,)
    assert torch.allclose(product, hessian @ vector, atol=1e-9, rtol=1e-7)


def test_hessian_vector_product_over_micro_batches_is_the_batch_product(case):
    """The row-weighted accumulation over chunks must equal the whole-batch product
    exactly, since the mean loss is linear in the per-row losses."""
    model, loss_fn, batch, hessian = case
    torch.manual_seed(2)
    vector = torch.randn(197, dtype=torch.float64)

    whole = hessian_vector_product(model, loss_fn, batch, vector)
    chunked = hessian_vector_product(model, loss_fn, batch, vector, micro_batch_size=7)

    assert torch.allclose(whole, chunked, atol=1e-10)
    assert torch.allclose(chunked, hessian @ vector, atol=1e-9, rtol=1e-7)


def test_hessian_vector_product_refuses_a_vector_of_the_wrong_length(case):
    model, loss_fn, batch, _ = case
    with pytest.raises(AssertionError, match="197 parameters"):
        hessian_vector_product(model, loss_fn, batch, torch.zeros(196))


def test_top_eigenvalue_is_within_1e_3_relative_of_the_exact_one(case):
    model, loss_fn, batch, hessian = case
    exact = torch.linalg.eigvalsh(hessian)  # (n,), ascending
    largest_magnitude = exact[exact.abs().argmax()].item()

    eigenvalues, eigenvectors = top_hessian_eigenpairs(
        model, loss_fn, batch, top_n=2, max_iterations=5000, tolerance=1e-7
    )

    assert abs(eigenvalues[0] - largest_magnitude) / abs(largest_magnitude) < 1e-3
    assert len(eigenvectors) == 2 and eigenvectors[0].shape == (197,)
    assert abs(eigenvectors[0].norm().item() - 1.0) < 1e-5
    # The returned vector is an eigenvector to the same tolerance.
    residual = hessian @ eigenvectors[0] - eigenvalues[0] * eigenvectors[0]
    assert residual.norm().item() / abs(largest_magnitude) < 1e-2


def test_second_eigenpair_is_deflated_against_the_first(case):
    model, loss_fn, batch, hessian = case
    exact = torch.linalg.eigvalsh(hessian)
    by_magnitude = exact[exact.abs().argsort(descending=True)]  # (n,)

    eigenvalues, eigenvectors = top_hessian_eigenpairs(
        model, loss_fn, batch, top_n=2, max_iterations=5000, tolerance=1e-8
    )

    assert abs(torch.dot(eigenvectors[0], eigenvectors[1]).item()) < 1e-4
    assert abs(eigenvalues[1] - by_magnitude[1].item()) / abs(by_magnitude[1]) < 1e-2


def test_lanczos_nodes_are_ritz_values_and_the_first_moment_is_exact(case):
    """The toy Hessian has a 13-dimensional null space, so its Krylov space closes
    at step 185 and a run to 197 steps hits a breakdown PyHessian's beta != 0
    restart never sees in floating point. At 120 steps the Ritz properties hold:
    every node lies inside the spectrum, the extreme nodes have converged to the
    extreme eigenvalues, the weights sum to 1 and the first moment equals the
    start vector's Rayleigh quotient exactly, the quadrature's defining property."""
    model, loss_fn, batch, hessian = case
    exact = torch.linalg.eigvalsh(hessian)  # (n,), ascending
    spread = (exact[-1] - exact[0]).item()

    nodes, weights = hessian_density(model, loss_fn, batch, lanczos_steps=120, seed=0)

    node_values = torch.tensor(nodes[0], dtype=torch.float64)  # (120,)
    assert len(nodes) == 1 and node_values.shape == (120,)
    assert (node_values >= exact[0] - 1e-6).all() and (
        node_values <= exact[-1] + 1e-6
    ).all()
    assert abs(node_values.max() - exact[-1]).item() < 1e-4 * spread
    assert abs(node_values.min() - exact[0]).item() < 1e-4 * spread
    assert abs(sum(weights[0]) - 1.0) < 1e-8

    # The same draw the estimator made, PyHessian's epsilon included.
    seed_everything(0)
    start = torch.randint(0, 2, (197,)).double() * 2 - 1  # (n,)
    start = start / (start.norm().item() + NORMALISATION_EPSILON)
    rayleigh = (start @ hessian @ start).item()
    first_moment = sum(node * weight for node, weight in zip(nodes[0], weights[0]))
    assert first_moment == pytest.approx(rayleigh, rel=1e-8)


def test_slq_first_moment_is_within_5_percent_of_trace_over_count(case):
    """Hutchinson's estimator has a relative std of 0.87 per Rademacher vector on
    the plain CE Hessian, whose trace is small against its Frobenius norm at a
    random init, so no feasible vector count reaches 5%. A ridge term of weight 1
    adds the identity, the trace then dominates and the per-vector std falls to
    1.7%, which puts 8 vectors at 0.6% and the 5% bound at 8 standard deviations."""
    model, loss_fn, batch, hessian = case
    parameters = hessian_parameters(model)

    def ridge_loss(logits, labels):
        penalty = 0.5 * sum((parameter**2).sum() for parameter in parameters)
        return loss_fn(logits, labels) + penalty

    ridge_hessian = hessian + torch.eye(197, dtype=torch.float64)  # (n, n)
    expected = torch.trace(ridge_hessian).item() / 197

    nodes, weights = hessian_density(
        model, ridge_loss, batch, lanczos_steps=40, num_vectors=8, seed=3
    )
    moments = [
        sum(node * weight for node, weight in zip(run_nodes, run_weights))
        for run_nodes, run_weights in zip(nodes, weights)
    ]
    first_moment = sum(moments) / len(moments)

    assert abs(first_moment - expected) / abs(expected) < 0.05


def test_density_curve_integrates_to_1_and_peaks_at_the_node():
    density, grid = density_curve([[0.5, 2.0]], [[0.25, 0.75]], num_bins=2001)

    assert density.shape == (2001,) and grid.shape == (2001,)
    assert abs(density.sum() * (grid[1] - grid[0]) - 1.0) < 1e-9
    assert grid[0] == pytest.approx(0.49) and grid[-1] == pytest.approx(2.01)
    assert abs(grid[density.argmax()] - 2.0) < 2e-3, "the heavier node is the peak"
