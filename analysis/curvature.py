"""Curvature of the loss at a checkpoint, from Hessian-vector products alone.

3 statistics of the loss Hessian over the network's parameters, none of which
forms the matrix (86 M parameters would make it 7e15 entries): the top
eigenpairs by power iteration with deflation, the eigenvalue density by
stochastic Lanczos quadrature and the Gaussian-kernel curve BackdoorBench draws
from that density. The estimators are PyHessian's (Yao et al., "PyHessian:
Neural Networks Through the Lens of the Hessian", 2020) re-implemented over 1
product, with the vectors held flat rather than as PyHessian's per-parameter
lists so a dot product is 1 call. The curve is visual_hessian.py's
density_generate.

Every product is a double backward. The fused attention kernels have no double
backward, so the products run under the math scaled-dot-product kernel. They run
in the parameters' own dtype with no autocast, because the second derivative of a
bfloat16 forward is noise.

Hessian-vector product, as PyHessian's hessian_vector_product computes it:

    original form
        Hv = d/dtheta ( (dL/dtheta)^T v )

    symbols
        L       the mean loss over the batch
        theta   every parameter that requires a gradient, flattened
        H       the Hessian d^2 L / d theta^2
        v       a vector in parameter space, same length as theta

A batch may be split into micro-batches, PyHessian's dataloader_hv_product path.
The loss is a mean over rows, so the Hessian of the batch is the row-weighted
mean of the micro-batch Hessians, which the accumulation reproduces exactly.
ViT-B/16 at 224 keeps a 32-image double-backward graph at 16.3 GiB, so the tool
runs 2 micro-batches of 16 to stay under the login node's 15 GB budget.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from lightning import seed_everything
from torch.nn.attention import SDPBackend, sdpa_kernel

Batch = tuple[torch.Tensor, torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# PyHessian's guard against dividing by the norm of a zero vector.
NORMALISATION_EPSILON = 1e-6


def hessian_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The parameters the Hessian is taken over, every one that requires a gradient.

    Same selection as PyHessian's get_params_grad and in model.parameters() order,
    which is the order every flat vector of the estimators follows.
    """
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return parameters


def parameter_count(parameters: list[nn.Parameter]) -> int:
    """The length of a flat vector over these parameters."""
    count = sum(parameter.numel() for parameter in parameters)
    return count


def _split_like(
    vector: torch.Tensor, parameters: list[nn.Parameter]
) -> list[torch.Tensor]:
    """A flat parameter-space vector viewed as 1 tensor per parameter, no copy."""
    sizes = [parameter.numel() for parameter in parameters]
    pieces = torch.split(vector, sizes)  # tuple of (numel_i,)
    shaped = [
        piece.view_as(parameter) for piece, parameter in zip(pieces, parameters)
    ]  # parameter shapes
    return shaped


def _flatten(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Per-parameter tensors concatenated into 1 flat vector, (n_params,)."""
    flat = torch.cat([tensor.reshape(-1) for tensor in tensors])  # (n_params,)
    return flat


def _normalised(vector: torch.Tensor) -> torch.Tensor:
    """The vector at unit length, with PyHessian's epsilon in the denominator."""
    scaled = vector / (vector.norm().item() + NORMALISATION_EPSILON)  # (n_params,)
    return scaled


def _orthonormalised(vector: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    """The vector with its component along each basis vector removed, then normalised.

    Sequential Gram-Schmidt, as PyHessian's orthnormal, each projection taken
    against the already-updated vector. The basis may live on another device than
    the vector, which is how a Lanczos basis too large for the GPU is held on the
    CPU: the work moves to the basis, the result returns to the vector's device.
    """
    if not basis:
        unit = _normalised(vector)
        return unit

    # An explicit copy, so the in-place updates never reach the caller's vector
    # when it already lives on the basis device.
    working = vector.to(basis[0].device, copy=True)  # (n_params,)
    for basis_vector in basis:
        coefficient = torch.dot(working, basis_vector).item()
        working.sub_(basis_vector, alpha=coefficient)  # (n_params,)

    unit = _normalised(working).to(vector.device)  # (n_params,)
    return unit


def hessian_vector_product(
    model: nn.Module,
    loss_fn: LossFunction,
    batch: Batch,
    vector: torch.Tensor,
    micro_batch_size: int | None = None,
) -> torch.Tensor:
    """H v for the mean loss over the batch, (n_params,), by double backward.

    vector is flat over hessian_parameters(model). The batch is (images, labels)
    on any device, moved to the parameters' device chunk by chunk.
    micro_batch_size bounds the double-backward graph held at once, None keeps
    the batch whole. The model's mode is left as the caller set it, so a caller
    wanting the deterministic Hessian puts the model in eval mode first.
    """
    parameters = hessian_parameters(model)
    device = parameters[0].device
    images, labels = batch
    total = images.size(0)
    chunk = micro_batch_size if micro_batch_size is not None else total
    assert vector.shape == (parameter_count(parameters),), (
        f"vector has shape {tuple(vector.shape)}, the model has "
        f"{parameter_count(parameters)} parameters requiring a gradient"
    )
    vector_pieces = _split_like(vector, parameters)  # parameter shapes

    accumulated = torch.zeros_like(vector)  # (n_params,)
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, total, chunk):
            chunk_images = images[start : start + chunk].to(device)  # (chunk, 3, H, W)
            chunk_labels = labels[start : start + chunk].to(device)  # (chunk,)

            loss = loss_fn(model(chunk_images), chunk_labels)  # scalar
            gradients = torch.autograd.grad(
                loss, parameters, create_graph=True
            )  # parameter shapes, graph attached
            products = torch.autograd.grad(
                gradients, parameters, grad_outputs=vector_pieces
            )  # parameter shapes

            # The chunk's loss is a mean over its rows, so weighting the product
            # by the row count and dividing by the total afterwards gives the
            # product for the mean loss over the whole batch.
            accumulated += _flatten(products) * chunk_images.size(0)  # (n_params,)

    product = accumulated / total  # (n_params,)
    return product


def top_hessian_eigenpairs(
    model: nn.Module,
    loss_fn: LossFunction,
    batch: Batch,
    top_n: int = 2,
    max_iterations: int = 1000,
    tolerance: float = 1e-3,
    seed: int = 0,
    micro_batch_size: int | None = None,
) -> tuple[list[float], list[torch.Tensor]]:
    """The top_n eigenvalues of largest magnitude and their unit eigenvectors.

    Power iteration with deflation, PyHessian's eigenvalues() line for line: a
    Gaussian start vector, orthogonalisation against every eigenvector already
    found before each product, the Rayleigh quotient as the estimate and a stop
    when 2 consecutive quotients agree to tolerance in relative terms. 2 quirks are
    kept because the numbers upstream reports carry them: the eigenvalue kept at
    the stop is the earlier of the 2 agreeing quotients, and the eigenvector is
    the normalised product of the final iteration.

    original form (iteration k, for the j-th eigenpair)
        v_k    = orthnormal(v_k, {u_1 .. u_{j-1}})
        w      = H v_k
        lambda = w^T v_k
        v_k+1  = w / (||w|| + 1e-6)
        stop when |lambda_prev - lambda| / (|lambda_prev| + 1e-6) < tol

    symbols
        H          the batch Hessian, applied through hessian_vector_product
        u_i        the i-th eigenvector already found, deflated against
        lambda     the Rayleigh quotient, the eigenvalue estimate
        tol        the relative tolerance between consecutive estimates

    Returns eigenvalues in the order found, largest magnitude first, with the
    eigenvectors as flat (n_params,) tensors on the parameters' device.
    """
    assert top_n >= 1, "at least 1 eigenpair must be asked for"
    parameters = hessian_parameters(model)
    device = parameters[0].device
    dtype = parameters[0].dtype
    count = parameter_count(parameters)
    seed_everything(seed)

    eigenvalues: list[float] = []
    eigenvectors: list[torch.Tensor] = []
    for _ in range(top_n):
        vector = _normalised(torch.randn(count, device=device, dtype=dtype))
        eigenvalue = None
        for _ in range(max_iterations):
            vector = _orthonormalised(vector, eigenvectors)  # (n_params,)
            product = hessian_vector_product(
                model, loss_fn, batch, vector, micro_batch_size
            )  # (n_params,)
            rayleigh = torch.dot(product, vector).item()
            vector = _normalised(product)  # (n_params,)

            if eigenvalue is None:
                eigenvalue = rayleigh
            elif (
                abs(eigenvalue - rayleigh) / (abs(eigenvalue) + NORMALISATION_EPSILON)
                < tolerance
            ):
                break
            else:
                eigenvalue = rayleigh

        eigenvalues.append(eigenvalue)
        eigenvectors.append(vector)

    return eigenvalues, eigenvectors


def hessian_density(
    model: nn.Module,
    loss_fn: LossFunction,
    batch: Batch,
    lanczos_steps: int = 100,
    num_vectors: int = 1,
    seed: int = 0,
    micro_batch_size: int | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """The eigenvalue density by stochastic Lanczos quadrature, PyHessian's density().

    For each of num_vectors Rademacher start vectors, lanczos_steps steps of the
    Lanczos recurrence with full reorthogonalisation build the tridiagonal T whose
    eigenvalues are the quadrature nodes and whose eigenvectors' squared first
    components are the weights (Golub and Meurant, 2010, Theorem 6.2 as PyHessian
    applies it). Returns (nodes, weights), each num_vectors lists of lanczos_steps
    floats, the layout density_curve reads.

    original form (Lanczos step i, start v_0 Rademacher and normalised)
        w      = H v_i
        alpha_i = w^T v_i
        w      = w - alpha_i v_i - beta_i v_{i-1}
        beta_{i+1} = ||w||
        v_{i+1} = orthnormal(w, {v_0 .. v_i})
        T      = tridiag(beta, alpha, beta),  T = Q Lambda Q^T
        nodes  = diag(Lambda),  weights = (Q_{0,:})^2

    symbols
        H        the batch Hessian, applied through hessian_vector_product
        v_i      the i-th Lanczos vector, orthonormal to every earlier one
        alpha_i  the diagonal of T, the Rayleigh quotient at v_i
        beta_i   the off-diagonal of T, the residual norm
        Q        the eigenvectors of T as columns

    The Lanczos basis is kept on the CPU. Full reorthogonalisation needs every
    earlier vector, 100 of 86 M floats is 34 GB and the GPU budget is 15 GB. A
    zero residual, which PyHessian restarts from a Gaussian vector, is handled the
    same way. T is diagonalised by eigh rather than PyHessian's general eig,
    which on a symmetric tridiagonal matrix is the same decomposition with the
    real parts already taken.
    """
    parameters = hessian_parameters(model)
    device = parameters[0].device
    dtype = parameters[0].dtype
    count = parameter_count(parameters)
    seed_everything(seed)

    nodes_per_vector: list[list[float]] = []
    weights_per_vector: list[list[float]] = []
    for _ in range(num_vectors):
        rademacher = torch.randint(0, 2, (count,), device=device).to(dtype) * 2 - 1
        vector = _normalised(rademacher)  # (n_params,)
        basis = [vector.cpu()]
        alphas: list[float] = []
        betas: list[float] = []
        previous = vector
        residual = vector

        for step in range(lanczos_steps):
            if step > 0:
                beta = residual.norm().item()
                betas.append(beta)
                # A vanished residual means the Krylov space closed early, and
                # PyHessian carries on from a fresh Gaussian vector.
                restart = (
                    residual
                    if beta != 0.0
                    else torch.randn(count, device=device, dtype=dtype)
                )
                vector = _orthonormalised(restart, basis)  # (n_params,)
                basis.append(vector.cpu())

            product = hessian_vector_product(
                model, loss_fn, batch, vector, micro_batch_size
            )  # (n_params,)
            alpha = torch.dot(product, vector).item()
            alphas.append(alpha)
            if step == 0:
                residual = product - alpha * vector  # (n_params,)
            else:
                residual = (
                    product - alpha * vector - betas[-1] * previous
                )  # (n_params,)
            previous = vector

        tridiagonal = _tridiagonal(alphas, betas)  # (steps, steps)
        node_values, node_vectors = torch.linalg.eigh(tridiagonal)
        weights = node_vectors[0, :] ** 2  # (steps,)
        nodes_per_vector.append(node_values.tolist())
        weights_per_vector.append(weights.tolist())

    return nodes_per_vector, weights_per_vector


def _tridiagonal(alphas: list[float], betas: list[float]) -> torch.Tensor:
    """The symmetric tridiagonal Lanczos matrix T, (steps, steps), in float64.

    float64 because the eigenvectors' first components are squared into the
    quadrature weights, and the matrix is 100 by 100 so precision is free.
    """
    steps = len(alphas)
    matrix = torch.zeros(steps, steps, dtype=torch.float64)  # (steps, steps)
    for index, alpha in enumerate(alphas):
        matrix[index, index] = alpha
        if index < steps - 1:
            matrix[index + 1, index] = betas[index]
            matrix[index, index + 1] = betas[index]
    return matrix


def density_curve(
    eigenvalues: list[list[float]],
    weights: list[list[float]],
    num_bins: int = 10000,
    sigma_squared: float = 1e-5,
    overhead: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """The smoothed spectral density on a grid, visual_hessian.py's density_generate.

    Each quadrature node becomes a Gaussian bump of weight w on a grid that runs
    from the mean minimum node to the mean maximum node with the overhead added at
    each end. The kernel variance is sigma_squared times the grid's width when
    that width exceeds 1. The curve is averaged over the SLQ runs and divided by
    its rectangle-rule integral, upstream's np.sum(density) * dx, so it integrates
    to 1 on the grid. Returns (density, grid), both (num_bins,).

    original form
        rho(t) = (1/n_v) sum_l sum_i w_i^(l) g(t - lambda_i^(l))
        g(x)   = exp(-x^2 / (2 sigma^2)) / sqrt(2 pi sigma^2)

    symbols
        lambda_i^(l)  the i-th node of SLQ run l
        w_i^(l)       its weight
        n_v           the number of SLQ runs
        sigma^2       sigma_squared * max(1, lambda_max - lambda_min)
    """
    node_array = np.asarray(eigenvalues, dtype=np.float64)  # (runs, steps)
    weight_array = np.asarray(weights, dtype=np.float64)  # (runs, steps)

    lambda_max = np.mean(np.max(node_array, axis=1)) + overhead
    lambda_min = np.mean(np.min(node_array, axis=1)) - overhead
    grid = np.linspace(lambda_min, lambda_max, num=num_bins)  # (num_bins,)
    variance = sigma_squared * max(1.0, lambda_max - lambda_min)

    # Upstream evaluates the kernel in a double loop over bins and runs. Broadcasting
    # (runs, bins, steps) is the same sum without the loop.
    offsets = grid[None, :, None] - node_array[:, None, :]  # (runs, bins, steps)
    kernel = np.exp(-(offsets**2) / (2.0 * variance)) / np.sqrt(
        2 * np.pi * variance
    )  # (runs, bins, steps)
    per_run = np.sum(kernel * weight_array[:, None, :], axis=2)  # (runs, bins)
    density = np.mean(per_run, axis=0)  # (num_bins,)

    normalisation = np.sum(density) * (grid[1] - grid[0])
    normalised_density = density / normalisation  # (num_bins,)
    return normalised_density, grid
