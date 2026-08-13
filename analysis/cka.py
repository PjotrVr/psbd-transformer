"""Centered Kernel Alignment (CKA), biased and debiased.

CKA scores how similar two sets of features are on the same inputs, invariant to
rotation and isotropic scaling. Two flavors are provided.

The biased estimator is simple and fine when the sample count is much larger than
the feature dimension. When it is not, it assigns even unrelated representations a
positive baseline similarity that shrinks only as the sample count grows.

The debiased estimator (Nguyen et al., 2021, built on the unbiased HSIC of Song et
al., 2012) removes that baseline. Its expectation is 0 under independence, so
unrelated representations score near 0 and values are comparable across sample
sizes. Prefer it when the sample count is close to or below the feature dimension,
which for ViT-B/16 CLS features of dimension 768 means a few hundred samples. It
can fall slightly outside 0 to 1, which is expected for a finite-sample unbiased
estimate.

The placement experiment uses this to compare a layer to itself with dropout off
versus on, separately for clean and backdoor, which shows which placement perturbs
the backdoor signal more. Both feature tensors are [num_samples, dim] on the same
samples in the same order.
"""

import torch


def _center_columns(features: torch.Tensor) -> torch.Tensor:
    return features - features.mean(dim=0, keepdim=True)


def linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Biased linear CKA in feature space, cheap when the dimension is small.

    original form
        CKA(X, Y) = frobenius_norm(Y^T X)^2 / (frobenius_norm(X^T X) * frobenius_norm(Y^T Y))
    simplified form
        similarity = squared alignment of the centered cross gram over the self grams
    """
    x = _center_columns(features_x.float())
    y = _center_columns(features_y.float())
    cross = (y.t() @ x).norm() ** 2
    normalizer = (x.t() @ x).norm() * (y.t() @ y).norm()
    return (cross / normalizer.clamp_min(1e-12)).item()


def _linear_gram(features: torch.Tensor) -> torch.Tensor:
    # Double precision because the unbiased estimator sums over n squared gram
    # entries, which loses accuracy in float32 for larger sample counts.
    x = features.double()
    return x @ x.t()


def _rbf_gram(features: torch.Tensor, sigma: float | None) -> torch.Tensor:
    x = features.double()
    squared_distances = torch.cdist(x, x) ** 2
    if sigma is None:
        # Median heuristic removes the bandwidth hyperparameter.
        bandwidth = squared_distances.median().sqrt().clamp_min(1e-8)
    else:
        bandwidth = torch.tensor(float(sigma), dtype=torch.double)
    return torch.exp(-squared_distances / (2.0 * bandwidth**2))


def biased_hsic(gram_k: torch.Tensor, gram_l: torch.Tensor) -> torch.Tensor:
    """Biased HSIC, centering the grams then summing their elementwise product.

    original form (H is the centering matrix I - (1/n) 1 1^T)
        HSIC(K, L) = 1/(n-1)^2 * trace(K H L H)
    simplified form
        center K, then sum it elementwise times L
    """
    n = gram_k.size(0)
    centering = (
        torch.eye(n, dtype=gram_k.dtype) - torch.ones(n, n, dtype=gram_k.dtype) / n
    )
    centered_k = centering @ gram_k @ centering
    return (centered_k * gram_l).sum() / (n - 1) ** 2


def unbiased_hsic(gram_k: torch.Tensor, gram_l: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC (Song et al., 2012), which removes the finite-sample bias.

    K_tilde and L_tilde are the grams with their diagonals set to 0.

    original form
        HSIC_u(K, L) = 1/(n(n-3)) * [ trace(K_tilde L_tilde)
                                      + (1^T K_tilde 1)(1^T L_tilde 1) / ((n-1)(n-2))
                                      - 2/(n-2) * 1^T K_tilde L_tilde 1 ]
    simplified form
        dot_term     = sum of elementwise K_tilde times L_tilde
        total_k      = sum of all entries of K_tilde,  total_l = same for L_tilde
        row_coupling = sum over rows of (row_sum of K_tilde) times (row_sum of L_tilde)
        hsic = (dot_term + total_k*total_l/((n-1)(n-2)) - 2*row_coupling/(n-2)) / (n(n-3))
    """
    n = gram_k.size(0)
    if n < 4:
        raise ValueError("Unbiased HSIC needs at least 4 samples")
    k = gram_k.clone().fill_diagonal_(0.0)
    l = gram_l.clone().fill_diagonal_(0.0)
    dot_term = (k * l).sum()
    total_k = k.sum()
    total_l = l.sum()
    row_coupling = (k.sum(dim=0) * l.sum(dim=0)).sum()
    return (
        dot_term
        + total_k * total_l / ((n - 1) * (n - 2))
        - 2.0 * row_coupling / (n - 2)
    ) / (n * (n - 3))


def _cka_from_grams(gram_x, gram_y, hsic) -> float:
    numerator = hsic(gram_x, gram_y)
    denominator = (hsic(gram_x, gram_x) * hsic(gram_y, gram_y)).clamp_min(1e-12).sqrt()
    return (numerator / denominator).item()


def debiased_linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Linear CKA with the bias removed, comparable across sample sizes."""
    return _cka_from_grams(
        _linear_gram(features_x), _linear_gram(features_y), unbiased_hsic
    )


def rbf_cka(
    features_x: torch.Tensor, features_y: torch.Tensor, sigma: float | None = None
) -> float:
    """Biased kernel CKA with an RBF kernel, for nonlinear similarity."""
    return _cka_from_grams(
        _rbf_gram(features_x, sigma), _rbf_gram(features_y, sigma), biased_hsic
    )


def debiased_rbf_cka(
    features_x: torch.Tensor, features_y: torch.Tensor, sigma: float | None = None
) -> float:
    """Debiased kernel CKA with an RBF kernel."""
    return _cka_from_grams(
        _rbf_gram(features_x, sigma), _rbf_gram(features_y, sigma), unbiased_hsic
    )


def layerwise_cka_matrix(
    features_by_layer_a: dict[int, torch.Tensor],
    features_by_layer_b: dict[int, torch.Tensor],
    debiased: bool = True,
) -> torch.Tensor:
    """CKA between every layer of one model and every layer of another.

    Grams are precomputed once per layer, so the estimator runs on gram pairs
    rather than recomputing the gram inside every cell.
    """
    layers_a = sorted(features_by_layer_a)
    layers_b = sorted(features_by_layer_b)
    hsic = unbiased_hsic if debiased else biased_hsic

    grams_a = {layer: _linear_gram(features_by_layer_a[layer]) for layer in layers_a}
    grams_b = {layer: _linear_gram(features_by_layer_b[layer]) for layer in layers_b}

    matrix = torch.zeros(len(layers_a), len(layers_b))
    for i, layer_a in enumerate(layers_a):
        for j, layer_b in enumerate(layers_b):
            matrix[i, j] = _cka_from_grams(grams_a[layer_a], grams_b[layer_b], hsic)
    return matrix
