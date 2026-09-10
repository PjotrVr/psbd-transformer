"""Centered Kernel Alignment (CKA), biased and debiased.

CKA scores how similar 2 sets of features are on the same inputs, invariant to
rotation and isotropic scaling. 2 estimators are provided.

The biased estimator is simple and fine when the sample count is much larger than
the feature dimension. When it is not, it assigns even unrelated representations a
positive baseline similarity that shrinks only as the sample count grows.

The debiased estimator (Nguyen et al., 2021, built on the unbiased HSIC of Song et
al., 2012) removes that baseline. Its expectation is 0 under independence, so
unrelated representations score near 0 and values are comparable across sample
sizes. Prefer it when the sample count is close to or below the feature dimension,
which for ViT-B/16 CLS features of dimension 768 means sample counts in the
hundreds. It can fall slightly outside 0 to 1, which is expected for a
finite-sample unbiased estimate.

The placement experiment uses this to compare a layer to itself with dropout off
versus on, separately for clean and backdoor, which shows which placement perturbs
the backdoor signal more. Both feature tensors are (num_samples, dim) on the same
samples in the same order.
"""

import torch

# Guards every normalization below. The denominators are products of Frobenius
# norms or of HSIC self-terms, both of which are 0 for a constant representation.
NORMALIZER_FLOOR = 1e-12


def _center_columns(features: torch.Tensor) -> torch.Tensor:
    """Subtract each feature dimension's mean, leaving the (num_samples, dim) shape."""
    centered = features - features.mean(dim=0, keepdim=True)
    return centered


def linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Biased linear CKA in feature space, cheap when the dimension is small.

    original form
        CKA(X, Y) = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )
    simplified form
        similarity = squared alignment of the centered cross gram over the
                     2 self grams
    """
    x = _center_columns(features_x.float())  # (num_samples, dim_x)
    y = _center_columns(features_y.float())  # (num_samples, dim_y)

    cross = (y.t() @ x).norm() ** 2
    normalizer = (x.t() @ x).norm() * (y.t() @ y).norm()

    similarity = (cross / normalizer.clamp_min(NORMALIZER_FLOOR)).item()
    return similarity


def _linear_gram(features: torch.Tensor) -> torch.Tensor:
    """The (num_samples, num_samples) linear gram matrix, in double precision.

    Double because the unbiased estimator sums over n squared gram entries, which
    loses accuracy in float32 for larger sample counts.
    """
    x = features.double()

    gram = x @ x.t()  # (num_samples, num_samples)
    return gram


def _rbf_gram(features: torch.Tensor, sigma: float | None) -> torch.Tensor:
    """The (num_samples, num_samples) RBF gram matrix, in double precision."""
    x = features.double()
    squared_distances = torch.cdist(x, x) ** 2  # (num_samples, num_samples)

    if sigma is None:
        # Median heuristic removes the bandwidth hyperparameter.
        bandwidth = squared_distances.median().sqrt().clamp_min(1e-8)
    else:
        bandwidth = torch.tensor(float(sigma), dtype=torch.double)

    gram = torch.exp(-squared_distances / (2.0 * bandwidth**2))
    return gram


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
    )  # (n, n)

    centered_k = centering @ gram_k @ centering

    hsic = (centered_k * gram_l).sum() / (n - 1) ** 2
    return hsic


def unbiased_hsic(gram_k: torch.Tensor, gram_l: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC (Song et al., 2012), which removes the finite-sample bias.

    K_tilde and L_tilde are the grams with their diagonals set to 0.

        original form
            HSIC_u(K, L) = 1/(n(n-3)) * [ trace(K_tilde L_tilde)
                                          + (1^T K_tilde 1)(1^T L_tilde 1)
                                            / ((n-1)(n-2))
                                          - 2/(n-2) * 1^T K_tilde L_tilde 1 ]
        simplified form
            dot_term     = sum of elementwise K_tilde times L_tilde
            total_k      = sum of all entries of K_tilde, total_l the same for L_tilde
            row_coupling = sum over rows of (row sum of K_tilde) times (row sum of L_tilde)
            hsic = (dot_term + total_k*total_l/((n-1)(n-2)) - 2*row_coupling/(n-2))
                   / (n(n-3))
    """
    n = gram_k.size(0)
    if n < 4:
        raise ValueError("Unbiased HSIC needs at least 4 samples")

    k = gram_k.clone().fill_diagonal_(0.0)  # (n, n)
    m = gram_l.clone().fill_diagonal_(0.0)  # (n, n)
    dot_term = (k * m).sum()
    total_k = k.sum()
    total_l = m.sum()
    row_coupling = (k.sum(dim=0) * m.sum(dim=0)).sum()

    hsic = (
        dot_term
        + total_k * total_l / ((n - 1) * (n - 2))
        - 2.0 * row_coupling / (n - 2)
    ) / (n * (n - 3))
    return hsic


def _cka_from_grams(gram_x, gram_y, hsic) -> float:
    """Normalize a cross HSIC by the geometric mean of the 2 self HSICs."""
    numerator = hsic(gram_x, gram_y)
    denominator = (
        (hsic(gram_x, gram_x) * hsic(gram_y, gram_y)).clamp_min(NORMALIZER_FLOOR).sqrt()
    )

    similarity = (numerator / denominator).item()
    return similarity


def debiased_linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Linear CKA with the bias removed, comparable across sample sizes."""
    similarity = _cka_from_grams(
        _linear_gram(features_x), _linear_gram(features_y), unbiased_hsic
    )
    return similarity


def rbf_cka(
    features_x: torch.Tensor, features_y: torch.Tensor, sigma: float | None = None
) -> float:
    """Biased kernel CKA with an RBF kernel, for nonlinear similarity."""
    similarity = _cka_from_grams(
        _rbf_gram(features_x, sigma), _rbf_gram(features_y, sigma), biased_hsic
    )
    return similarity


def debiased_rbf_cka(
    features_x: torch.Tensor, features_y: torch.Tensor, sigma: float | None = None
) -> float:
    """Debiased kernel CKA with an RBF kernel."""
    similarity = _cka_from_grams(
        _rbf_gram(features_x, sigma), _rbf_gram(features_y, sigma), unbiased_hsic
    )
    return similarity


def layerwise_cka_matrix(
    features_by_layer_a: dict[int, torch.Tensor],
    features_by_layer_b: dict[int, torch.Tensor],
    debiased: bool = True,
) -> torch.Tensor:
    """CKA between every layer of a model and every layer of another model.

    Returns a (len(layers_a), len(layers_b)) matrix indexed by the 2 sorted layer
    lists. Grams are precomputed once per layer, so the estimator runs on gram
    pairs rather than recomputing the gram inside every cell.
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
