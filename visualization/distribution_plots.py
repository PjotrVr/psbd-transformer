"""Figures over the latent distribution statistics of analysis.distribution.

The 4 builders here used to live beside the statistics they draw and moved so
that analysis/ imports no matplotlib. Each takes the statistics' own inputs, a
distribution table or a pair of (num_samples, dim) feature tensors, and returns
a Figure rather than saving a file, which makes them usable in a notebook and
lets a script choose the path and the format.
"""

import matplotlib.figure
import matplotlib.pyplot as plt
import torch

from analysis.direction import backdoor_direction, project_onto_direction
from analysis.distribution import mahalanobis_distances
from analysis.embedding import pca_project, umap_project


def plot_layer_profile(
    table: list[dict[str, float]],
    metrics: tuple[str, ...] = (
        "separation_auroc",
        "cka",
        "standardized_shift",
        "direction_norm",
    ),
) -> matplotlib.figure.Figure:
    """A panel per metric against layer index, sharing the x axis.

    This is the view that answers "at what depth does the backdoor appear",
    which is the question the placement results turn on.
    """
    if not table:
        raise ValueError("no layers to plot; the distribution table is empty")

    layers = [row["layer"] for row in table]
    present = [metric for metric in metrics if metric in table[0]]
    if not present:
        raise ValueError(f"none of the requested metrics {metrics} are in the table")

    figure, axes = plt.subplots(
        len(present), 1, figsize=(7, 2.2 * len(present)), sharex=True, squeeze=False
    )
    for axis, metric in zip(axes[:, 0], present):
        axis.plot(layers, [row[metric] for row in table], marker="o", markersize=4)
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(alpha=0.3)
        if metric in ("separation_auroc", "cka"):
            axis.axhline(0.5 if metric == "separation_auroc" else 1.0, ls="--", lw=1)

    axes[-1, 0].set_xlabel("layer")
    figure.tight_layout()
    return figure


def plot_projection_histogram(
    clean_features: torch.Tensor,
    backdoor_features: torch.Tensor,
    bins: int = 60,
) -> matplotlib.figure.Figure:
    """Both populations projected onto the backdoor direction, as histograms.

    The single most direct picture of the distributional claim: overlap means the
    trigger is not separable at this layer by a linear read of this direction.
    """
    direction = backdoor_direction(clean_features, backdoor_features)
    clean_projected = project_onto_direction(clean_features, direction).numpy()
    backdoor_projected = project_onto_direction(backdoor_features, direction).numpy()

    figure, axis = plt.subplots(figsize=(7, 4))
    span = (
        min(clean_projected.min(), backdoor_projected.min()),
        max(clean_projected.max(), backdoor_projected.max()),
    )
    axis.hist(clean_projected, bins=bins, range=span, alpha=0.6, label="clean")
    axis.hist(backdoor_projected, bins=bins, range=span, alpha=0.6, label="backdoor")
    axis.set_xlabel("projection onto the backdoor direction")
    axis.set_ylabel("samples")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_distance_distribution(
    clean_features: torch.Tensor,
    backdoor_features: torch.Tensor,
    bins: int = 60,
) -> matplotlib.figure.Figure:
    """Mahalanobis distance from the clean distribution, for both populations.

    Clean samples are scored against the distribution they define, so their curve
    is the reference shape. A backdoor curve shifted right means triggered images
    are outliers under the model's own notion of clean variation.
    """
    clean_distances = mahalanobis_distances(clean_features, clean_features).numpy()
    backdoor_distances = mahalanobis_distances(
        clean_features, backdoor_features
    ).numpy()

    figure, axis = plt.subplots(figsize=(7, 4))
    span = (0.0, max(clean_distances.max(), backdoor_distances.max()))
    axis.hist(clean_distances, bins=bins, range=span, alpha=0.6, label="clean")
    axis.hist(backdoor_distances, bins=bins, range=span, alpha=0.6, label="backdoor")
    axis.set_xlabel("mahalanobis distance from the clean distribution")
    axis.set_ylabel("samples")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_embedding_scatter(
    clean_features: torch.Tensor,
    backdoor_features: torch.Tensor,
    method: str = "pca",
    clean_labels: torch.Tensor | None = None,
) -> matplotlib.figure.Figure:
    """Both populations in 2 dimensions, projected jointly so the axes agree.

    Projecting the concatenation rather than each population separately is what
    makes the 2 clouds comparable. Fitting twice would give 2 unrelated
    coordinate systems. Passing clean_labels colours the clean cloud by class,
    which shows whether the backdoor cloud lands on the target class or beside it.
    """
    combined = torch.cat([clean_features, backdoor_features], dim=0)
    if method == "pca":
        projected = pca_project(combined, num_components=2)
    elif method == "umap":
        projected = umap_project(combined)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    clean_count = clean_features.shape[0]
    figure, axis = plt.subplots(figsize=(6.5, 6))

    if clean_labels is None:
        axis.scatter(
            projected[:clean_count, 0],
            projected[:clean_count, 1],
            s=8,
            alpha=0.5,
            label="clean",
        )
    else:
        axis.scatter(
            projected[:clean_count, 0],
            projected[:clean_count, 1],
            s=8,
            alpha=0.5,
            c=clean_labels.numpy(),
            cmap="tab20",
            label="clean by class",
        )
    axis.scatter(
        projected[clean_count:, 0],
        projected[clean_count:, 1],
        s=8,
        alpha=0.5,
        c="black",
        marker="x",
        label="backdoor",
    )
    axis.set_title(f"{method.upper()} of the latent distribution")
    axis.legend()
    figure.tight_layout()
    return figure
