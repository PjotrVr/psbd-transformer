"""2-dimensional projections of features for visualization: PCA and UMAP.

t-SNE is not included. UMAP preserves more global structure at similar or lower
cost, so for cluster visualization it is the better default. PCA covers the
linear case.

PCA is the honest first choice for the backdoor question because the backdoor is
hypothesized to be a linear direction, and a linear projection cannot invent
structure that is not there. UMAP reveals nonlinear cluster structure that PCA
misses, but it warps distances and can produce clusters that are artifacts of its
hyperparameters, so treat it as a qualitative illustration and always report
n_neighbors and min_dist.

Each function takes a (num_samples, dim) tensor and returns a (num_samples, 2)
numpy array.

tsne_project was added later for the BackdoorBench tool mirror in
visualization.cheap_tools, which draws t-SNE beside UMAP so a reader used to
the upstream figure sees the same view. The preference above stands for the
project's own figures.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_project(features: torch.Tensor, num_components: int = 2) -> np.ndarray:
    """Linear projection onto the leading num_components principal directions."""
    projected = PCA(n_components=num_components).fit_transform(
        features.float().numpy()
    )  # (num_samples, num_components)
    return projected


def umap_project(
    features: torch.Tensor,
    num_neighbors: int = 15,
    min_distance: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """UMAP projection. Requires the umap-learn package.

    num_neighbors trades local detail for global structure, and min_distance
    controls how tightly points may pack. Both change the picture, so report them.
    """
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "umap_project needs umap-learn, install it with pip install umap-learn"
        ) from error

    projector = umap.UMAP(
        n_neighbors=num_neighbors, min_dist=min_distance, random_state=seed
    )

    projected = projector.fit_transform(features.float().numpy())  # (num_samples, 2)
    return projected


def tsne_project(features: torch.Tensor, seed: int = 0) -> np.ndarray:
    """t-SNE projection with BackdoorBench's settings, (num_samples, 2).

    visual_utils.get_embedding (line 550) builds
    TSNE(n_components=2, init="random", random_state=0) and takes fit_transform,
    so the perplexity is sklearn's default of 30 and the sample count must
    exceed it. Only the seed is exposed.
    """
    projector = TSNE(n_components=2, init="random", random_state=seed)

    projected = projector.fit_transform(features.float().numpy())  # (num_samples, 2)
    return projected


def unit_square(embedding: np.ndarray) -> np.ndarray:
    """The embedding min-max scaled to [0, 1] per axis, same shape.

    visual_utils.plot_embedding (lines 496 and 497) rescales every embedding
    this way before drawing, so the axes of a t-SNE, a UMAP and a PCA figure
    all run from 0 to 1 and the 3 read alike. An axis with no spread divides
    0 by 0, which is left to surface rather than hidden, since such an
    embedding has nothing to draw.
    """
    lowest = np.min(embedding, axis=0)  # (2,)
    highest = np.max(embedding, axis=0)  # (2,)

    scaled = (embedding - lowest) / (highest - lowest)  # (num_samples, 2)
    return scaled
