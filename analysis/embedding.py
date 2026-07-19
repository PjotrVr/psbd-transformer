"""Two-dimensional projections of features for visualization: PCA and UMAP.

t-SNE is deliberately not included. UMAP preserves more global structure at
similar or lower cost, so for cluster visualization it is the better default,
and PCA covers the linear case. See the notes below for when to reach for each.

PCA is the honest first choice for the backdoor question because the backdoor is
hypothesized to be a linear direction, and a linear projection cannot invent
structure that is not there. UMAP reveals nonlinear cluster structure that PCA
misses, but it warps distances and can produce clusters that are artifacts of
its hyperparameters, so treat it as a qualitative illustration and always report
n_neighbors and min_dist.

Each function takes a [num_samples, dim] tensor and returns a [num_samples, 2]
numpy array.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA


def pca_project(features: torch.Tensor, num_components: int = 2) -> np.ndarray:
    return PCA(n_components=num_components).fit_transform(features.float().numpy())


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
    return projector.fit_transform(features.float().numpy())
