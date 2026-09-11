"""How the clean and backdoor latent distributions differ, layer by layer.

The rest of this package answers pointwise questions: which direction the trigger
writes into (direction.py), how similar 2 representations are overall (cka.py),
where the points land in 2 dimensions (embedding.py). This module answers the
distributional question. Given the same images with and without the trigger, it
asks how far apart the 2 populations sit, how separable they are, whether the
backdoor population collapses onto a smaller subspace and whether it drifts
toward the attack's target class.

Every function takes plain (num_samples, dim) float tensors and returns plain
numbers or tensors, with no plotting and no file writing, so a notebook, a script
and a test all call the same code. The figures over these statistics live in
visualization.distribution_plots, which returns a Figure rather than saving a
file, so a notebook draws them too.

Swin's blocks change width between stages (96, 192, 384, 768), so a per-layer
table has layers of different dim. Every statistic here is computed within a
layer, so that is fine, but a raw direction norm at layer 2 is not comparable to
a norm at layer 20 without accounting for it.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .cka import debiased_linear_cka
from .direction import backdoor_direction, project_onto_direction

# A covariance estimated from hundreds of samples in 768 dimensions is singular,
# so Mahalanobis distance needs a shrinkage term. 0.1 is the Ledoit-Wolf ballpark
# and is applied as a fixed fraction of the mean eigenvalue rather than estimated,
# because the estimate itself is unstable in this regime.
COVARIANCE_SHRINKAGE = 0.1

# Eigenvalues below this fraction of the largest are treated as numerical noise
# when measuring how many directions a population actually occupies.
SPECTRUM_FLOOR = 1e-12


def separation_auroc(
    clean_scores: torch.Tensor, backdoor_scores: torch.Tensor
) -> float:
    """AUROC of a 1-dimensional score separating clean from backdoor samples.

    0.5 means the 2 populations are indistinguishable along this score and 1.0
    means perfectly separated. Values below 0.5 mean the score separates them in
    the opposite direction, which is information rather than failure, so it is
    reported unflipped.
    """
    scores = torch.cat([clean_scores, backdoor_scores]).float().numpy()
    labels = np.concatenate(
        [np.zeros(len(clean_scores)), np.ones(len(backdoor_scores))]
    )
    if len(np.unique(labels)) < 2:
        return float("nan")
    if not np.isfinite(scores).all():
        # An upstream statistic already reported itself undefined, for instance a
        # layer too small to hold any samples out for cross-fitting. Passing that
        # through would raise inside sklearn and lose every other layer's row.
        return float("nan")

    auroc = float(roc_auc_score(labels, scores))
    return auroc


def standardized_mean_shift(
    clean_features: torch.Tensor, backdoor_features: torch.Tensor
) -> float:
    """Cohen's d between the 2 populations along their mean-difference direction.

    original form
        d = (mean_1 - mean_2) / pooled_standard_deviation
    simplified form, projected onto the direction that separates the means
        shift = (mean of backdoor projections - mean of clean projections)
                / pooled standard deviation of those projections

    Reported in units of within-population spread, so it stays comparable across
    layers of different width where a raw distance would not.
    """
    direction = backdoor_features.mean(dim=0) - clean_features.mean(dim=0)
    clean_projected = project_onto_direction(clean_features, direction)
    backdoor_projected = project_onto_direction(backdoor_features, direction)

    clean_count = len(clean_projected)
    backdoor_count = len(backdoor_projected)
    if clean_count < 2 or backdoor_count < 2:
        return float("nan")

    pooled_variance = (
        (clean_count - 1) * clean_projected.var(unbiased=True)
        + (backdoor_count - 1) * backdoor_projected.var(unbiased=True)
    ) / (clean_count + backdoor_count - 2)
    pooled_deviation = pooled_variance.clamp_min(SPECTRUM_FLOOR).sqrt()

    shift = (
        (backdoor_projected.mean() - clean_projected.mean()) / pooled_deviation
    ).item()
    return shift


def shrunk_covariance(features: torch.Tensor) -> torch.Tensor:
    """Sample covariance pulled toward a scaled identity, (dim, dim).

    original form
        Sigma_shrunk = (1 - alpha) * Sigma_sample + alpha * (trace(Sigma) / dim) * I

    Without the shrinkage term the inverse does not exist whenever the sample
    count is below the feature dimension, which is the normal case here.
    """
    centered = features - features.mean(dim=0, keepdim=True)
    sample_count = centered.shape[0]
    if sample_count < 2:
        raise ValueError(f"covariance needs at least 2 samples, got {sample_count}")

    covariance = (centered.T @ centered) / (sample_count - 1)
    dimension = covariance.shape[0]
    average_variance = torch.diagonal(covariance).mean()
    identity = torch.eye(dimension, dtype=covariance.dtype)

    shrunk = (
        1.0 - COVARIANCE_SHRINKAGE
    ) * covariance + COVARIANCE_SHRINKAGE * average_variance * identity
    return shrunk


def mahalanobis_distances(
    reference_features: torch.Tensor, query_features: torch.Tensor
) -> torch.Tensor:
    """Distance of every query sample from the reference distribution, (num_query,).

    original form
        d(x) = sqrt((x - mu)^T Sigma^-1 (x - mu))

    The reference is normally the clean population, so this asks how unusual each
    triggered image looks under the model's own notion of clean variation. Unlike
    a raw euclidean distance it accounts for the fact that some directions in
    feature space vary a lot anyway.

    The shrinkage is not neutral, and the direction of its bias matters here.
    Replacing eigenvalue lam with 0.9 lam + 0.1 mean_lam leaves a wide direction
    almost untouched and inflates a narrow direction toward the mean, so the same 3
    standard deviation move reads about 3.96 along the widest clean direction and
    about 0.84 along the narrowest. A trigger writes off the clean manifold, which
    is the narrow case, so the shrinkage works against the signal. It is kept
    because without it the covariance is singular whenever n is below dim, which
    is the normal case, but the number should be read as conservative rather than
    as calibrated. Comparisons between 2 populations measured against the same
    reference remain sound.

    Returns nan for every query when the reference population has no spread at
    all, because the quantity is undefined there rather than merely large.
    """
    reference = reference_features.double()
    query = query_features.double()

    if reference.shape[0] < 2:
        # A single row defines no distribution to measure against. A layer this
        # small must not take the whole table down with it, so it reports absent.
        return torch.full((query.shape[0],), float("nan"))

    covariance = shrunk_covariance(reference)
    if torch.diagonal(covariance).mean() <= SPECTRUM_FLOOR:
        # A reference population with no spread is a single point, and distance
        # "in units of its variation" is then undefined rather than large. ViT's
        # layer 0 under the cls reduction is exactly this case: the class token
        # enters the stack as a learned constant, identical for every image.
        return torch.full((query.shape[0],), float("nan"))

    centered = query - reference.mean(dim=0, keepdim=True)

    # solve is used rather than an explicit inverse because it is both more
    # accurate and cheaper for the same result.
    solved = torch.linalg.solve(covariance, centered.T).T
    squared = (centered * solved).sum(dim=1).clamp_min(0.0)

    distances = squared.sqrt().float()
    return distances


def effective_rank(features: torch.Tensor) -> float:
    """How many directions the population occupies, by participation ratio.

    original form
        participation ratio = (sum of eigenvalues)^2 / sum of (eigenvalues^2)

    3 other statistics also answer to "effective rank" in the literature: Roy and
    Vetterli's entropy of the normalized singular values, Kumar et al.'s
    thresholded count and the stable rank, which divides by the largest eigenvalue.
    This is none of them, so a write-up has to give the formula on first use. The
    participation ratio is used because it is a smooth function of the whole
    spectrum, needs no threshold and has a closed-form sample bias in n and dim.

    A value near 1 means the population lies on a single direction, which is the
    sharp form of "does the trigger collapse the representation", a question raw
    variance cannot answer because a collapse can raise total variance while
    removing directions.

    Read it as a relative quantity, never as a count of directions. A sample
    covariance from n samples in dim dimensions is biased downward and tracks the
    Marchenko-Pastur prediction dim / (1 + dim / n), so an isotropic population at
    dim 768 and n 1000 reads about 434 rather than 768. The bias depends on n and
    dim alone, so 2 populations of the same width at the same n are comparable to
    each other, which is the comparison rank_ratio makes and why rank_ratio is the
    number to quote.
    """
    centered = features.double() - features.double().mean(dim=0, keepdim=True)
    sample_count = centered.shape[0]
    if sample_count < 2:
        return float("nan")

    # Singular values of the centered matrix are the square roots of the
    # covariance eigenvalues scaled by the sample count, and taking them directly
    # avoids forming a (dim, dim) matrix for a wide feature space.
    singular_values = torch.linalg.svdvals(centered)
    eigenvalues = singular_values.pow(2) / (sample_count - 1)
    eigenvalues = eigenvalues[eigenvalues > eigenvalues.max() * SPECTRUM_FLOOR]

    total = eigenvalues.sum()
    rank = (total.pow(2) / eigenvalues.pow(2).sum()).item()
    return rank


# Neighbours used by the LID estimator. Ma et al. use 20 on CIFAR-scale data and
# report the estimate is stable across a wide range around it.
LID_NEIGHBOURS = 20


def local_intrinsic_dimensionality(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    num_neighbours: int = LID_NEIGHBOURS,
) -> torch.Tensor:
    """Per-sample local intrinsic dimensionality, (num_query,).

    original form, the Levina and Bickel maximum likelihood estimator as used by
    Ma et al. (ICLR 2018)

        LID(x) = - ( (1/k) * sum over i of log( r_i(x) / r_k(x) ) )^-1

    where r_i(x) is the distance from x to its i-th nearest neighbour in the
    reference set and r_k(x) is the furthest of the k.

    Measured because the 2 nearest opposite results in the literature use it. Ma
    et al. find adversarial inputs at higher LID than clean inputs, and COLLIDER
    filters backdoor training data on the premise that clean samples have low
    LID. Both say corrupted inputs are locally higher dimensional, the opposite
    sign to the rank ratio collapse measured here.

    The 2 do not contradict each other. LID is a local neighbourhood expansion
    rate at a single point, while the participation ratio is a global second
    moment property of a population, so a sample can sit in a locally sparse
    region while its population occupies few directions. Measuring both on the
    same features turns that from an argument into a result.

    A query drawn from the reference set is its own nearest neighbour at distance
    0, which would make the log diverge, so 1 extra neighbour is taken and the
    first dropped. That is a no-op for a genuinely disjoint reference set beyond
    costing 1 neighbour.
    """
    if reference_features.shape[0] <= num_neighbours + 1:
        raise ValueError(
            f"LID needs more than {num_neighbours + 1} reference samples, got "
            f"{reference_features.shape[0]}"
        )

    distances = torch.cdist(query_features.float(), reference_features.float())
    nearest = distances.topk(num_neighbours + 1, largest=False).values[:, 1:]

    furthest = nearest[:, -1:].clamp_min(SPECTRUM_FLOOR)
    ratios = (nearest / furthest).clamp_min(SPECTRUM_FLOOR)

    lid = -1.0 / ratios.log().mean(dim=1).clamp_max(-SPECTRUM_FLOOR)
    return lid


def class_centroids(
    features: torch.Tensor, labels: torch.Tensor
) -> dict[int, torch.Tensor]:
    """Mean feature per class label, for the classes present in this sample."""
    centroids = {
        int(label): features[labels == label].mean(dim=0) for label in labels.unique()
    }
    return centroids


def target_class_alignment(
    clean_features: torch.Tensor,
    backdoor_features: torch.Tensor,
    clean_labels: torch.Tensor,
    target_label: int,
) -> float:
    """Cosine between the trigger's shift and the direction of the target class.

    The backdoor's stated job is to move a representation into the target class
    region. This measures whether it actually does that, rather than moving the
    representation somewhere merely unusual: 1.0 means the trigger pushes exactly
    along the clean target-class direction, and 0.0 means it pushes somewhere
    orthogonal to it while still producing the target prediction.

    Returns nan when the target class is absent from the sample, since the
    reference direction is then undefined.
    """
    centroids = class_centroids(clean_features, clean_labels)
    if target_label not in centroids:
        return float("nan")

    trigger_shift = backdoor_features.mean(dim=0) - clean_features.mean(dim=0)
    target_shift = centroids[target_label] - clean_features.mean(dim=0)

    alignment = torch.nn.functional.cosine_similarity(
        trigger_shift.unsqueeze(0), target_shift.unsqueeze(0)
    ).item()
    return alignment


def has_spread(features: torch.Tensor) -> bool:
    """Whether a population varies at all, which several statistics require.

    A population with no spread is a single repeated point. Similarity and rank
    are undefined on it, and a routine that returns 0 instead of saying so turns
    a degenerate layer into a fabricated collapse, which is worse than a gap.
    """
    spread = features.float().var(dim=0, unbiased=False).max().item() > SPECTRUM_FLOOR
    return spread


# Folds used when a statistic needs a direction fitted off the samples it scores.
# 2 is the cheapest split that leaves every sample scored exactly once.
CROSSFIT_FOLDS = 2


def crossfit_projection_scores(
    clean_features: torch.Tensor,
    backdoor_features: torch.Tensor,
    folds: int = CROSSFIT_FOLDS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project both populations onto a direction fitted without the scored samples.

    The backdoor direction is the mean paired difference, so fitting it on the
    same samples a separation statistic then scores is circular: the direction is
    chosen to maximize exactly the gap being measured. 2 populations drawn from a
    single distribution then report an in-sample AUROC well above 0.5, and the
    floor scales with sqrt(dim / num_samples). Swin's width grows from 96 to 768
    across its stack, so an in-sample statistic plotted against depth produces a
    rising curve out of pure noise, in the shape a backdoor appearing with depth
    would produce.

    Cross-fitting removes it. Each fold's samples are scored by a direction
    estimated from the other folds only, so under the null the expected projection
    is 0 and the statistic sits at its true value. A real effect is unaffected.

    The split is on the pair index, because row i of each population is the same
    image, and splitting them independently would break the pairing the direction
    is estimated from.
    """
    if clean_features.shape != backdoor_features.shape:
        raise ValueError(
            "cross-fitting needs index-aligned populations, got "
            f"{tuple(clean_features.shape)} and {tuple(backdoor_features.shape)}"
        )

    sample_count = clean_features.shape[0]
    if sample_count < 2 * folds:
        # Too few samples to hold any out, so there is no honest estimate to give.
        nan_scores = torch.full((sample_count,), float("nan"))
        return nan_scores, nan_scores.clone()

    # A deterministic interleaved split, so the result does not depend on a seed
    # and neighbouring rows never land in the same fold.
    fold_of_sample = torch.arange(sample_count) % folds

    clean_scores = torch.empty(sample_count, dtype=torch.float32)
    backdoor_scores = torch.empty(sample_count, dtype=torch.float32)
    for fold in range(folds):
        held_out = fold_of_sample == fold
        fitted_on = ~held_out
        direction = backdoor_direction(
            clean_features[fitted_on], backdoor_features[fitted_on]
        )
        clean_scores[held_out] = project_onto_direction(
            clean_features[held_out], direction
        ).float()
        backdoor_scores[held_out] = project_onto_direction(
            backdoor_features[held_out], direction
        ).float()

    return clean_scores, backdoor_scores


def standardized_shift_from_scores(
    clean_scores: torch.Tensor, backdoor_scores: torch.Tensor
) -> float:
    """Cohen's d between 2 already-projected populations.

        original form
            d = (mean_1 - mean_2) / pooled_standard_deviation

    Takes scores rather than features so the caller controls whether the
    projection direction was fitted on these samples or held out from them.
    """
    clean_count = len(clean_scores)
    backdoor_count = len(backdoor_scores)
    if clean_count < 2 or backdoor_count < 2:
        return float("nan")

    pooled_variance = (
        (clean_count - 1) * clean_scores.var(unbiased=True)
        + (backdoor_count - 1) * backdoor_scores.var(unbiased=True)
    ) / (clean_count + backdoor_count - 2)
    pooled_deviation = pooled_variance.clamp_min(SPECTRUM_FLOOR).sqrt()

    shift = ((backdoor_scores.mean() - clean_scores.mean()) / pooled_deviation).item()
    return shift


def layer_distribution_row(
    clean_features: torch.Tensor,
    backdoor_features: torch.Tensor,
    clean_labels: torch.Tensor | None = None,
    target_label: int | None = None,
) -> dict[str, float]:
    """Every distributional statistic for a layer, as a flat dict.

    Kept flat and JSON-friendly so a list of these goes straight into a DataFrame,
    a table or a pinned test fixture.
    """
    direction = backdoor_direction(clean_features, backdoor_features)

    # Fitted off the samples they score. See crossfit_projection_scores for why
    # the in-sample form is not usable: its null floor rises with dim / n, which
    # on Swin turns the stage widths into an apparent depth trend.
    clean_projected, backdoor_projected = crossfit_projection_scores(
        clean_features, backdoor_features
    )

    # ViT's layer 0 under the cls reduction is the class token before any block
    # has run, a learned constant identical for every image. Similarity and rank
    # have no meaning there, so they are reported absent rather than as 0, which
    # an aggregation would otherwise read as a total collapse.
    comparable = has_spread(clean_features) and has_spread(backdoor_features)

    clean_rank = effective_rank(clean_features)
    backdoor_rank = effective_rank(backdoor_features)

    row = {
        "direction_norm": direction.norm().item(),
        "separation_auroc": separation_auroc(clean_projected, backdoor_projected),
        "standardized_shift": standardized_shift_from_scores(
            clean_projected, backdoor_projected
        ),
        "cka": debiased_linear_cka(clean_features, backdoor_features)
        if comparable
        else float("nan"),
        "effective_rank_clean": clean_rank,
        "effective_rank_backdoor": backdoor_rank,
        # Below 1 means the trigger confines the representation to fewer
        # directions than clean data occupies, which is the collapse a backdoor
        # produces when it overwrites the representation rather than adding to it.
        "rank_ratio": backdoor_rank / clean_rank if clean_rank else float("nan"),
        "mahalanobis_median": mahalanobis_distances(clean_features, backdoor_features)
        .median()
        .item(),
    }

    # Both populations are scored against the CLEAN reference, so the comparison
    # is "how locally sparse does each look inside clean data", which is the
    # question Ma et al. and COLLIDER answer with the opposite sign.
    if comparable and clean_features.shape[0] > LID_NEIGHBOURS + 1:
        clean_lid = local_intrinsic_dimensionality(clean_features, clean_features)
        backdoor_lid = local_intrinsic_dimensionality(backdoor_features, clean_features)
        row["lid_clean"] = clean_lid.median().item()
        row["lid_backdoor"] = backdoor_lid.median().item()
        row["lid_ratio"] = row["lid_backdoor"] / row["lid_clean"]
    else:
        row["lid_clean"] = float("nan")
        row["lid_backdoor"] = float("nan")
        row["lid_ratio"] = float("nan")

    if clean_labels is not None and target_label is not None:
        row["target_alignment"] = (
            target_class_alignment(
                clean_features, backdoor_features, clean_labels, target_label
            )
            if comparable
            else float("nan")
        )

    return row


def layer_distribution_table(
    clean_features: dict[int, torch.Tensor],
    backdoor_features: dict[int, torch.Tensor],
    clean_labels: torch.Tensor | None = None,
    target_label: int | None = None,
) -> list[dict[str, float]]:
    """A row per layer, in layer order, each carrying a "layer" key.

    Feed the result to pandas.DataFrame to get the table a notebook wants.
    """
    shared_layers = sorted(set(clean_features) & set(backdoor_features))
    table = [
        {
            "layer": layer,
            **layer_distribution_row(
                clean_features[layer],
                backdoor_features[layer],
                clean_labels,
                target_label,
            ),
        }
        for layer in shared_layers
    ]
    return table
