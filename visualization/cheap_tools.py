"""Registry of the cheap visualisation tools, each a (case, args) -> None callable.

A tool reads what it needs from the loaded case (model, loaders, metadata,
device, dataset statistics) and the parsed arguments (layer, view, samples,
classes, seed, batch size, precision), writes its figure as PDF and PNG under
case.out_dir with a JSON sidecar of every number it plotted, a CSV where
BackdoorBench writes one, and returns nothing. cli.visualize merges this
registry with expensive_tools.TOOLS.

15 tools mirror the cheap half of BackdoorBench's analysis module at commit
f02e353, statistic for statistic, over this project's paired PSBD splits. The
tool table, each tool's ViT mapping and every deviation from the upstream code
are in docs/visualization/README.md.
"""

import argparse
import csv
import math
import os
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from analysis.attribution import (
    class_activation_map,
    expected_gradients,
    frequency_saliency,
)
from analysis.direction import outlier_dimensions
from analysis.embedding import pca_project, tsne_project, umap_project, unit_square
from analysis.features import default_reduction, extract_layer_features
from analysis.lipschitz import mlp_output_channel_lipschitz
from analysis.neurons import (
    block_count,
    class_purity,
    layer_activations,
    mean_activation,
    paired_layer_tac,
    top_activating_indices,
    top_k_rule,
)
from analysis.samples import (
    SampleView,
    clean_view,
    paired_examples,
    paired_views,
    select_classes,
    to_display,
    to_pixels,
    view_samples,
)
from analysis.stealth import compute_stealth_metrics
from defences.inference import forward_probs
from detectors.sentinet import MASK_THRESHOLD, saliency_mask

from evaluation.metrics import confusion_matrix

from .bars import (
    class_purity_bars,
    metric_radar,
    overlaid_activation_bars,
    stealth_bars,
)
from .heatmaps import TOKEN_MAP_GRID, layer_by_dimension_heatmap, token_map_grid
from .matrices import confusion_matrix_figure
from .panels import (
    activating_image_grid,
    attribution_rows,
    overlay_heat_map,
    paired_image_panels,
)
from .scatter import embedding_scatter
from .sidecar import write_sidecar
from .style import (
    LIPSCHITZ_CMAP,
    POISON_COLOUR,
    TAC_CMAP,
    class_colour,
    colour_table,
    save_figure,
)

# Upstream's --normalize_by_layer for the 2 heatmaps, off by default as there.
NORMALIZE_BY_LAYER = False

# The TAC convention the heatmap draws, BackdoorBench's L1 form. The
# trigger_activation_change docstring names the other.
TAC_NORM = "l1_sum"

# The mixed view's poison ratio is the checkpoint's own poison rate, as
# upstream's pratio is. A benign checkpoint records 0, which would make the
# mixed view the clean view under another name, so it gets this floor.
MIXED_RATIO_FLOOR = 0.1

# The image panels' layout, 2 clean then 2 poisoned, as every upstream panel.
PANEL_CLEAN = 2
PANEL_POISONED = 2

# visual_act.py's num_neuron and num_image caps, and visual_shap.py's background size.
TOP_DIMENSIONS_FOR_IMAGES = 16
IMAGES_PER_DIMENSION = 8
EXPECTED_GRADIENT_BACKGROUND = 200

# Marker size and opacity of the embedding scatters, 1 setting for all 3 so
# they read alike.
EMBEDDING_MARK_SIZE = 6.0
EMBEDDING_ALPHA = 0.7


def _use_bfloat16(args: argparse.Namespace) -> bool:
    flag = not args.no_bfloat16
    return flag


def _target_label(case) -> int:
    label = int(case.manifest["probe_target_label"])
    return label


def _layer(case, args: argparse.Namespace) -> int:
    """The block index a per-layer tool reads, the last block when none was asked for."""
    blocks = block_count(case.model, case.architecture)
    layer = blocks if args.layer is None else int(args.layer)
    if not 0 <= layer <= blocks:
        raise ValueError(f"layer {layer} is outside 0 to {blocks}")
    return layer


def _mixed_ratio(case) -> float:
    ratio = max(float(case.metadata.get("poison_rate") or 0.0), MIXED_RATIO_FLOOR)
    return ratio


def _class_order(case, args: argparse.Namespace) -> np.ndarray:
    order = select_classes(
        case.num_classes, args.classes, _target_label(case), args.seed
    )
    return order


def _view(case, args: argparse.Namespace) -> SampleView:
    view = view_samples(
        case.loaders,
        case.manifest,
        args.view,
        args.samples,
        _class_order(case, args),
        _mixed_ratio(case),
        args.seed,
    )
    return view


def _settings(case, args: argparse.Namespace, **extra) -> dict:
    """What the tool ran with, for the sidecar."""
    settings = {
        "layer": args.layer,
        "view": args.view,
        "samples": args.samples,
        "classes": args.classes,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "bfloat16": _use_bfloat16(args),
        "probe_attack": case.manifest["probe_attack"],
        "probe_target_label": _target_label(case),
        **extra,
    }
    return settings


def _figure_path(case, tool: str) -> str:
    path = os.path.join(case.out_dir, f"{tool}.pdf")
    return path


def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    """1 CSV with a header row, the format upstream's pandas to_csv writes."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _listed(values) -> list:
    """A tensor or array as nested lists, for the JSON sidecar."""
    array = (
        values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    )
    listed = array.tolist()
    return listed


@torch.inference_mode()
def _probabilities(
    case, args: argparse.Namespace, images: torch.Tensor
) -> torch.Tensor:
    """Softmax over the images in batches, (N, num_classes) float32 on the CPU."""
    chunks = [
        forward_probs(case.model, batch, case.device, _use_bfloat16(args)).cpu()
        for batch in images.split(args.batch_size)
    ]
    probs = torch.cat(chunks)  # (N, num_classes)
    return probs


def _pooled_features(
    case, args: argparse.Namespace, images: torch.Tensor, layer: int
) -> torch.Tensor:
    """The head's own token at 1 layer for every image, (N, dim), the embedding tools' input."""
    loader = DataLoader(
        TensorDataset(images, torch.zeros(images.shape[0], dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    by_layer = extract_layer_features(
        case.model,
        loader,
        case.device,
        _use_bfloat16(args),
        reduction=default_reduction(case.architecture),
        architecture=case.architecture,
    )
    features = by_layer[layer]  # (N, dim)
    return features


def _layer_tac(case, args: argparse.Namespace, layer: int) -> torch.Tensor:
    """TAC at 1 layer over the paired views, (dim,), the ranking several tools share."""
    clean, backdoor = paired_views(case.loaders, case.manifest)
    tac = paired_layer_tac(
        case.model,
        clean.images,
        backdoor.images,
        (layer,),
        case.device,
        args.batch_size,
        _use_bfloat16(args),
        TAC_NORM,
        case.architecture,
    )[layer]
    return tac


def run_tac(case, args: argparse.Namespace) -> None:
    """visual_tac.py: TAC of every residual dimension at every block, as 1 heatmap."""
    clean, backdoor = paired_views(case.loaders, case.manifest)
    layers = tuple(range(1, block_count(case.model, case.architecture) + 1))
    tac = paired_layer_tac(
        case.model,
        clean.images,
        backdoor.images,
        layers,
        case.device,
        args.batch_size,
        _use_bfloat16(args),
        TAC_NORM,
        case.architecture,
    )
    matrix = np.stack([tac[layer].numpy() for layer in layers])  # (num_layers, dim)
    outliers = {layer: _listed(outlier_dimensions(tac[layer])) for layer in layers}

    figure = layer_by_dimension_heatmap(
        matrix,
        [str(layer) for layer in layers],
        TAC_CMAP,
        "TAC",
        f"TAC of {len(clean)} image pairs, max {matrix.max():.1f}, {case.folder}",
        NORMALIZE_BY_LAYER,
    )
    path = _figure_path(case, "tac")
    save_figure(figure, path)
    _write_csv(
        os.path.join(case.out_dir, "tac.csv"),
        ["layer", "neuron", "tac"],
        [
            [layer, neuron, float(matrix[row, neuron])]
            for row, layer in enumerate(layers)
            for neuron in range(matrix.shape[1])
        ],
    )
    write_sidecar(
        path,
        "tac",
        case.folder,
        _settings(
            case,
            args,
            view="paired",
            norm=TAC_NORM,
            normalize_by_layer=NORMALIZE_BY_LAYER,
            pairs=len(clean),
        ),
        {
            "layers": list(layers),
            "tac": _listed(matrix),
            "outlier_dimensions": outliers,
        },
    )


def run_lipschitz(case, args: argparse.Namespace) -> None:
    """visual_lips.py: the per-dimension Lipschitz bound of every block's MLP write, as 1 heatmap."""
    by_layer = mlp_output_channel_lipschitz(case.model)
    layers = sorted(by_layer)
    matrix = np.stack(
        [by_layer[layer].numpy() for layer in layers]
    )  # (num_layers, dim)

    figure = layer_by_dimension_heatmap(
        matrix,
        [str(layer) for layer in layers],
        LIPSCHITZ_CMAP,
        "Lipschitz",
        f"MLP output Lipschitz per dimension, max {matrix.max():.2f}, {case.folder}",
        NORMALIZE_BY_LAYER,
    )
    path = _figure_path(case, "lipschitz")
    save_figure(figure, path)
    _write_csv(
        os.path.join(case.out_dir, "lipschitz.csv"),
        ["layer", "neuron", "lips"],
        [
            [layer, neuron, float(matrix[row, neuron])]
            for row, layer in enumerate(layers)
            for neuron in range(matrix.shape[1])
        ],
    )
    write_sidecar(
        path,
        "lipschitz",
        case.folder,
        _settings(case, args, normalize_by_layer=NORMALIZE_BY_LAYER, data_free=True),
        {"layers": layers, "lipschitz": _listed(matrix)},
    )


def run_neuron_activation(case, args: argparse.Namespace) -> None:
    """visual_na.py: mean token-sum activation per dimension, clean against triggered, overlaid."""
    layer = _layer(case, args)
    clean, backdoor = paired_views(case.loaders, case.manifest)
    clean_features = layer_activations(
        case.model,
        clean.images,
        layer,
        case.device,
        args.batch_size,
        _use_bfloat16(args),
        "sum",
        case.architecture,
    )  # (N, dim)
    triggered_features = layer_activations(
        case.model,
        backdoor.images,
        layer,
        case.device,
        args.batch_size,
        _use_bfloat16(args),
        "sum",
        case.architecture,
    )  # (N, dim)
    clean_mean = mean_activation(clean_features).numpy()  # (dim,)
    triggered_mean = mean_activation(triggered_features).numpy()  # (dim,)

    figure = overlaid_activation_bars(
        clean_mean,
        triggered_mean,
        f"layer {layer}, {len(clean)} image pairs, {case.folder}",
    )
    path = _figure_path(case, "neuron_activation")
    save_figure(figure, path)
    write_sidecar(
        path,
        "neuron_activation",
        case.folder,
        _settings(
            case,
            args,
            view="paired",
            layer=layer,
            reduction="token_sum",
            pairs=len(clean),
        ),
        {
            "clean_mean": _listed(clean_mean),
            "triggered_mean": _listed(triggered_mean),
            "sorted_dimensions": _listed(np.argsort(clean_mean)[::-1]),
        },
    )


def run_activation_distribution(case, args: argparse.Namespace) -> None:
    """visual_actdist.py: the class shares of each dimension's top images, poisoned in black."""
    layer = _layer(case, args)
    view = _view(case, args)
    features = layer_activations(
        case.model,
        view.images,
        layer,
        case.device,
        args.batch_size,
        _use_bfloat16(args),
        "sum",
        case.architecture,
    )  # (N, dim)

    class_order = _class_order(case, args)
    num_poisoned = int(view.poison_mask.sum())
    k = top_k_rule(len(view), len(class_order), num_poisoned)
    top = top_activating_indices(features, k)  # (k, dim)
    purity = class_purity(top, view.labels, view.poison_mask, case.num_classes).numpy()

    # Only the classes present get a column and a legend entry, as upstream's
    # label_set, and the poisoned pseudo class comes last.
    present = sorted(int(c) for c in torch.unique(view.labels[~view.poison_mask]))
    columns = present + ([case.num_classes] if num_poisoned else [])
    shares = purity[:, columns]  # (dim, len(columns))
    table = colour_table(class_order, case.num_classes)  # (num_classes + 1, 3)
    poison_share = purity[:, case.num_classes]
    row_order = (
        np.argsort(-poison_share, kind="stable")
        if num_poisoned
        else np.arange(purity.shape[0])
    )

    legend_labels = [f"class {c}" for c in present] + (
        ["poisoned"] if num_poisoned else []
    )
    legend_colours = [
        class_colour(int(np.flatnonzero(class_order == c)[0])) for c in present
    ] + ([POISON_COLOUR] if num_poisoned else [])
    figure = class_purity_bars(
        shares[row_order],
        table[columns],
        legend_labels,
        legend_colours,
        f"top-{k} images per dimension, layer {layer}, {args.view}",
    )
    path = _figure_path(case, "activation_distribution")
    save_figure(figure, path)
    _write_csv(
        os.path.join(case.out_dir, "activation_distribution.csv"),
        ["neuron"]
        + [f"class_{c}" for c in present]
        + (["poisoned"] if num_poisoned else []),
        [[int(d)] + [float(v) for v in shares[d]] for d in range(shares.shape[0])],
    )
    write_sidecar(
        path,
        "activation_distribution",
        case.folder,
        _settings(
            case,
            args,
            layer=layer,
            top_k=k,
            reduction="token_sum",
            rows=len(view),
            poisoned_rows=num_poisoned,
            sorted_by_poison_share=bool(num_poisoned),
        ),
        {
            "columns": columns,
            "row_order": _listed(row_order),
            "purity": _listed(shares[row_order]),
        },
    )


def run_activating_images(case, args: argparse.Namespace) -> None:
    """visual_act.py: the top images of the strongest TAC dimensions, red titles on poisoned ones."""
    layer = _layer(case, args)
    view = _view(case, args)
    features = layer_activations(
        case.model,
        view.images,
        layer,
        case.device,
        args.batch_size,
        _use_bfloat16(args),
        "sum",
        case.architecture,
    )  # (N, dim)
    k = min(IMAGES_PER_DIMENSION, len(view))
    top = top_activating_indices(features, k)  # (k, dim)

    tac = _layer_tac(case, args, layer)  # (dim,)
    dimensions = torch.argsort(tac, descending=True)[:TOP_DIMENSIONS_FOR_IMAGES]
    rows = top[:, dimensions].T  # (num_dims, k)
    pixels = to_display(
        to_pixels(view.images[rows.reshape(-1)], case.mean, case.std)
    ).reshape(
        rows.shape[0], k, *view.images.shape[2:], view.images.shape[1]
    )  # (num_dims, k, H, W, C)
    values = features[rows, dimensions[:, None]].numpy()  # (num_dims, k)
    poisoned = view.poison_mask[rows].numpy()  # (num_dims, k)

    figure = activating_image_grid(
        pixels,
        dimensions.numpy(),
        values,
        poisoned,
        f"layer {layer}, top {k} images of the top {len(dimensions)} TAC dimensions, {args.view}",
    )
    path = _figure_path(case, "activating_images")
    save_figure(figure, path)
    write_sidecar(
        path,
        "activating_images",
        case.folder,
        _settings(
            case,
            args,
            layer=layer,
            images_per_dimension=k,
            dimension_rule="top_tac",
            rows=len(view),
        ),
        {
            "dimensions": _listed(dimensions),
            "tac": _listed(tac[dimensions]),
            "rows": _listed(rows),
            "test_indices": _listed(view.test_indices[rows]),
            "values": _listed(values),
            "poisoned": _listed(poisoned),
        },
    )


def _run_embedding(
    case,
    args: argparse.Namespace,
    tool: str,
    project: Callable,
    axis_names: tuple[str, str],
) -> None:
    """Shared body of the 3 embedding tools: features at the layer, a projection, the unit square."""
    layer = _layer(case, args)
    view = _view(case, args)
    features = _pooled_features(case, args, view.images, layer)  # (N, dim)
    embedding = unit_square(project(features))  # (N, 2)

    class_order = _class_order(case, args)
    figure = embedding_scatter(
        embedding,
        view.labels.numpy(),
        view.poison_mask.numpy(),
        class_order,
        f"{tool} of layer {layer}, {args.view}, {case.folder}",
        axis_names,
        EMBEDDING_MARK_SIZE,
        EMBEDDING_ALPHA,
    )
    path = _figure_path(case, tool)
    save_figure(figure, path)
    write_sidecar(
        path,
        tool,
        case.folder,
        _settings(
            case,
            args,
            layer=layer,
            reduction=default_reduction(case.architecture),
            rows=len(view),
        ),
        {
            "embedding": _listed(embedding),
            "labels": _listed(view.labels),
            "poison_mask": _listed(view.poison_mask),
            "test_indices": _listed(view.test_indices),
            "class_order": _listed(class_order),
        },
    )


def run_tsne(case, args: argparse.Namespace) -> None:
    """visual_tsne.py: t-SNE of the head's token at the layer, coloured by class, poisoned in black."""
    _run_embedding(
        case, args, "tsne", lambda f: tsne_project(f, args.seed), ("dim 1", "dim 2")
    )


def run_umap(case, args: argparse.Namespace) -> None:
    """visual_umap.py: UMAP of the same features."""
    _run_embedding(
        case,
        args,
        "umap",
        lambda f: umap_project(f, seed=args.seed),
        ("dim 1", "dim 2"),
    )


def run_pca(case, args: argparse.Namespace) -> None:
    """The linear view beside the 2 nonlinear ones, no upstream counterpart."""
    _run_embedding(case, args, "pca", lambda f: pca_project(f, 2), ("PC 1", "PC 2"))


def run_confusion(case, args: argparse.Namespace) -> None:
    """visual_cm.py: the row-normalised confusion matrix of the view, true class down."""
    view = _view(case, args)
    predictions = _probabilities(case, args, view.images).argmax(dim=1)  # (N,)
    matrix = confusion_matrix(
        predictions, view.labels, case.num_classes, normalise=True
    )
    accuracy = float((predictions == view.labels).float().mean())

    figure = confusion_matrix_figure(
        matrix.numpy(), True, f"{args.view}, accuracy {accuracy:.3f}, {case.folder}"
    )
    path = _figure_path(case, "confusion")
    save_figure(figure, path)
    write_sidecar(
        path,
        "confusion",
        case.folder,
        _settings(
            case, args, normalised=True, rows=len(view), label_convention="true_class"
        ),
        {"matrix": _listed(matrix), "accuracy": accuracy},
    )


def run_token_maps(case, args: argparse.Namespace) -> None:
    """visual_fm.py's analogue: the token grid of the top TAC dimensions for 1 image, clean and triggered."""
    layer = _layer(case, args)
    clean, backdoor = paired_views(case.loaders, case.manifest)
    tac = _layer_tac(case, args, layer)  # (dim,)
    dimensions = torch.argsort(tac, descending=True)[: TOKEN_MAP_GRID**2]

    row = int(np.random.default_rng(args.seed).integers(len(backdoor)))
    pair = torch.stack([clean.images[row], backdoor.images[row]])  # (2, C, H, W)
    tokens = layer_activations(
        case.model,
        pair,
        layer,
        case.device,
        args.batch_size,
        _use_bfloat16(args),
        "none",
        case.architecture,
    )  # (2, tokens, dim)
    first_patch = 1 if case.architecture == "vit" else 0
    patches = tokens[:, first_patch:, :][:, :, dimensions]  # (2, patches, num_dims)
    grid = math.isqrt(patches.shape[1])
    if grid * grid != patches.shape[1]:
        raise ValueError(f"{patches.shape[1]} patch tokens do not form a square grid")
    maps = patches.permute(0, 2, 1).reshape(2, len(dimensions), grid, grid).numpy()

    figure = token_map_grid(
        maps[0],
        maps[1],
        dimensions.numpy(),
        tac[dimensions].numpy(),
        f"layer {layer}, test image {int(backdoor.test_indices[row])}, class {int(clean.labels[row])}, {case.folder}",
    )
    path = _figure_path(case, "token_maps")
    save_figure(figure, path)
    write_sidecar(
        path,
        "token_maps",
        case.folder,
        _settings(
            case,
            args,
            layer=layer,
            test_index=int(backdoor.test_indices[row]),
            grid=grid,
            dimension_rule="top_tac",
        ),
        {
            "dimensions": _listed(dimensions),
            "tac": _listed(tac[dimensions]),
            "clean_maps": _listed(maps[0]),
            "triggered_maps": _listed(maps[1]),
        },
    )


def _examples(case, args: argparse.Namespace) -> SampleView:
    """The 2 clean and 2 poisoned rows of the image panels, the same images twice."""
    clean, backdoor = paired_views(case.loaders, case.manifest)
    examples = paired_examples(clean, backdoor, PANEL_CLEAN, PANEL_POISONED, args.seed)
    return examples


def _panel_titles(
    examples: SampleView, probs: torch.Tensor
) -> tuple[list[str], list[str]]:
    """(image titles, prediction titles) read from the poison mask and the softmax."""
    confidence, predicted = probs.max(dim=1)  # (N,), (N,)
    image_titles = [
        f"{'poisoned' if bool(poisoned) else 'clean'}: class {int(label)}"
        for label, poisoned in zip(examples.labels, examples.poison_mask)
    ]
    prediction_titles = [
        f"predicted {int(label)}, {100 * float(p):.1f}%"
        for label, p in zip(predicted, confidence)
    ]
    return image_titles, prediction_titles


def _trigger_coverage(examples: SampleView, cam: torch.Tensor, mean, std) -> list[dict]:
    """How much of each poisoned panel's trigger the thresholded Grad-CAM mask covers.

    The smoke of 2026-09-10 recorded 0 coverage at every block on the BadNet
    checkpoint, which is why SentiNet carries nothing on this ViT, so the
    panel records the same number beside its picture. The trigger pixels of a
    poisoned panel are the pixels that differ from the clean panel of the same
    test index, found by that index and never by position.
    """
    masks = saliency_mask(cam, tuple(cam.shape[1:]), MASK_THRESHOLD)[:, 0]  # (N, H, W)
    pixels = to_pixels(examples.images, mean, std)  # (N, C, H, W)
    clean_panel_of = {
        int(index): row
        for row, index in enumerate(examples.test_indices)
        if not bool(examples.poison_mask[row])
    }
    coverage = []
    for row in range(len(examples)):
        counterpart = clean_panel_of.get(int(examples.test_indices[row]))
        if not bool(examples.poison_mask[row]) or counterpart is None:
            continue
        trigger = (pixels[row] != pixels[counterpart]).any(dim=0)  # (H, W)
        covered = float((masks[row] & trigger).sum() / trigger.sum().clamp_min(1))
        coverage.append(
            {
                "panel": row,
                "trigger_pixels": int(trigger.sum()),
                "covered_share": covered,
                "mask_share": float(masks[row].float().mean()),
            }
        )
    return coverage


def run_gradcam(case, args: argparse.Namespace) -> None:
    """visual_gradcam.py: 2 clean and 2 poisoned images with the Grad-CAM of the predicted class overlaid."""
    examples = _examples(case, args)
    cam, _ = class_activation_map(
        case.model, examples.images, case.device, _use_bfloat16(args)
    )  # (4, H, W)
    probs = _probabilities(case, args, examples.images)  # (4, classes)
    pixels = to_display(to_pixels(examples.images, case.mean, case.std))  # (4, H, W, C)
    overlays = overlay_heat_map(pixels, cam.numpy())  # (4, H, W, 3)
    image_titles, prediction_titles = _panel_titles(examples, probs)
    coverage = _trigger_coverage(examples, cam, case.mean, case.std)

    figure = paired_image_panels(
        pixels, overlays, image_titles, prediction_titles, "rgb"
    )
    path = _figure_path(case, "gradcam")
    save_figure(figure, path)
    write_sidecar(
        path,
        "gradcam",
        case.folder,
        _settings(case, args, mask_threshold=MASK_THRESHOLD, overlay="jet_half_blend"),
        {
            "test_indices": _listed(examples.test_indices),
            "labels": _listed(examples.labels),
            "poison_mask": _listed(examples.poison_mask),
            "predicted": _listed(probs.argmax(dim=1)),
            "confidence": _listed(probs.max(dim=1).values),
            "cam": _listed(cam),
            "trigger_coverage": coverage,
        },
    )


def run_frequency(case, args: argparse.Namespace) -> None:
    """visual_fre.py: the same 4 images with the frequency saliency map of the top logit."""
    examples = _examples(case, args)
    maps = np.stack(
        [
            frequency_saliency(case.model, image, case.device, _use_bfloat16(args))
            for image in examples.images
        ]
    )  # (4, H, W) uint8
    probs = _probabilities(case, args, examples.images)  # (4, classes)
    pixels = to_display(to_pixels(examples.images, case.mean, case.std))  # (4, H, W, C)
    image_titles, prediction_titles = _panel_titles(examples, probs)

    figure = paired_image_panels(
        pixels, maps, image_titles, prediction_titles, "frequency"
    )
    path = _figure_path(case, "frequency")
    save_figure(figure, path)
    write_sidecar(
        path,
        "frequency",
        case.folder,
        _settings(case, args, transform="ifft2_log_magnitude_minmax"),
        {
            "test_indices": _listed(examples.test_indices),
            "labels": _listed(examples.labels),
            "poison_mask": _listed(examples.poison_mask),
            "predicted": _listed(probs.argmax(dim=1)),
            "maps": _listed(maps),
        },
    )


def run_expected_gradients(case, args: argparse.Namespace) -> None:
    """visual_shap.py: expected gradients of the top 2 classes for the 4 images, against a clean background."""
    examples = _examples(case, args)
    whole_clean = clean_view(case.loaders, case.manifest)
    drawn = np.random.default_rng(args.seed).choice(
        len(whole_clean),
        min(EXPECTED_GRADIENT_BACKGROUND, len(whole_clean)),
        replace=False,
    )
    background = whole_clean.images[torch.as_tensor(drawn)]  # (num_background, C, H, W)
    attributions, classes = expected_gradients(
        case.model,
        examples.images,
        background,
        case.device,
        _use_bfloat16(args),
        seed=args.seed,
        batch_size=args.batch_size,
    )  # (4, 2, C, H, W), (4, 2)
    summed = attributions.sum(dim=2).numpy()  # (4, 2, H, W)
    pixels = to_display(to_pixels(examples.images, case.mean, case.std))  # (4, H, W, C)

    image_titles = [
        f"{'poisoned' if bool(poisoned) else 'clean'}: class {int(label)}"
        for label, poisoned in zip(examples.labels, examples.poison_mask)
    ]
    class_titles = [[f"class {int(c)}" for c in row] for row in classes]
    figure = attribution_rows(pixels, summed, class_titles, image_titles)
    path = _figure_path(case, "expected_gradients")
    save_figure(figure, path)
    write_sidecar(
        path,
        "expected_gradients",
        case.folder,
        _settings(
            case,
            args,
            background_rows=int(background.shape[0]),
            background="clean_split",
            ranked_outputs=int(classes.shape[1]),
        ),
        {
            "test_indices": _listed(examples.test_indices),
            "labels": _listed(examples.labels),
            "poison_mask": _listed(examples.poison_mask),
            "classes": _listed(classes),
            "attributions_summed": _listed(summed),
        },
    )


def run_stealth(case, args: argparse.Namespace) -> None:
    """visual_quality.py: PSNR, SSIM and LPIPS between every image and its triggered self."""
    clean, backdoor = paired_views(case.loaders, case.manifest)
    metrics = compute_stealth_metrics(
        to_pixels(clean.images, case.mean, case.std),
        to_pixels(backdoor.images, case.mean, case.std),
        case.device,
        batch_size=args.batch_size,
        max_samples=args.samples,
        seed=args.seed,
    )

    figure = stealth_bars(metrics, case.manifest["probe_attack"])
    path = _figure_path(case, "stealth")
    save_figure(figure, path)
    _write_csv(
        os.path.join(case.out_dir, "stealth.csv"),
        [
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
            "lpips_mean",
            "lpips_std",
            "n_pairs",
        ],
        [
            [
                metrics[k]
                for k in (
                    "psnr_mean",
                    "psnr_std",
                    "ssim_mean",
                    "ssim_std",
                    "lpips_mean",
                    "lpips_std",
                    "n_pairs",
                )
            ]
        ],
    )
    write_sidecar(
        path, "stealth", case.folder, _settings(case, args, view="paired"), metrics
    )


def run_metrics(case, args: argparse.Namespace) -> None:
    """visual_metric.py: the C-ACC, 1 - ASR and RA radar, RA from a pass over the triggered rows."""
    clean, backdoor = paired_views(case.loaders, case.manifest)
    target = _target_label(case)
    triggered_predictions = _probabilities(case, args, backdoor.images).argmax(
        dim=1
    )  # (N,)
    robust_accuracy = float((triggered_predictions == backdoor.labels).float().mean())

    # C-ACC and ASR come from the checkpoint's own args.json when the trainer
    # recorded them, the pass here fills them in otherwise.
    clean_accuracy = case.metadata.get("clean_accuracy")
    if clean_accuracy is None:
        clean_predictions = _probabilities(case, args, clean.images).argmax(dim=1)
        clean_accuracy = float((clean_predictions == clean.labels).float().mean())
    asr = case.metadata.get("asr")
    if asr is None:
        asr = float((triggered_predictions == target).float().mean())

    names = ["C-ACC", "1 - ASR", "RA"]
    values = [float(clean_accuracy), 1.0 - float(asr), robust_accuracy]
    figure = metric_radar(names, values, f"metrics, {case.folder}")
    path = _figure_path(case, "metrics")
    save_figure(figure, path)
    _write_csv(
        os.path.join(case.out_dir, "metrics.csv"),
        ["clean_accuracy", "asr", "robust_accuracy"],
        [[float(clean_accuracy), float(asr), robust_accuracy]],
    )
    write_sidecar(
        path,
        "metrics",
        case.folder,
        _settings(
            case,
            args,
            view="paired",
            rows=len(backdoor),
            accuracy_source=case.metadata.get("clean_accuracy_source", "pass"),
        ),
        {
            "names": names,
            "values": values,
            "clean_accuracy": float(clean_accuracy),
            "asr": float(asr),
            "robust_accuracy": robust_accuracy,
        },
    )


TOOLS: dict[str, Callable] = {
    "tac": run_tac,
    "lipschitz": run_lipschitz,
    "neuron_activation": run_neuron_activation,
    "activation_distribution": run_activation_distribution,
    "activating_images": run_activating_images,
    "tsne": run_tsne,
    "umap": run_umap,
    "pca": run_pca,
    "confusion": run_confusion,
    "token_maps": run_token_maps,
    "gradcam": run_gradcam,
    "frequency": run_frequency,
    "expected_gradients": run_expected_gradients,
    "stealth": run_stealth,
    "metrics": run_metrics,
}
