"""Registry of the expensive visualisation tools, each a callable of (case, args).

Same contract as cheap_tools.TOOLS. These run minutes per checkpoint on an A100
(the Hessian spectrum, the loss landscape, feature visualisation) and are never
part of --all-cheap. Each writes <out_dir>/<tool>.pdf, .png and .json and
prints its wall time. The constants below are the cost knobs, sized for the
login node's 15 GB GPU budget and measured in docs/visualization/expensive-tools.md.

The view names the splits a tool reads. paired is the clean split and the
backdoor split of the same rows, clean_test and bd_test are 1 of the 2, and
mixed is half the rows of each in 1 batch. The backdoor split carries the
attack-success label, so its loss is the attacker's objective, which is what a
loss surface or a curvature on triggered inputs asks about.
"""

import os
import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from analysis.curvature import density_curve, hessian_density, top_hessian_eigenpairs
from analysis.features import transformer_blocks
from analysis.landscape import direction_cosine, loss_surface, random_direction
from analysis.synthesis import maximising_input, rank_dimensions_by_tac
from models.backbones import network_core

from .sidecar import write_sidecar
from .spectra import plot_hessian_density
from .style import DOUBLE_COLUMN, paper_style, save_figure
from .surfaces import plot_loss_surfaces

# Upstream evaluates 1 batch of 128. The retained double-backward graph of 32
# ViT-B/16 images at 224 peaks at 16.3 GiB, so 32 rows run as 2 micro-batches.
HESSIAN_BATCH_SIZE = 32
HESSIAN_MICRO_BATCH_SIZE = 16
HESSIAN_TOP_N = 2
HESSIAN_MAX_ITERATIONS = 1000
HESSIAN_TOLERANCE = 1e-3
HESSIAN_LANCZOS_STEPS = 100
HESSIAN_NUM_VECTORS = 1

# Upstream's 51 by 51 grid over the full training set is 4 to 6 GPU hours on
# ViT-B/16. 21 by 21 over 512 images is about 10 minutes for both splits.
LANDSCAPE_IMAGES = 512
LANDSCAPE_GRID_POINTS = 21
LANDSCAPE_RANGE = (-1.0, 1.0)
LANDSCAPE_NORMALISATION = "filter"
LANDSCAPE_DIRECTION_FILE = "landscape_directions.pt"

# 16 images per batch is 1 forward and 1 backward per step for every dimension.
SYNTHESIS_DIMENSIONS = 16
SYNTHESIS_TAC_BATCHES = 4
SYNTHESIS_STEPS = 300
SYNTHESIS_GRID_COLUMNS = 4

VIEW_SPLITS = {
    "paired": ("clean", "backdoor"),
    "clean_test": ("clean",),
    "bd_test": ("backdoor",),
    "mixed": ("mixed",),
}


def _first_rows(loader: DataLoader, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The first rows images and labels of a shuffle=False loader, on the CPU."""
    images, labels = [], []
    collected = 0
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels.long())
        collected += batch_images.size(0)
        if collected >= rows:
            break

    stacked_images = torch.cat(images)[:rows]  # (rows, 3, H, W)
    stacked_labels = torch.cat(labels)[:rows]  # (rows,)
    return stacked_images, stacked_labels


def _view_tensors(
    case, view: str, rows: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The named batches a view asks for, each (images, labels) of at most rows."""
    if view not in VIEW_SPLITS:
        raise ValueError(
            f"unknown view {view!r}, expected one of {sorted(VIEW_SPLITS)}"
        )

    if view == "mixed":
        half = rows // 2
        clean_images, clean_labels = _first_rows(case.loaders["clean"], half)
        backdoor_images, backdoor_labels = _first_rows(case.loaders["backdoor"], half)
        mixed = (
            torch.cat([clean_images, backdoor_images]),  # (2 * half, 3, H, W)
            torch.cat([clean_labels, backdoor_labels]),  # (2 * half,)
        )
        return {"mixed": mixed}

    tensors = {
        name: _first_rows(case.loaders[name], rows) for name in VIEW_SPLITS[view]
    }
    return tensors


def _chunks(
    images: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The rows as device-resident batches, so 441 passes cost no loader restarts."""
    chunks = [
        (
            images[start : start + batch_size].to(device),
            labels[start : start + batch_size].to(device),
        )
        for start in range(0, images.size(0), batch_size)
    ]
    return chunks


def _compact(values) -> list:
    """A tensor or array as a list of floats rounded to 6 significant digits."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    compacted = [float(f"{value:.6g}") for value in array]
    return compacted


def _figure_path(case, tool: str) -> str:
    path = os.path.join(case.out_dir, f"{tool}.pdf")
    return path


def run_hessian(case, args) -> None:
    """Top-2 Hessian eigenvalues and the SLQ spectral density on 1 batch per split."""
    started = time.time()
    case.model.eval()
    loss_fn = nn.CrossEntropyLoss()

    curves, tops, nodes_all, weights_all = {}, {}, {}, {}
    node_low, node_high = float("inf"), float("-inf")
    for name, batch in _view_tensors(case, args.view, HESSIAN_BATCH_SIZE).items():
        eigenvalues, _ = top_hessian_eigenpairs(
            case.model,
            loss_fn,
            batch,
            top_n=HESSIAN_TOP_N,
            max_iterations=HESSIAN_MAX_ITERATIONS,
            tolerance=HESSIAN_TOLERANCE,
            seed=args.seed,
            micro_batch_size=HESSIAN_MICRO_BATCH_SIZE,
        )
        nodes, weights = hessian_density(
            case.model,
            loss_fn,
            batch,
            lanczos_steps=HESSIAN_LANCZOS_STEPS,
            num_vectors=HESSIAN_NUM_VECTORS,
            seed=args.seed,
            micro_batch_size=HESSIAN_MICRO_BATCH_SIZE,
        )
        density, grid = density_curve(nodes, weights)
        curves[name] = (density, grid)
        tops[name] = eigenvalues
        nodes_all[name] = nodes
        weights_all[name] = weights
        node_low = min(node_low, float(np.min(nodes)))
        node_high = max(node_high, float(np.max(nodes)))
        print(
            f"[hessian] {name}: top eigenvalues "
            + ", ".join(f"{value:.4f}" for value in eigenvalues),
            flush=True,
        )

    figure_path = _figure_path(case, "hessian")
    plot_hessian_density(curves, tops, (node_low, node_high), case.folder, figure_path)
    wall = time.time() - started
    write_sidecar(
        figure_path,
        "hessian",
        case.folder,
        settings={
            "view": args.view,
            "samples": HESSIAN_BATCH_SIZE,
            "micro_batch_size": HESSIAN_MICRO_BATCH_SIZE,
            "seed": args.seed,
            "top_n": HESSIAN_TOP_N,
            "max_iterations": HESSIAN_MAX_ITERATIONS,
            "tolerance": HESSIAN_TOLERANCE,
            "lanczos_steps": HESSIAN_LANCZOS_STEPS,
            "num_vectors": HESSIAN_NUM_VECTORS,
            "dtype": "float32",
            "wall_seconds": round(wall, 1),
        },
        plotted={
            name: {
                "top_eigenvalues": tops[name],
                "nodes": [_compact(run) for run in nodes_all[name]],
                "weights": [_compact(run) for run in weights_all[name]],
                "grid": _compact(curves[name][1]),
                "density": _compact(curves[name][0]),
            }
            for name in curves
        },
    )
    print(f"[hessian] wall {wall:.0f} s", flush=True)


def _load_or_draw_directions(case, seed: int) -> tuple[tuple[list, list], float]:
    """The checkpoint's 2 cached directions, drawn and cached when absent.

    Cached under out_dir so every surface of this checkpoint, clean or triggered,
    now or in a later run, shares its coordinates. Seeds are seed and seed + 1,
    2 independent draws, and a cache made under other seeds is redrawn.
    """
    path = os.path.join(case.out_dir, LANDSCAPE_DIRECTION_FILE)
    seeds = [seed, seed + 1]
    if os.path.exists(path):
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if (
            cached["seeds"] == seeds
            and cached["normalisation"] == LANDSCAPE_NORMALISATION
        ):
            first = [tensor.to(case.device) for tensor in cached["first"]]
            second = [tensor.to(case.device) for tensor in cached["second"]]
            print(f"[landscape] directions loaded from {path}", flush=True)
            return (first, second), cached["cosine"]

    first = random_direction(case.model, seeds[0], LANDSCAPE_NORMALISATION)
    second = random_direction(case.model, seeds[1], LANDSCAPE_NORMALISATION)
    cosine = direction_cosine(first, second)
    torch.save(
        {
            "seeds": seeds,
            "normalisation": LANDSCAPE_NORMALISATION,
            "cosine": cosine,
            "first": [tensor.cpu() for tensor in first],
            "second": [tensor.cpu() for tensor in second],
        },
        path,
    )
    print(
        f"[landscape] directions drawn, cosine {cosine:.2e}, cached at {path}",
        flush=True,
    )
    return (first, second), cosine


def run_landscape(case, args) -> None:
    """The loss and accuracy surfaces on a 21 by 21 grid, 512 images per split."""
    started = time.time()
    case.model.eval()
    loss_fn = nn.CrossEntropyLoss()
    directions, cosine = _load_or_draw_directions(case, args.seed)
    grid = torch.linspace(*LANDSCAPE_RANGE, LANDSCAPE_GRID_POINTS)  # (points,)
    rows = min(args.samples, LANDSCAPE_IMAGES)

    losses_by_split, accuracies_by_split = {}, {}
    for name, (images, labels) in _view_tensors(case, args.view, rows).items():
        batches = _chunks(images, labels, args.batch_size, case.device)
        losses, accuracies = loss_surface(
            case.model,
            batches,
            directions,
            grid,
            grid,
            loss_fn,
            case.device,
            not args.no_bfloat16,
        )
        losses_by_split[name] = losses.numpy()
        accuracies_by_split[name] = accuracies.numpy()
        print(
            f"[landscape] {name}: loss {losses.min():.4f} to {losses.max():.4f} "
            f"over {images.size(0)} rows",
            flush=True,
        )

    figure_path = _figure_path(case, "landscape")
    plot_loss_surfaces(
        losses_by_split, grid.numpy(), grid.numpy(), case.folder, figure_path
    )
    wall = time.time() - started
    write_sidecar(
        figure_path,
        "landscape",
        case.folder,
        settings={
            "view": args.view,
            "samples": rows,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "direction_seeds": [args.seed, args.seed + 1],
            "normalisation": LANDSCAPE_NORMALISATION,
            "grid_points": LANDSCAPE_GRID_POINTS,
            "range": list(LANDSCAPE_RANGE),
            "bfloat16": not args.no_bfloat16,
            "direction_cosine": cosine,
            "wall_seconds": round(wall, 1),
        },
        plotted={
            "alphas": _compact(grid),
            "betas": _compact(grid),
            "losses": {
                name: losses.tolist() for name, losses in losses_by_split.items()
            },
            "accuracies": {
                name: accuracies.tolist()
                for name, accuracies in accuracies_by_split.items()
            },
        },
    )
    print(f"[landscape] wall {wall:.0f} s", flush=True)


def _plot_synthesised_grid(
    images: torch.Tensor,
    dimensions: list[int],
    tac: list[float],
    title: str,
    pdf_path: str,
) -> None:
    """The synthesised images in a grid, each titled by its dimension and its TAC.

    Upstream lays its channels out 16 per row, titled Kernel i. 16 images at
    native resolution read better as a 4 by 4 square, drawn with nearest
    interpolation so every pixel of a 32 by 32 canvas stays a visible square.
    """
    count = images.size(0)
    columns = SYNTHESIS_GRID_COLUMNS
    rows = int(np.ceil(count / columns))
    with paper_style():
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * rows / columns + 0.4),
            squeeze=False,
        )
        for index, axis in enumerate(axes.flat):
            axis.set_axis_off()
            if index >= count:
                continue
            pixels = images[index].permute(1, 2, 0).clamp(0, 1).numpy()  # (H, W, 3)
            axis.imshow(pixels, interpolation="nearest")
            axis.set_title(f"dim {dimensions[index]}, TAC {tac[index]:.2f}")
        figure.suptitle(title)
        figure.tight_layout()

    save_figure(figure, pdf_path)


def run_feature_visualization(case, args) -> None:
    """The input that maximises each of the 16 highest-TAC dimensions of a block.

    --layer picks the block, the last one by default as upstream picks the last
    convolution. The ranking always reads the paired clean and backdoor loaders,
    since TAC is a paired statistic, so --view has no effect here.
    """
    started = time.time()
    case.model.eval()
    use_bfloat16 = not args.no_bfloat16
    blocks = transformer_blocks(network_core(case.model), case.architecture)
    layer = args.layer if args.layer is not None else len(blocks)

    order, tac = rank_dimensions_by_tac(
        case.model,
        case.loaders["clean"],
        case.loaders["backdoor"],
        layer,
        case.device,
        use_bfloat16,
        max_batches=SYNTHESIS_TAC_BATCHES,
        architecture=case.architecture,
    )
    dimensions = order[:SYNTHESIS_DIMENSIONS]  # (16,)
    images, objectives = maximising_input(
        case.model,
        layer,
        dimensions,
        case.image_size,
        case.mean,
        case.std,
        case.device,
        use_bfloat16,
        steps=SYNTHESIS_STEPS,
        seed=args.seed,
        architecture=case.architecture,
    )
    print(
        f"[feature_visualization] layer {layer}, dimensions {dimensions.tolist()}, "
        f"objective {objectives[0].mean():.3f} to {objectives[-1].mean():.3f}",
        flush=True,
    )

    figure_path = _figure_path(case, "feature_visualization")
    _plot_synthesised_grid(
        images,
        dimensions.tolist(),
        tac[dimensions].tolist(),
        f"{case.folder}, block {layer}",
        figure_path,
    )
    wall = time.time() - started
    write_sidecar(
        figure_path,
        "feature_visualization",
        case.folder,
        settings={
            "layer": layer,
            "view": args.view,
            "samples": SYNTHESIS_TAC_BATCHES * args.batch_size,
            "seed": args.seed,
            "steps": SYNTHESIS_STEPS,
            "dimensions": SYNTHESIS_DIMENSIONS,
            "image_size": case.image_size,
            "bfloat16": use_bfloat16,
            "wall_seconds": round(wall, 1),
        },
        plotted={
            "dimensions": dimensions.tolist(),
            "tac": _compact(tac[dimensions]),
            "objective_first_step": _compact(objectives[0]),
            "objective_last_step": _compact(objectives[-1]),
            "images_uint8": (images.clamp(0, 1) * 255).round().to(torch.uint8).tolist(),
        },
    )
    print(f"[feature_visualization] wall {wall:.0f} s", flush=True)


TOOLS: dict[str, Callable] = {
    "hessian": run_hessian,
    "landscape": run_landscape,
    "feature_visualization": run_feature_visualization,
}
