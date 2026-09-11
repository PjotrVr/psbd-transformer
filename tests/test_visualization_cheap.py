"""The figure builders, the embedding projections and every cheap tool end to end on a toy case.

The figure builders are checked for the contract a tool relies on: a Figure
comes back, the purity raster fills rows in proportion, the overlay stays in
range and save_figure writes both files. The 2 projections upstream calls
through libraries are checked against those libraries' direct calls with the
same seed. Then every registered tool runs on a synthetic VisualCase built
from a tiny ViT and paired loaders, on the CPU, and must leave a PDF, a PNG
and a sidecar behind with the plotted arrays inside.
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import VisionTransformer

from analysis.embedding import tsne_project, umap_project, unit_square
from cli.visualize import VisualCase
from tests.reference import backdoorbench as upstream
from visualization import cheap_tools
from visualization.bars import (
    PURITY_COLUMNS,
    class_purity_bars,
    metric_radar,
    overlaid_activation_bars,
    purity_raster,
    stealth_bars,
)
from visualization.heatmaps import layer_by_dimension_heatmap, token_map_grid
from visualization.matrices import confusion_matrix_figure
from visualization.panels import (
    activating_image_grid,
    attribution_rows,
    overlay_heat_map,
    paired_image_panels,
)
from visualization.scatter import embedding_scatter
from visualization.sidecar import sidecar_path
from visualization.style import colour_table, save_figure

matplotlib.use("Agg")

NUM_ROWS = 64
NUM_CLASSES = 10
TARGET = 0
EXPECTED_TOOLS = {
    "tac",
    "lipschitz",
    "neuron_activation",
    "activation_distribution",
    "activating_images",
    "tsne",
    "umap",
    "pca",
    "confusion",
    "token_maps",
    "gradcam",
    "frequency",
    "expected_gradients",
    "stealth",
    "metrics",
}


def test_every_figure_builder_returns_a_figure():
    generator = np.random.default_rng(0)
    figures = [
        layer_by_dimension_heatmap(
            generator.random((3, 20)), ["1", "2", "3"], "Blues", "TAC", "t", False
        ),
        layer_by_dimension_heatmap(
            generator.random((3, 20)), ["1", "2", "3"], "Oranges", "L", "t", True
        ),
        token_map_grid(
            generator.random((5, 2, 2)),
            generator.random((5, 2, 2)),
            np.arange(5),
            generator.random(5),
            "t",
        ),
        overlaid_activation_bars(generator.random(30), generator.random(30), "t"),
        class_purity_bars(
            np.array([[0.5, 0.5], [1.0, 0.0]]),
            np.array([[1, 0, 0], [0, 0, 0]]),
            ["a", "b"],
            ["#ff0000", "#000000"],
            "t",
        ),
        stealth_bars(
            {
                "psnr_mean": 30.0,
                "psnr_std": 1.0,
                "ssim_mean": 0.9,
                "ssim_std": 0.01,
                "lpips_mean": 0.1,
                "lpips_std": 0.01,
                "n_pairs": 4,
            },
            "badnet",
        ),
        metric_radar(["a", "b", "c"], [0.9, 0.5, 0.1], "t"),
        embedding_scatter(
            generator.random((20, 2)),
            generator.integers(0, 3, 20),
            generator.random(20) < 0.3,
            np.arange(3),
            "t",
            ("x", "y"),
            5.0,
            0.5,
        ),
        confusion_matrix_figure(generator.random((4, 4)), True, "t"),
        confusion_matrix_figure(generator.random((25, 25)), False, "t"),
        paired_image_panels(
            generator.random((4, 8, 8, 3)),
            generator.random((4, 8, 8, 3)),
            ["a"] * 4,
            ["b"] * 4,
            "rgb",
        ),
        paired_image_panels(
            generator.random((4, 8, 8, 3)),
            generator.integers(0, 255, (4, 8, 8)),
            ["a"] * 4,
            ["b"] * 4,
            "frequency",
        ),
        attribution_rows(
            generator.random((2, 8, 8, 3)),
            generator.standard_normal((2, 2, 8, 8)),
            [["c", "d"]] * 2,
            ["a", "b"],
        ),
        activating_image_grid(
            generator.random((2, 3, 8, 8, 3)),
            np.arange(2),
            generator.random((2, 3)),
            generator.random((2, 3)) < 0.5,
            "t",
        ),
    ]
    for figure in figures:
        assert isinstance(figure, matplotlib.figure.Figure)
        matplotlib.pyplot.close(figure)


def test_the_panels_refuse_other_than_4_images_and_unknown_kinds():
    pixels = np.zeros((3, 4, 4, 3))
    with pytest.raises(ValueError, match="4 images"):
        paired_image_panels(pixels, pixels, ["a"] * 3, ["b"] * 3, "rgb")
    with pytest.raises(ValueError, match="companion kind"):
        paired_image_panels(
            np.zeros((4, 4, 4, 3)), np.zeros((4, 4, 4, 3)), ["a"] * 4, ["b"] * 4, "cam"
        )


def test_the_purity_raster_fills_rows_in_proportion():
    colours = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    raster = purity_raster(np.array([[0.25, 0.75, 0.0], [0.0, 0.0, 1.0]]), colours)
    assert raster.shape == (2, PURITY_COLUMNS, 3)
    assert np.array_equal(
        raster[0, : PURITY_COLUMNS // 4], np.tile(colours[0], (PURITY_COLUMNS // 4, 1))
    )
    assert np.array_equal(
        raster[0, PURITY_COLUMNS // 4 :],
        np.tile(colours[1], (3 * PURITY_COLUMNS // 4, 1)),
    )
    assert np.array_equal(raster[1], np.tile(colours[2], (PURITY_COLUMNS, 1)))


def test_the_overlay_stays_in_range_and_peaks_at_1():
    generator = np.random.default_rng(1)
    overlay = overlay_heat_map(
        generator.random((2, 6, 6, 3)), generator.random((2, 6, 6))
    )
    assert overlay.shape == (2, 6, 6, 3)
    assert overlay.min() >= 0.0 and np.allclose(overlay.reshape(2, -1).max(axis=1), 1.0)


def test_colour_table_marks_the_poisoned_class_black_and_the_rest_grey():
    table = colour_table(np.array([3, 1]), 5)
    assert table.shape == (6, 3)
    assert np.array_equal(table[5], [0.0, 0.0, 0.0])
    assert np.allclose(table[0], table[2]) and not np.allclose(table[3], table[1])


def test_save_figure_writes_pdf_and_png(tmp_path):
    figure = metric_radar(["a", "b", "c"], [0.1, 0.2, 0.3], "t")
    path = str(tmp_path / "radar.pdf")
    save_figure(figure, path)
    assert os.path.exists(path) and os.path.exists(str(tmp_path / "radar.png"))
    with pytest.raises(ValueError, match="pdf"):
        save_figure(
            metric_radar(["a", "b", "c"], [0.1, 0.2, 0.3], "t"),
            str(tmp_path / "radar.svg"),
        )


def test_tsne_project_is_the_library_call_with_the_same_seed():
    features = torch.rand(50, 6, generator=torch.Generator().manual_seed(2))
    ours = tsne_project(features, seed=0)
    theirs = TSNE(n_components=2, init="random", random_state=0).fit_transform(
        features.numpy()
    )
    assert np.allclose(ours, theirs)
    assert np.allclose(ours, upstream.get_embedding_tsne(features.numpy()))


def test_umap_project_is_the_library_call_with_the_same_seed():
    umap = pytest.importorskip("umap")
    features = torch.rand(60, 6, generator=torch.Generator().manual_seed(3))
    ours = umap_project(features, seed=0)
    theirs = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(
        features.numpy()
    )
    assert np.allclose(ours, theirs)


def test_unit_square_is_upstream_plot_embedding_scaling():
    embedding = np.random.default_rng(4).standard_normal((30, 2)) * 5 + 2
    ours = unit_square(embedding)
    assert np.allclose(ours, upstream.plot_embedding_scaling(embedding))
    assert np.allclose(ours.min(axis=0), 0.0) and np.allclose(ours.max(axis=0), 1.0)


@pytest.fixture(scope="module")
def toy_case(tmp_path_factory) -> VisualCase:
    """A VisualCase over a tiny ViT and synthetic paired loaders, on the CPU."""
    torch.manual_seed(0)
    network = VisionTransformer(
        image_size=32,
        patch_size=16,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=NUM_CLASSES,
    )
    nn.init.normal_(network.heads.head.weight, std=0.5)
    model = nn.Sequential(transforms_v2.Resize((32, 32)), network).eval()

    generator = torch.Generator().manual_seed(1)
    images = torch.rand(NUM_ROWS, 3, 32, 32, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (NUM_ROWS,), generator=generator)
    test_indices = torch.randperm(500, generator=generator)[:NUM_ROWS]
    eligible = [row for row in range(NUM_ROWS) if int(labels[row]) != TARGET]
    triggered = images[eligible].clone()
    triggered[:, :, -4:, -4:] = 1.0
    validation = torch.rand(16, 3, 32, 32, generator=generator)
    loaders = {
        "validation": DataLoader(
            TensorDataset(validation, torch.zeros(16, dtype=torch.long)), batch_size=16
        ),
        "clean": DataLoader(
            TensorDataset(images, labels), batch_size=16, shuffle=False
        ),
        "backdoor": DataLoader(
            TensorDataset(triggered, torch.full((len(eligible),), TARGET)),
            batch_size=16,
            shuffle=False,
        ),
    }
    manifest = {
        "analysis_clean_indices": test_indices.tolist(),
        "analysis_backdoor_indices": test_indices[eligible].tolist(),
        "probe_attack": "badnet_a2o",
        "probe_target_label": TARGET,
    }
    metadata = {
        "dataset": "toy",
        "attack": "badnet_a2o",
        "poison_rate": 0.1,
        "clean_accuracy": 0.5,
        "asr": 0.9,
    }
    case = VisualCase(
        folder="toy_case",
        metadata=metadata,
        model=model,
        loaders=loaders,
        manifest=manifest,
        device=torch.device("cpu"),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        num_classes=NUM_CLASSES,
        image_size=32,
        architecture="vit",
        out_dir=str(tmp_path_factory.mktemp("visual")),
    )
    return case


def toy_args(**overrides) -> argparse.Namespace:
    settings = {
        "layer": None,
        "view": "paired",
        "samples": NUM_ROWS,
        "classes": 10,
        "seed": 0,
        "batch_size": 16,
        "no_bfloat16": True,
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


def test_the_registry_lists_the_15_cheap_tools():
    assert set(cheap_tools.TOOLS) == EXPECTED_TOOLS


@pytest.mark.parametrize("tool", sorted(EXPECTED_TOOLS))
def test_every_cheap_tool_writes_its_figure_and_sidecar(toy_case, tool):
    cheap_tools.TOOLS[tool](toy_case, toy_args())
    pdf = os.path.join(toy_case.out_dir, f"{tool}.pdf")
    assert os.path.exists(pdf) and os.path.exists(pdf[:-4] + ".png")
    with open(sidecar_path(pdf)) as handle:
        sidecar = json.load(handle)
    assert sidecar["tool"] == tool and sidecar["folder"] == "toy_case"
    assert sidecar["plotted"] and sidecar["settings"]["seed"] == 0


def test_tac_csv_has_a_row_per_layer_and_dimension(toy_case):
    cheap_tools.run_tac(toy_case, toy_args())
    with open(os.path.join(toy_case.out_dir, "tac.csv")) as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "layer,neuron,tac" and len(lines) == 1 + 2 * 16


def test_gradcam_sidecar_records_trigger_coverage_of_the_poisoned_panels(toy_case):
    cheap_tools.run_gradcam(toy_case, toy_args())
    with open(os.path.join(toy_case.out_dir, "gradcam.json")) as handle:
        sidecar = json.load(handle)
    coverage = sidecar["plotted"]["trigger_coverage"]
    assert [entry["panel"] for entry in coverage] == [2, 3]
    assert all(entry["trigger_pixels"] == 16 for entry in coverage)
    assert sidecar["plotted"]["poison_mask"] == [False, False, True, True]


@pytest.mark.parametrize("view", ["clean_test", "bd_test", "mixed"])
def test_the_view_tools_accept_every_view(toy_case, view):
    cheap_tools.run_confusion(toy_case, toy_args(view=view))
    cheap_tools.run_activation_distribution(toy_case, toy_args(view=view, layer=1))
    with open(os.path.join(toy_case.out_dir, "confusion.json")) as handle:
        assert json.load(handle)["settings"]["view"] == view


def test_a_layer_outside_the_model_is_refused(toy_case):
    with pytest.raises(ValueError, match="outside"):
        cheap_tools.run_neuron_activation(toy_case, toy_args(layer=5))
