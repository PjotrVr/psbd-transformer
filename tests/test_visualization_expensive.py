"""The 3 expensive tools end to end on the synthetic ViT, with every cost knob shrunk.

Each tool runs against a case shaped like cli.visualize's VisualCase, writes its
PDF, PNG and sidecar into a temporary directory. The sidecar is checked for
the arrays the figure drew. The knobs are module constants, patched down so the
whole file runs on the CPU in under a minute.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from experiments.preflight.synthetic import build_backdoored_model, build_splits
from visualization import expensive_tools

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    """torch defaults to 64 threads on the 128-core login node, where the tiny CPU
    ops of a toy model run 1000 times slower than at 8 from OpenMP overhead."""
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 8))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def synthetic_case(tmp_path_factory):
    out_dir = str(tmp_path_factory.mktemp("visual"))
    inner = build_backdoored_model().inner.eval()
    case = SimpleNamespace(
        folder="synthetic",
        metadata={"attack": "synthetic"},
        model=inner,
        loaders=build_splits(num_samples=32, batch_size=16),
        manifest={},
        device=DEVICE,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        num_classes=10,
        image_size=32,
        architecture="vit",
        out_dir=out_dir,
    )
    return case


@pytest.fixture
def args():
    return SimpleNamespace(
        layer=None,
        view="paired",
        samples=16,
        classes=10,
        seed=0,
        batch_size=16,
        no_bfloat16=True,
    )


@pytest.fixture(autouse=True)
def small_knobs(monkeypatch):
    monkeypatch.setattr(expensive_tools, "HESSIAN_BATCH_SIZE", 8)
    monkeypatch.setattr(expensive_tools, "HESSIAN_MICRO_BATCH_SIZE", 4)
    monkeypatch.setattr(expensive_tools, "HESSIAN_MAX_ITERATIONS", 3)
    monkeypatch.setattr(expensive_tools, "HESSIAN_LANCZOS_STEPS", 6)
    monkeypatch.setattr(expensive_tools, "LANDSCAPE_GRID_POINTS", 3)
    monkeypatch.setattr(expensive_tools, "LANDSCAPE_IMAGES", 16)
    monkeypatch.setattr(expensive_tools, "SYNTHESIS_STEPS", 3)
    monkeypatch.setattr(expensive_tools, "SYNTHESIS_DIMENSIONS", 4)
    monkeypatch.setattr(expensive_tools, "SYNTHESIS_TAC_BATCHES", 1)


def read_sidecar(out_dir: str, tool: str) -> dict:
    with open(os.path.join(out_dir, f"{tool}.json")) as handle:
        return json.load(handle)


def assert_figure_written(out_dir: str, tool: str) -> None:
    for extension in ("pdf", "png", "json"):
        path = os.path.join(out_dir, f"{tool}.{extension}")
        assert os.path.exists(path) and os.path.getsize(path) > 0, path


def test_the_registry_holds_exactly_the_3_expensive_tools():
    assert set(expensive_tools.TOOLS) == {
        "hessian",
        "landscape",
        "feature_visualization",
    }


def test_hessian_tool_writes_its_figure_and_the_spectrum_it_drew(synthetic_case, args):
    expensive_tools.TOOLS["hessian"](synthetic_case, args)

    assert_figure_written(synthetic_case.out_dir, "hessian")
    sidecar = read_sidecar(synthetic_case.out_dir, "hessian")
    assert sidecar["tool"] == "hessian" and sidecar["folder"] == "synthetic"
    assert sidecar["settings"]["samples"] == 8
    assert sidecar["settings"]["lanczos_steps"] == 6
    for split in ("clean", "backdoor"):
        plotted = sidecar["plotted"][split]
        assert len(plotted["top_eigenvalues"]) == 2
        assert len(plotted["nodes"][0]) == 6 and len(plotted["weights"][0]) == 6
        assert abs(sum(plotted["weights"][0]) - 1.0) < 1e-4
        assert len(plotted["grid"]) == 10000 and len(plotted["density"]) == 10000


def test_hessian_tool_reads_1_split_under_a_single_split_view(synthetic_case, args):
    args.view = "bd_test"
    expensive_tools.TOOLS["hessian"](synthetic_case, args)

    sidecar = read_sidecar(synthetic_case.out_dir, "hessian")
    assert list(sidecar["plotted"]) == ["backdoor"]


def test_landscape_tool_writes_both_surfaces_and_caches_its_directions(
    synthetic_case, args
):
    before = {
        key: value.clone() for key, value in synthetic_case.model.state_dict().items()
    }
    expensive_tools.TOOLS["landscape"](synthetic_case, args)

    assert_figure_written(synthetic_case.out_dir, "landscape")
    cache = os.path.join(synthetic_case.out_dir, "landscape_directions.pt")
    assert os.path.exists(cache)
    sidecar = read_sidecar(synthetic_case.out_dir, "landscape")
    assert sidecar["settings"]["grid_points"] == 3
    assert sidecar["settings"]["samples"] == 16
    assert abs(sidecar["settings"]["direction_cosine"]) < 0.2
    for split in ("clean", "backdoor"):
        losses = sidecar["plotted"]["losses"][split]
        accuracies = sidecar["plotted"]["accuracies"][split]
        assert len(losses) == 3 and len(losses[0]) == 3
        assert all(0.0 <= value <= 1.0 for row in accuracies for value in row)
    after = synthetic_case.model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)

    # A second run reuses the cached directions, so the surface is the same.
    expensive_tools.TOOLS["landscape"](synthetic_case, args)
    again = read_sidecar(synthetic_case.out_dir, "landscape")
    assert again["plotted"]["losses"] == sidecar["plotted"]["losses"]


def test_mixed_view_concatenates_half_of_each_split(synthetic_case, args):
    tensors = expensive_tools._view_tensors(synthetic_case, "mixed", 8)

    assert list(tensors) == ["mixed"]
    images, labels = tensors["mixed"]
    assert images.shape == (8, 3, 32, 32) and labels.shape == (8,)
    with pytest.raises(ValueError, match="unknown view"):
        expensive_tools._view_tensors(synthetic_case, "everything", 8)


def test_feature_visualization_tool_writes_the_grid_of_top_tac_dimensions(
    synthetic_case, args
):
    expensive_tools.TOOLS["feature_visualization"](synthetic_case, args)

    assert_figure_written(synthetic_case.out_dir, "feature_visualization")
    sidecar = read_sidecar(synthetic_case.out_dir, "feature_visualization")
    assert sidecar["settings"]["layer"] == 2, "the last block of the 2-block model"
    assert sidecar["settings"]["steps"] == 3
    plotted = sidecar["plotted"]
    assert len(plotted["dimensions"]) == 4 and len(set(plotted["dimensions"])) == 4
    assert plotted["tac"] == sorted(plotted["tac"], reverse=True)
    images = torch.tensor(plotted["images_uint8"])
    assert images.shape == (4, 3, 32, 32)
    assert images.min() >= 0 and images.max() <= 255


def test_feature_visualization_honours_an_explicit_layer(synthetic_case, args):
    args.layer = 1
    expensive_tools.TOOLS["feature_visualization"](synthetic_case, args)

    sidecar = read_sidecar(synthetic_case.out_dir, "feature_visualization")
    assert sidecar["settings"]["layer"] == 1
