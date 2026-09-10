"""SentiNet pinned on the CPU: the CAM site, the analytic map, the transplant and the envelope.

The synthetic fixture installs its backdoor at the logits by boolean indexing on
the input, so the trigger never passes through the tokens Grad-CAM reads and
SentiNet cannot separate on it by construction. These tests therefore take the
contract (shape, finiteness, determinism, no parameter gradient, native
resolution) and the CAM site from the fixture, the Grad-CAM arithmetic from a
model whose logit is a fixed linear read of the mean patch token, the transplant
statistics from a per-image reference loop, the envelope from points on a known
parabola and the direction from a model whose prediction is driven by a corner
region. The fixture's own AUROC is printed and never asserted.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import EncoderBlock

from defences.decision import HEADLINE_QUANTILE, detection_report
from detectors.sentinet import (
    BOUNDARY_BIN_WIDTH,
    BOUNDARY_POINTS_PER_BIN,
    CAM_LAYER_OFFSET,
    DEFAULT_NUM_OVERLAYS,
    MASK_THRESHOLD,
    OFFICIAL_TOOLBOX_MASK_FRACTION,
    boundary_residual,
    cam_layer,
    cam_token_weights,
    collect_overlay_pixels,
    draw_inert_pixels,
    fit_decision_boundary,
    grad_cam,
    overlay_statistics,
    resolve_cam_site,
    saliency_mask,
    scale_per_image,
    sentinet_scores,
    sentinet_statistics,
)
from experiments.preflight.synthetic import (
    BACKDOOR_LOGIT,
    IMAGE_SIZE,
    TARGET_CLASS,
    TRIGGER_SIZE,
    apply_trigger,
    build_backdoored_model,
    build_splits,
    trigger_pattern,
)

DEVICE = torch.device("cpu")
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)

# Enough overlays to give fooled a resolution of 1/8, few enough to keep the
# file under a minute.
FEW_OVERLAYS = 8

# The fixture's patch size, so its 32-pixel image is a 4 by 4 grid of cells.
PATCH = 8
GRID = IMAGE_SIZE // PATCH

# Steepness and half-open point of the planted trigger gate, as in the CD-L
# test: the exact checkerboard opens it to 0.99995 and a random corner, which
# deviates by 0.5 on average, closes it to 0.00005.
GATE_SHARPNESS = 40.0
GATE_MARGIN = 0.25


@pytest.fixture(scope="module")
def synthetic_case():
    model = build_backdoored_model()
    loaders = build_splits(num_samples=32, batch_size=16)
    return model, loaders


def test_the_constants_are_the_recorded_ones():
    assert DEFAULT_NUM_OVERLAYS == 100
    assert MASK_THRESHOLD == 0.85
    assert OFFICIAL_TOOLBOX_MASK_FRACTION == 0.15
    assert BOUNDARY_BIN_WIDTH == 0.04
    assert BOUNDARY_POINTS_PER_BIN == 2
    assert CAM_LAYER_OFFSET == {"vit": -1, "swin": 0}


def test_cam_layer_reads_the_last_block_input_on_vit_and_output_on_swin():
    assert cam_layer(12, "vit") == 11
    assert cam_layer(24, "swin") == 24
    with pytest.raises(ValueError, match="no CAM layer rule"):
        cam_layer(12, "resnet")


def test_resolve_cam_site_finds_the_fixture_vit_through_its_wrapper(synthetic_case):
    model, _ = synthetic_case

    assert resolve_cam_site(model.inner) == (1, "vit")
    assert resolve_cam_site(model) == (1, "vit")


def test_grad_cam_on_the_fixture_has_the_grid_shape_and_nonzero_variance(
    synthetic_case,
):
    """The test that catches the zero-gradient site if CAM_LAYER_OFFSET ever moves."""
    model, loaders = synthetic_case
    images, _ = next(iter(loaders["clean"]))  # (16, 3, 32, 32)
    layer, architecture = resolve_cam_site(model.inner)

    cam, predicted = grad_cam(model.inner, images, DEVICE, False, layer, architecture)

    assert cam.shape == (16, GRID, GRID)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert cam.var() > 0, "a constant map means the CAM site receives no gradient"
    with torch.inference_mode():
        assert torch.equal(predicted, model.inner(images).argmax(dim=1))


def test_the_last_block_output_gives_an_all_zero_map_on_vit(synthetic_case):
    """torchvision's ViT reads x[:, 0] after the encoder, so every patch token of
    the last block's OUTPUT receives gradient 0, the map is constant and scales to
    all 0. CAM_LAYER_OFFSET["vit"] = -1 exists to step back from that site."""
    model, loaders = synthetic_case
    images, _ = next(iter(loaders["clean"]))  # (16, 3, 32, 32)
    num_blocks = 2
    assert cam_layer(num_blocks, "vit") == num_blocks - 1

    cam_at_output, _ = grad_cam(model.inner, images, DEVICE, False, num_blocks, "vit")

    assert torch.equal(cam_at_output, torch.zeros_like(cam_at_output))


class MeanTokenReadout(nn.Module):
    """1 encoder block whose output patch tokens are averaged and read by a linear head.

    With the logit a fixed linear read of the mean patch token, the gradient of
    logit c with respect to every output patch token is the head row for c
    divided by the patch count, so the Grad-CAM weights and the map are known in
    closed form and the port's arithmetic can be checked against them.
    """

    def __init__(self, hidden_dim: int = 16, num_classes: int = 5, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=PATCH, stride=PATCH)
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.block = EncoderBlock(
            num_heads=2,
            hidden_dim=hidden_dim,
            mlp_dim=2 * hidden_dim,
            dropout=0.0,
            attention_dropout=0.0,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        patches = (
            self.patch_embedding(images).flatten(2).transpose(1, 2)
        )  # (batch, patches, dim)
        class_tokens = self.class_token.expand(
            images.size(0), -1, -1
        )  # (batch, 1, dim)
        tokens = torch.cat([class_tokens, patches], dim=1)  # (batch, 1 + patches, dim)
        return tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        block_output = self.block(self.embed(images))  # (batch, 1 + patches, dim)
        pooled = block_output[:, 1:, :].mean(dim=1)  # (batch, dim)
        logits = self.head(pooled)  # (batch, classes)
        return logits


def test_grad_cam_matches_the_analytic_map_on_a_mean_token_readout():
    model = MeanTokenReadout().eval()
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(6, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    num_patches = GRID * GRID

    # The head reads the block OUTPUT, so the closed form holds at layer 1, the
    # output of the only block, whatever CAM_LAYER_OFFSET says for a real ViT.
    with torch.inference_mode():
        block_output = model.block(model.embed(images))  # (6, 17, dim)
        predicted = model(images).argmax(dim=1)  # (6,)
        expected_weights = model.head.weight[predicted] / num_patches  # (6, dim)
        weighted = (block_output[:, 1:, :] * expected_weights[:, None, :]).sum(dim=2)
        expected_cam = scale_per_image(F.relu(weighted).view(6, GRID, GRID))

    _, weights, cam_predicted = cam_token_weights(
        model, images, DEVICE, False, 1, "vit"
    )
    cam, _ = grad_cam(model, images, DEVICE, False, 1, "vit")

    assert torch.equal(cam_predicted, predicted)
    assert torch.allclose(weights, expected_weights, atol=1e-5)
    assert torch.allclose(cam, expected_cam, atol=1e-5)


def test_saliency_mask_is_never_empty_and_holds_the_peak_cell():
    generator = torch.Generator().manual_seed(2)
    random_maps = scale_per_image(torch.rand(5, GRID, GRID, generator=generator))
    constant_map = scale_per_image(torch.full((1, GRID, GRID), 0.3))  # all 0
    cam = torch.cat([random_maps, constant_map])  # (6, 4, 4)

    mask = saliency_mask(cam, (IMAGE_SIZE, IMAGE_SIZE))  # (6, 1, 32, 32)

    assert mask.shape == (6, 1, IMAGE_SIZE, IMAGE_SIZE) and mask.dtype == torch.bool
    assert (mask.flatten(1).sum(dim=1) >= 1).all()
    peak_cells = cam.flatten(1).argmax(dim=1)  # (6,)
    for index, cell in enumerate(peak_cells.tolist()):
        row, column = divmod(cell, GRID)
        cell_pixels = mask[
            index,
            0,
            row * PATCH : (row + 1) * PATCH,
            column * PATCH : (column + 1) * PATCH,
        ]
        assert cell_pixels.any(), f"map {index}: peak cell {cell} is outside the mask"
    # A constant map has no salient region, so only the fallback pixel survives,
    # while a random map keeps its region well short of the whole image.
    assert mask[5].sum() == 1
    assert (mask[:5].flatten(1).float().mean(dim=1) < 0.5).all()


def reference_transplant(model, image, mask, label, overlays, inert):
    """Beatrix's _superimpose per composite: background * mask + overlay * (1 - mask).

    The adversarial composite is Beatrix's exactly. The inert composite follows
    Algorithm 3 with the noise inside the region, where Beatrix and BackdoorBench
    paste the input's region onto the noise instead.
    """
    region = mask.float()  # (1, H, W)
    adversarial = torch.stack(
        [image * region + overlay * (1 - region) for overlay in overlays]
    )  # (num_overlays, 3, H, W)
    inert_copies = torch.stack(
        [
            noise * region + overlay * (1 - region)
            for overlay, noise in zip(overlays, inert)
        ]
    )  # (num_overlays, 3, H, W)

    with torch.inference_mode():
        fooled_count = int((model(adversarial).argmax(dim=1) == label).sum().item())
        inert_probs = torch.softmax(
            model(inert_copies), dim=1
        )  # (num_overlays, classes)
        avg_conf = inert_probs.max(dim=1).values.mean().item()
    return fooled_count, avg_conf


def test_overlay_statistics_matches_a_per_image_reference_loop(synthetic_case):
    model, _ = synthetic_case
    network = model.inner
    generator = torch.Generator().manual_seed(3)
    pixels = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    overlays = torch.rand(FEW_OVERLAYS, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    inert = torch.rand(FEW_OVERLAYS, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    masks = torch.zeros(4, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    # The whole image, so every adversarial composite is the input itself.
    masks[0] = True
    masks[1, :, :8, :8] = True
    masks[2, :, 10:26, 4:20] = True
    masks[3, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = True
    with torch.inference_mode():
        predicted = network(pixels).argmax(dim=1)  # (4,)

    fooled, avg_conf = overlay_statistics(
        network,
        pixels,
        masks,
        predicted,
        overlays,
        inert,
        IDENTITY_MEAN,
        IDENTITY_STD,
        DEVICE,
        False,
    )

    assert fooled.shape == avg_conf.shape == (4,)
    assert fooled[0].item() == 1.0
    for index in range(4):
        expected_count, expected_conf = reference_transplant(
            network, pixels[index], masks[index], predicted[index], overlays, inert
        )
        assert round(fooled[index].item() * FEW_OVERLAYS) == expected_count
        assert avg_conf[index].item() == pytest.approx(expected_conf, abs=1e-6)


def test_overlay_statistics_refuses_a_mask_of_the_wrong_shape(synthetic_case):
    model, _ = synthetic_case
    pixels = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    overlays = torch.rand(FEW_OVERLAYS, 3, IMAGE_SIZE, IMAGE_SIZE)
    wrong_masks = torch.ones(2, 1, IMAGE_SIZE // 2, IMAGE_SIZE, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask shape"):
        overlay_statistics(
            model.inner,
            pixels,
            wrong_masks,
            torch.zeros(2, dtype=torch.long),
            overlays,
            overlays,
            IDENTITY_MEAN,
            IDENTITY_STD,
            DEVICE,
            False,
        )


def test_fit_decision_boundary_recovers_a_known_parabola_as_an_upper_envelope():
    true_coefficients = np.array([0.5, -0.3, 0.2])
    # 0.01 + 0.02 k never sits on a bin edge, so every 0.04 bin holds 2 points.
    conf = np.linspace(0.01, 0.99, 50)  # (50,)
    on_curve = np.polyval(true_coefficients, conf)  # (50,)
    # Points 0.1 under the curve in every bin must not enter the envelope.
    fooled = torch.from_numpy(np.concatenate([on_curve, on_curve - 0.1]))  # (100,)
    avg_conf = torch.from_numpy(np.concatenate([conf, conf]))  # (100,)

    coefficients = fit_decision_boundary(fooled, avg_conf)

    assert coefficients.shape == (3,)
    assert np.allclose(coefficients, true_coefficients, atol=1e-6)
    residual = boundary_residual(fooled, avg_conf, coefficients)  # (100,)
    assert torch.allclose(residual[:50], torch.zeros(50), atol=1e-5)
    assert torch.allclose(residual[50:], torch.full((50,), -0.1), atol=1e-5)


def test_the_envelope_degree_falls_back_under_3_populated_bins():
    two_bins = fit_decision_boundary(
        torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0.10, 0.11, 0.50, 0.51])
    )
    one_bin = fit_decision_boundary(
        torch.tensor([0.1, 0.2]), torch.tensor([0.50, 0.51])
    )

    assert two_bins.shape == (2,)
    assert one_bin.shape == (1,)
    assert one_bin[0] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="no clean points"):
        fit_decision_boundary(torch.empty(0), torch.empty(0))


class CornerGatedClassifier(nn.Module):
    """Logits read every pixel until the trigger appears, then only the corner.

    A random linear read of the whole image stands in for class evidence. A
    differentiable read of the checkerboard gates that evidence out and pins the
    target logit, so a composite that carries the exact checkerboard is sent to
    the target whatever surrounds it. The gate is a sigmoid rather than the
    fixture's boolean test so that the gradient reaches the corner pixels.
    """

    def __init__(self, num_classes: int = 10, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("pattern", trigger_pattern())
        self.register_buffer(
            "content_weights",
            torch.randn(3 * IMAGE_SIZE * IMAGE_SIZE, num_classes, generator=generator),
        )
        self.register_buffer(
            "target_onehot",
            F.one_hot(torch.tensor(TARGET_CLASS), num_classes).float(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Centred, since on uniform pixels an uncentred read is dominated by each
        # class's column sum and 1 default class takes most overlays, which
        # would push fooled to 1.0 on clean inputs for a benign reason.
        content_logits = (images - 0.5).flatten(
            1
        ) @ self.content_weights  # (batch, classes)

        corner = images[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:]  # (batch, 3, T, T)
        deviation = (corner - self.pattern).abs().mean(dim=(1, 2, 3))  # (batch,)
        presence = torch.sigmoid(GATE_SHARPNESS * (GATE_MARGIN - deviation))
        presence = presence.unsqueeze(1)  # (batch, 1)

        backdoor_logits = BACKDOOR_LOGIT * self.target_onehot  # (classes,)
        logits = (1 - presence) * content_logits + presence * backdoor_logits
        return logits


def test_a_transplanted_trigger_region_fools_every_overlay_and_scores_as_poisoned():
    model = CornerGatedClassifier().eval()
    generator = torch.Generator().manual_seed(4)
    clean = torch.rand(9, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    overlays = torch.rand(FEW_OVERLAYS, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    inert = torch.rand(FEW_OVERLAYS, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    # Index 0 carries the trigger and 1 to 9 are its clean neighbours. Every
    # input gets the corner as its region, so only the region's content differs.
    inputs = torch.cat([apply_trigger(clean[:1]), clean])  # (10, 3, 32, 32)
    corner = torch.zeros(10, 1, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    corner[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = True
    with torch.inference_mode():
        predicted = model(inputs).argmax(dim=1)  # (10,)
    assert predicted[0] == TARGET_CLASS, "the planted backdoor must fire"

    fooled, avg_conf = overlay_statistics(
        model,
        inputs,
        corner,
        predicted,
        overlays,
        inert,
        IDENTITY_MEAN,
        IDENTITY_STD,
        DEVICE,
        False,
    )

    assert fooled[0].item() == 1.0
    assert fooled[1:].max() < 1.0
    # The envelope is fitted on the clean points alone, as the builder fits it on
    # the clean validation split, and the score is the negated residual exactly
    # as sentinet_scores returns it, so low means poisoned.
    coefficients = fit_decision_boundary(fooled[1:], avg_conf[1:])
    residual = boundary_residual(fooled, avg_conf, coefficients)  # (10,)
    scores = -residual  # (10,)
    assert residual[0] > residual[1:].max()
    assert scores[0] < scores[1:].min()


def score_fixture(model, loaders, seed=0):
    layer, architecture = resolve_cam_site(model)
    overlays = collect_overlay_pixels(
        loaders["validation"], FEW_OVERLAYS, seed, IDENTITY_MEAN, IDENTITY_STD
    )  # (8, 3, 32, 32)
    inert = draw_inert_pixels(FEW_OVERLAYS, tuple(overlays.shape[1:]), seed)
    fooled, avg_conf = sentinet_statistics(
        model,
        loaders["validation"],
        DEVICE,
        IDENTITY_MEAN,
        IDENTITY_STD,
        overlays,
        inert,
        layer,
        architecture,
        False,
    )
    coefficients = fit_decision_boundary(fooled, avg_conf)

    scores = {"validation": -boundary_residual(fooled, avg_conf, coefficients)}
    for split in ("clean", "backdoor"):
        scores[split] = sentinet_scores(
            model,
            loaders[split],
            DEVICE,
            IDENTITY_MEAN,
            IDENTITY_STD,
            overlays,
            inert,
            layer,
            architecture,
            coefficients,
            False,
        )
    return scores


def test_end_to_end_on_the_fixture_is_well_formed_and_deterministic(synthetic_case):
    model, loaders = synthetic_case

    first = score_fixture(model, loaders)
    second = score_fixture(model, loaders)

    for split, scores in first.items():
        assert scores.shape == (32,), split
        assert scores.dtype == torch.float32 and scores.device.type == "cpu"
        assert torch.isfinite(scores).all()
        assert torch.equal(scores, second[split])
    report = detection_report(
        first["validation"], first["clean"], first["backdoor"], HEADLINE_QUANTILE
    )
    # Printed and never asserted. The fixture's trigger reaches the logits by a
    # boolean index on the input, so no token Grad-CAM reads carries it and the
    # region SentiNet transplants is the random ViT's own saliency.
    print(
        f"\nsentinet on the synthetic fixture: auroc {report['auroc']:.3f}, "
        f"direction {report['direction']}, NOT_JUDGEABLE by construction"
    )


def test_no_model_parameter_receives_a_gradient_and_flags_are_restored(
    synthetic_case,
):
    model, loaders = synthetic_case
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    before = [parameter.requires_grad for parameter in parameters]

    score_fixture(model, loaders)

    assert all(parameter.grad is None for parameter in parameters)
    assert [parameter.requires_grad for parameter in parameters] == before
    parameters[0].requires_grad_(True)


def test_the_model_receives_the_native_resolution(synthetic_case):
    """The mask and the composites live at the dataset's resolution and the
    model's Resize upsamples: 16 by 16 images into a wrapper targeting 32 must
    reach the wrapper at 16, the network at 32 and yield a 16 by 16 mask."""
    model, _ = synthetic_case
    network = model.inner[1]
    seen_by_wrapper: list[tuple[int, ...]] = []
    seen_by_network: list[tuple[int, ...]] = []
    handles = [
        model.inner.register_forward_pre_hook(
            lambda _module, inputs: seen_by_wrapper.append(tuple(inputs[0].shape))
        ),
        network.register_forward_pre_hook(
            lambda _module, inputs: seen_by_network.append(tuple(inputs[0].shape))
        ),
    ]
    generator = torch.Generator().manual_seed(5)
    small = torch.rand(4, 3, 16, 16, generator=generator)
    overlays = torch.rand(FEW_OVERLAYS, 3, 16, 16, generator=generator)
    inert = torch.rand(FEW_OVERLAYS, 3, 16, 16, generator=generator)

    try:
        cam, predicted = grad_cam(model.inner, small, DEVICE, False, 1, "vit")
        masks = saliency_mask(cam, (16, 16))  # (4, 1, 16, 16)
        overlay_statistics(
            model.inner,
            small,
            masks,
            predicted,
            overlays,
            inert,
            IDENTITY_MEAN,
            IDENTITY_STD,
            DEVICE,
            False,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert masks.shape == (4, 1, 16, 16)
    assert set(seen_by_wrapper) == {(4, 3, 16, 16), (2 * FEW_OVERLAYS, 3, 16, 16)}
    assert set(seen_by_network) == {(4, 3, 32, 32), (2 * FEW_OVERLAYS, 3, 32, 32)}
