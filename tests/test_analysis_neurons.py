"""The neuron statistics against BackdoorBench's own lines, on a ConvNet and a toy ViT.

Upstream's unit is a ConvNet channel with a (height, width) map and ours a
residual dimension with a token axis, so every 4-d transcription is applied to
a (batch, channels, height, width) tensor and ours to the same numbers viewed
as (batch, height * width, channels). The toy ViT then checks the token
mapping end to end: the sum path equals the flatten path reshaped, the
streamed TAC equals the whole-tensor TAC and the class purity matches the
upstream loop over the classes it emits.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torchvision.models import VisionTransformer

from analysis.direction import trigger_activation_change
from analysis.features import captured_layers
from analysis.neurons import (
    block_count,
    class_purity,
    layer_activations,
    mean_activation,
    paired_layer_tac,
    token_sum,
    top_activating_indices,
    top_k_rule,
)
from tests.reference import backdoorbench as upstream

DEVICE = torch.device("cpu")
SYNTHETIC_CLASSES = 6


def channels_last_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """(batch, C, H, W) viewed the way a ViT block emits it, (batch, H * W, C)."""
    return feature_map.flatten(2).transpose(1, 2)


@pytest.fixture(scope="module")
def conv_features() -> tuple[torch.Tensor, torch.Tensor]:
    """Paired clean and triggered feature maps of a tiny ConvNet, (16, 6, 5, 5) each."""
    torch.manual_seed(0)
    network = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 6, 3, padding=1)
    )
    clean = torch.rand(16, 3, 5, 5)
    triggered = clean.clone()
    triggered[:, :, :2, :2] = 1.0
    with torch.inference_mode():
        return network(clean).clone(), network(triggered).clone()


@pytest.fixture(scope="module")
def tiny_vit() -> nn.Module:
    """A ViT with the real block structure at tiny width, 2 blocks, 5 tokens."""
    torch.manual_seed(0)
    network = VisionTransformer(
        image_size=32,
        patch_size=16,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=SYNTHETIC_CLASSES,
    )
    return nn.Sequential(transforms_v2.Resize((32, 32)), network).eval()


def test_token_sum_is_upstream_reduction_sum(conv_features):
    clean, _ = conv_features
    ours = token_sum(channels_last_tokens(clean))
    theirs = upstream.reduce_sum(clean)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_token_sum_equals_the_flatten_path_reshaped(conv_features):
    clean, _ = conv_features
    batch, channels = clean.shape[:2]
    flattened = upstream.reduce_flatten(clean)  # (batch, C * H * W)
    reshaped_sum = flattened.reshape(batch, channels, -1).sum(dim=2)  # (batch, C)
    assert torch.allclose(
        token_sum(channels_last_tokens(clean)), reshaped_sum, atol=1e-6
    )


def test_l1_sum_tac_is_visual_tac_line_for_line(conv_features):
    clean, triggered = conv_features
    ours = trigger_activation_change(
        channels_last_tokens(clean), channels_last_tokens(triggered), norm="l1_sum"
    )
    theirs = upstream.tac(triggered.numpy().copy(), clean.numpy().copy())
    assert np.allclose(ours.numpy(), theirs, atol=1e-6)


def test_l2_tac_is_the_per_sample_norm_averaged(conv_features):
    clean, triggered = conv_features
    ours = trigger_activation_change(
        channels_last_tokens(clean), channels_last_tokens(triggered), norm="l2"
    )
    per_sample = (triggered - clean).flatten(2).norm(dim=2)  # (batch, C)
    assert torch.allclose(ours, per_sample.mean(dim=0), atol=1e-6)


def test_the_2_tac_conventions_differ_on_a_multi_position_map(conv_features):
    clean, triggered = conv_features
    l1 = trigger_activation_change(
        channels_last_tokens(clean), channels_last_tokens(triggered), "l1_sum"
    )
    l2 = trigger_activation_change(
        channels_last_tokens(clean), channels_last_tokens(triggered), "l2"
    )
    assert not torch.allclose(l1, l2)


def test_an_unknown_norm_is_refused(conv_features):
    clean, triggered = conv_features
    with pytest.raises(ValueError, match="norm"):
        trigger_activation_change(
            channels_last_tokens(clean), channels_last_tokens(triggered), "l3"
        )


def test_mean_activation_and_order_match_visual_na(conv_features):
    clean, _ = conv_features
    features = token_sum(channels_last_tokens(clean))
    ours = mean_activation(features).numpy()
    theirs = upstream.neuron_activation_average(features.numpy())
    assert np.allclose(ours, theirs, atol=1e-6)
    assert np.array_equal(
        np.argsort(ours)[::-1], upstream.neuron_activation_order(theirs)
    )


def test_top_activating_indices_match_visual_act(conv_features):
    clean, _ = conv_features
    features = token_sum(channels_last_tokens(clean))
    ours = top_activating_indices(features, 5).numpy()
    theirs = upstream.top_indx(features.numpy())[:5]
    assert np.array_equal(ours, theirs)


def test_top_k_rule_is_the_num_image_rule():
    no_poison = np.zeros(40)
    assert top_k_rule(40, 4, 0) == upstream.num_image_rule(40, 4, no_poison) == 10
    some_poison = np.zeros(40)
    some_poison[:7] = 1
    assert top_k_rule(40, 4, 7) == upstream.num_image_rule(40, 4, some_poison) == 7


def test_class_purity_matches_the_actdist_loop_on_the_classes_it_emits(conv_features):
    clean, _ = conv_features
    features = token_sum(channels_last_tokens(clean))
    generator = torch.Generator().manual_seed(3)
    labels = torch.randint(0, SYNTHETIC_CLASSES, (16,), generator=generator)
    poison_mask = torch.zeros(16, dtype=torch.bool)
    poison_mask[[1, 4, 9]] = True
    k = top_k_rule(16, SYNTHETIC_CLASSES, 3)

    ours = class_purity(
        top_activating_indices(features, k), labels, poison_mask, SYNTHETIC_CLASSES
    )
    label_set, theirs = upstream.activation_distribution(
        features.numpy(),
        labels.numpy(),
        poison_mask.numpy().astype(int),
        SYNTHETIC_CLASSES,
        k,
    )
    assert np.allclose(ours[:, label_set].numpy(), theirs, atol=1e-6)
    assert torch.allclose(ours.sum(dim=1), torch.ones(ours.shape[0]))
    assert SYNTHETIC_CLASSES in label_set


def test_class_purity_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        class_purity(
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(4),
            torch.zeros(3, dtype=torch.bool),
            2,
        )


def test_layer_activations_reads_the_captured_block_output(tiny_vit):
    images = torch.rand(6, 3, 32, 32, generator=torch.Generator().manual_seed(4))
    with captured_layers(tiny_vit, (2,), "vit") as captured, torch.inference_mode():
        tiny_vit(images)
        expected = captured[2].clone()  # (6, 5, 16)

    sequence = layer_activations(tiny_vit, images, 2, DEVICE, 4, False, "none", "vit")
    summed = layer_activations(tiny_vit, images, 2, DEVICE, 4, False, "sum", "vit")
    assert torch.allclose(sequence, expected, atol=1e-6)
    assert torch.allclose(summed, expected.sum(dim=1), atol=1e-5)
    assert torch.allclose(
        summed, upstream.reduce_sum(expected.transpose(1, 2)), atol=1e-5
    )


def test_layer_activations_refuses_an_unknown_reduction(tiny_vit):
    with pytest.raises(ValueError, match="reduction"):
        layer_activations(
            tiny_vit, torch.rand(2, 3, 32, 32), 1, DEVICE, 2, False, "max", "vit"
        )


def test_streamed_tac_equals_whole_tensor_tac_at_every_layer(tiny_vit):
    generator = torch.Generator().manual_seed(5)
    clean = torch.rand(7, 3, 32, 32, generator=generator)
    triggered = clean.clone()
    triggered[:, :, -6:, -6:] = 1.0

    streamed = paired_layer_tac(
        tiny_vit, clean, triggered, (0, 1, 2), DEVICE, 3, False, "l1_sum", "vit"
    )
    for layer in (0, 1, 2):
        clean_tokens = layer_activations(
            tiny_vit, clean, layer, DEVICE, 7, False, "none", "vit"
        )
        triggered_tokens = layer_activations(
            tiny_vit, triggered, layer, DEVICE, 7, False, "none", "vit"
        )
        whole = trigger_activation_change(clean_tokens, triggered_tokens, "l1_sum")
        assert torch.allclose(streamed[layer], whole, atol=1e-5), f"layer {layer}"
        assert streamed[layer].shape == (16,)


def test_streamed_tac_refuses_unpaired_images(tiny_vit):
    with pytest.raises(ValueError, match="index-aligned"):
        paired_layer_tac(
            tiny_vit,
            torch.rand(3, 3, 32, 32),
            torch.rand(2, 3, 32, 32),
            (1,),
            DEVICE,
            2,
            False,
        )


def test_block_count_reads_the_model(tiny_vit):
    assert block_count(tiny_vit, "vit") == 2
    assert block_count(tiny_vit) == 2
