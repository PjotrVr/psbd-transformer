"""Building and loading the ViT-B/16 and Swin-S classifiers.

Both builders return nn.Sequential(Resize(224), network). The Resize front-end
lets the same model accept CIFAR's 32x32 and Tiny ImageNet's 64x64 without the
caller upscaling first, which keeps pixel-space triggers at their native
resolution all the way to the model boundary. Anything that resolves a dotted
module path (dropout placement, feature hooks) must go through network_core to
get past that wrapper.

Checkpoints are read in the BackdoorBench format: a dict with the model state
dict under "model" and the class count under "num_classes".

Note on Swin: BackdoorBench does not publish Swin backdoored checkpoints, so a
Swin run needs models trained in this repo.
"""

from typing import Callable

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torchvision.models import (
    Swin_S_Weights,
    ViT_B_16_Weights,
    swin_s,
    vit_b_16,
)

# Both backbones are ImageNet-pretrained at 224x224 and their patch embeddings
# assume it, so the wrapper resizes to this rather than the dataset's own size.
MODEL_INPUT_SIZE = 224


def build_vit(
    num_classes: int, dropout: float = 0.0, attention_dropout: float = 0.0
) -> nn.Module:
    """ViT-B/16 with the ImageNet head replaced by a fresh num_classes head.

    dropout defaults to 0.0, which is torchvision's default and PSBD's requirement:
    the paper trains without dropout and applies it only at inference, and every
    checkpoint in this project was built that way. The arguments exist to test the
    requirement rather than assume it. Dropout carries no parameters, so the
    pretrained weights load unchanged at any value.
    """
    network = vit_b_16(
        weights=ViT_B_16_Weights.IMAGENET1K_V1,
        dropout=dropout,
        attention_dropout=attention_dropout,
    )
    network.heads.head = nn.Linear(network.heads.head.in_features, num_classes)

    resized_model = nn.Sequential(
        transforms_v2.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)), network
    )
    return resized_model


def build_swin(num_classes: int) -> nn.Module:
    """Swin-S with an ImageNet head replaced by a fresh num_classes head."""
    network = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
    network.head = nn.Linear(network.head.in_features, num_classes)

    resized_model = nn.Sequential(
        transforms_v2.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)), network
    )
    return resized_model


ARCHITECTURE_BUILDERS: dict[str, Callable[[int], nn.Module]] = {
    "vit": build_vit,
    "swin": build_swin,
}


def load_checkpoint(
    architecture: str, checkpoint_path: str, device: torch.device, strict: bool = True
) -> nn.Module:
    """The named architecture with a checkpoint's weights loaded, on device, in eval mode.

    strict defaults on and that is load-bearing. Both builders start from ImageNet
    weights, so a key mismatch under strict=False leaves a working ImageNet backbone
    with a random head, and every downstream tool runs happily on a model with no
    backdoor in it. strict=False stays available for BackdoorBench checkpoints,
    whose key layout predates the Sequential(Resize, network) wrapper.
    """
    if architecture not in ARCHITECTURE_BUILDERS:
        raise ValueError(f"Unknown architecture: {architecture}")
    builder = ARCHITECTURE_BUILDERS[architecture]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = builder(checkpoint["num_classes"])

    state_dict = checkpoint.get("model", checkpoint)
    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()

    # DataParallel checkpoints prefix every key with "module.".
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    result = model.load_state_dict(state_dict, strict=strict)
    if result.missing_keys or result.unexpected_keys:
        print(
            f"{checkpoint_path}: {len(result.missing_keys)} missing, "
            f"{len(result.unexpected_keys)} unexpected keys"
        )

    loaded_model = model.to(device).eval()
    return loaded_model


# The 2 backbones have distinct state_dict key substrings, so a checkpoint's own
# weights identify its architecture when the folder name does not.
VIT_STATE_DICT_MARKERS = ("conv_proj", "class_token", "encoder.layers.encoder_layer_")
SWIN_STATE_DICT_MARKERS = ("features.",)


def detect_architecture(checkpoint_path: str) -> str:
    """The architecture a checkpoint's state_dict keys identify.

    Raises when the keys match both marker sets or neither, since a checkpoint that
    cannot be identified must not be loaded as a guess.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    keys = list(checkpoint.get("model", checkpoint).keys())

    is_vit = any(marker in key for key in keys for marker in VIT_STATE_DICT_MARKERS)
    is_swin = any(marker in key for key in keys for marker in SWIN_STATE_DICT_MARKERS)

    if is_vit and not is_swin:
        return "vit"
    if is_swin and not is_vit:
        return "swin"
    raise ValueError(
        f"state_dict at {checkpoint_path} matched vit={is_vit} swin={is_swin}, expected exactly one"
    )


def network_core(model: nn.Module) -> nn.Module:
    """The classifier network inside the Sequential(Resize, network) wrapper.

    Dotted module paths resolve from here, not from the wrapper, whose only children
    are the Resize and the network.
    """
    if isinstance(model, nn.Sequential):
        network = model[1]
        return network
    return model
