"""Loading and building ViT-B/16 and Swin-S classifiers.

The loaders read checkpoints saved in the BackdoorBench format, a dict with a
model state dict under "model" and the class count under "num_classes". The
builders create fresh networks with the same wrapper, used by training.

Note on Swin: BackdoorBench does not publish Swin backdoored checkpoints, so a
Swin run needs models you train yourself (see train.py). The ViT-specific
accessors below are used only by the ViT feature and weight analysis.
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torchvision.models import (
    Swin_S_Weights,
    ViT_B_16_Weights,
    swin_s,
    vit_b_16,
)


def _wrap_with_resize(network: nn.Module) -> nn.Module:
    """Prepend a Resize so the network accepts inputs of any spatial size."""
    return nn.Sequential(transforms_v2.Resize((224, 224)), network)


def build_vit(num_classes: int) -> nn.Module:
    network = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    network.heads.head = nn.Linear(network.heads.head.in_features, num_classes)
    return _wrap_with_resize(network)


def build_swin(num_classes: int) -> nn.Module:
    network = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
    network.head = nn.Linear(network.head.in_features, num_classes)
    return _wrap_with_resize(network)


def _strip_dataparallel_prefix(state_dict: dict) -> dict:
    """DataParallel checkpoints prefix every key with 'module.'."""
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _load_checkpoint_into(builder, checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = builder(checkpoint["num_classes"])

    state_dict = checkpoint.get("model", checkpoint)
    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()
    state_dict = _strip_dataparallel_prefix(state_dict)

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"Missing keys: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"Unexpected keys: {len(result.unexpected_keys)}")

    return model.to(device).eval()


def load_vit_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    return _load_checkpoint_into(build_vit, checkpoint_path, device)


def load_swin_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    return _load_checkpoint_into(build_swin, checkpoint_path, device)


def load_checkpoint(architecture: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    if architecture == "vit":
        return load_vit_checkpoint(checkpoint_path, device)
    if architecture == "swin":
        return load_swin_checkpoint(checkpoint_path, device)
    raise ValueError(f"Unknown architecture: {architecture}")


# ViT's wrapped vit_b_16 and Swin's wrapped swin_s have structurally distinct
# state_dict key substrings, so a checkpoint's own weights identify its
# architecture even when a folder name gives no hint (or an untrustworthy one).
VIT_STATE_DICT_MARKERS = ("conv_proj", "class_token", "encoder.layers.encoder_layer_")
SWIN_STATE_DICT_MARKERS = ("features.",)


def detect_architecture(checkpoint_path: str) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    keys = list(checkpoint["model"].keys())
    is_vit = any(marker in key for key in keys for marker in VIT_STATE_DICT_MARKERS)
    is_swin = any(marker in key for key in keys for marker in SWIN_STATE_DICT_MARKERS)
    if is_vit and not is_swin:
        return "vit"
    if is_swin and not is_vit:
        return "swin"
    raise ValueError(
        f"state_dict at {checkpoint_path} matched vit={is_vit} swin={is_swin}, expected exactly one"
    )


def vit_core(model: nn.Module) -> nn.Module:
    """Return the ViT network inside the Sequential(Resize, network) wrapper."""
    return model[1] if isinstance(model, nn.Sequential) else model
