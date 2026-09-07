"""A model with a backdoor we installed ourselves, so the right answer is known.

Every check in this package needs a case whose answer is not in question. A real
checkpoint cannot serve, because when a detector disagrees with a real checkpoint
there is no way to tell whether the detector is broken or the checkpoint is
unusual. Here the backdoor is installed by hand, its trigger is known, its target
is known, and its strength is chosen to be unmissable.

The point is to make a failure mean exactly one thing: the code is wrong.

Runs on CPU in seconds, so a check built on it belongs in the test suite rather
than in a cluster job.
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from torchvision.models import VisionTransformer

IMAGE_SIZE = 32
TRIGGER_SIZE = 6
TARGET_CLASS = 0
NUM_CLASSES = 10

# Large enough that the trigger dominates any content signal, so the backdoor is
# unmissable and a detector that misses it is broken rather than unlucky.
BACKDOOR_LOGIT = 12.0


def trigger_pattern():
    """A checkerboard, so the trigger is a SHAPE rather than a brightness level."""
    row = torch.arange(TRIGGER_SIZE)
    board = ((row[:, None] + row[None, :]) % 2).float()  # (TRIGGER_SIZE, TRIGGER_SIZE)
    return board


def apply_trigger(images):
    triggered = images.clone()
    triggered[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = trigger_pattern()
    return triggered


def has_trigger(images):
    """Detect the trigger by its SHAPE, invariant to any positive rescaling.

    A threshold test like "is this corner bright" is defeated by the very
    perturbations the detectors apply: SCALE-UP multiplies pixel values, so a
    clean corner crosses any fixed brightness threshold and the backdoor fires on
    clean data. That produced a detector reading 0.023 here and it was the
    synthetic model at fault, not the detector.

    Normalizing the corner by its own maximum removes the scale, so the check
    asks the question the trigger actually poses: is this pattern present.
    """
    corner = images[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:]  # (batch, 3, T, T)
    peak = corner.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    normalized = corner / peak

    reference = trigger_pattern().to(images.device)  # (T, T)
    deviation = (normalized - reference).abs().amax(dim=(1, 2, 3))  # (batch,)

    present = deviation < 0.25
    return present


def build_backdoored_model(seed=0):
    torch.manual_seed(seed)
    network = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=NUM_CLASSES,
    )
    # torchvision zero-initializes the head, so an untrained model returns the same
    # logits for every input and every assertion made on them holds vacuously.
    nn.init.normal_(network.heads.head.weight, std=0.5)
    nn.init.normal_(network.heads.head.bias, std=0.5)

    model = nn.Sequential(transforms_v2.Resize((IMAGE_SIZE, IMAGE_SIZE)), network)
    backdoored = BackdooredModel(model)
    return backdoored


class BackdooredModel(nn.Module):
    """Wraps a clean model and routes triggered inputs to the target class.

    The backdoor is added at the logits rather than trained in, because a trained
    one takes minutes and would make the expected answer approximate. This one is
    exact: a triggered input gets BACKDOOR_LOGIT added to the target class and
    nothing else changes.

    It is deliberately robust to activation perturbation, since the trigger is
    read from the INPUT rather than from any intermediate feature. That is the
    property every perturbation-consistency detector exists to find, so a detector
    that cannot find it here cannot find it anywhere.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        logits = self.inner(x)  # (batch, num_classes)
        triggered = has_trigger(x)  # (batch,)

        boost = torch.zeros_like(logits)
        boost[triggered, TARGET_CLASS] = BACKDOOR_LOGIT
        return logits + boost


def build_splits(num_samples=256, batch_size=64, seed=0):
    """Validation, clean and backdoor loaders over the same images, paired by index."""
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(seed)
    validation = torch.rand(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    clean = torch.rand(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)

    # A trigger stamped on the SAME images the clean split holds, which is the
    # pairing every paired statistic assumes.
    backdoor = apply_trigger(clean)

    labels = torch.randint(0, NUM_CLASSES, (num_samples,), generator=generator)
    loaders = {
        "validation": DataLoader(
            TensorDataset(validation, labels), batch_size=batch_size
        ),
        "clean": DataLoader(TensorDataset(clean, labels), batch_size=batch_size),
        "backdoor": DataLoader(
            TensorDataset(backdoor, torch.full_like(labels, TARGET_CLASS)),
            batch_size=batch_size,
        ),
    }
    return loaders


def attack_success_rate(model, loaders, device):
    """Fraction of triggered inputs the model sends to the target class."""
    predictions = []
    with torch.inference_mode():
        for images, _ in loaders["backdoor"]:
            predictions.append(model(images.to(device)).argmax(dim=1).cpu())

    rate = (torch.cat(predictions) == TARGET_CLASS).float().mean().item()
    return rate
