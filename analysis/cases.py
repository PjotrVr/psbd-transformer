"""One call from a checkpoint folder name to its clean and backdoor latent features.

Every latent question starts the same way: pick a trained model, build the same
test images with and without its trigger, and read the residual stream at every
block. That setup is 6 objects and a dozen arguments, which is fine inside a
script and hostile inside a notebook. This module collapses it to one call.

The checkpoint's own args.json supplies the dataset, attack, target label and
architecture, so the caller names a folder and nothing else. That is deliberate:
restating the attack by hand is how a feature analysis ends up probing a model
with a trigger it never learned, which produces a plausible number rather than an
error.
"""

from dataclasses import dataclass

import torch

from attacks import apply_config_overrides, build_attack, default_config
from data.registry import DATASET_REGISTRY
from models.backbones import detect_architecture, load_checkpoint
from data.splits import read_checkpoint_metadata, resolve_probe_attack

from .distribution import layer_distribution_table
from .features import default_reduction, extract_layer_features
from .latent import build_paired_loaders

CHECKPOINT_FILENAME = "attack_result.pt"


@dataclass(frozen=True)
class LatentCase:
    """Everything the distribution tools need about one checkpoint, already loaded.

    clean_features and backdoor_features are index-aligned layer by layer: row i
    of each is the same test image, once clean and once triggered. The paired
    statistics in distribution.py depend on that alignment.
    """

    folder: str
    architecture: str
    dataset: str
    attack: str
    target_label: int
    clean_features: dict[int, torch.Tensor]
    backdoor_features: dict[int, torch.Tensor]
    clean_labels: torch.Tensor

    @property
    def layers(self) -> list[int]:
        """Layer indices present, 0 for the block stack's input and 1..N per block."""
        indices = sorted(self.clean_features)
        return indices


def collect_labels(loader) -> torch.Tensor:
    """The loader's labels in served order, (num_samples,).

    Read from the clean loader, so these are the true classes rather than the
    attack's relabelled ones.
    """
    labels = torch.cat([batch_labels for _, batch_labels in loader])
    return labels


def load_latent_case(
    folder: str,
    samples: int = 1000,
    batch_size: int = 64,
    checkpoint_root: str = "checkpoints",
    raw_data_dir: str = "raw_data",
    device: torch.device | None = None,
    seed: int = 0,
    reduction: str | None = None,
    probe_attack: str | None = None,
    probe_target_label: int | None = None,
    use_bfloat16: bool = True,
) -> LatentCase:
    """Load a checkpoint and extract paired clean and backdoor per-layer features.

    folder is a name under checkpoint_root, for example
    "vit_cifar100_badnet_a2o_0_01". A benign checkpoint has no trigger of its own,
    so it needs probe_attack to say which trigger to probe it with; that is the
    negative control, and chance-level separation is the expected result.

    reduction defaults to whatever the architecture's head reads, the class token
    for ViT and the token mean for Swin.

    use_bfloat16 is exposed rather than fixed because autocast only engages on
    CUDA, so a GPU run and a CPU run of the same call are not the same
    computation. The per-sample paired difference is the part that suffers; pass
    False when comparing across devices or when a very weak trigger makes that
    difference the whole signal.
    """
    checkpoint_path = f"{checkpoint_root}/{folder}/{CHECKPOINT_FILENAME}"
    metadata = read_checkpoint_metadata(checkpoint_path)

    # The state dict is the only description of the architecture that cannot be
    # stale, so it wins over the folder name and over args.json.
    architecture = detect_architecture(checkpoint_path)
    attack_name, target_label = resolve_probe_attack(
        metadata, probe_attack, probe_target_label
    )
    dataset = metadata["dataset"]

    resolved_device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    resolved_reduction = (
        reduction if reduction is not None else default_reduction(architecture)
    )

    image_size = DATASET_REGISTRY[dataset].image_size
    # Rebuild the trigger the checkpoint was actually trained with. A probe attack
    # is chosen here rather than trained, so it takes no override; a backdoored
    # checkpoint rebuilds its own recorded config. Without this a run trained at a
    # non-default trigger is probed with the default one, which is the failure this
    # module's docstring warns about: a plausible number rather than an error.
    overrides = (
        metadata.get("attack_config_overrides")
        if attack_name == metadata.get("attack")
        else None
    )
    attack = build_attack(
        attack_name,
        apply_config_overrides(default_config(attack_name), overrides),
        image_size,
        target_label,
    )
    clean_loader, backdoor_loader = build_paired_loaders(
        dataset, attack, raw_data_dir, batch_size, samples, seed
    )

    model = load_checkpoint(architecture, checkpoint_path, resolved_device)
    clean_features = extract_layer_features(
        model,
        clean_loader,
        resolved_device,
        use_bfloat16=use_bfloat16,
        reduction=resolved_reduction,
        architecture=architecture,
    )
    backdoor_features = extract_layer_features(
        model,
        backdoor_loader,
        resolved_device,
        use_bfloat16=use_bfloat16,
        reduction=resolved_reduction,
        architecture=architecture,
    )

    case = LatentCase(
        folder=folder,
        architecture=architecture,
        dataset=dataset,
        attack=attack_name,
        target_label=target_label,
        clean_features=clean_features,
        backdoor_features=backdoor_features,
        clean_labels=collect_labels(clean_loader),
    )
    return case


def case_distribution_table(case: LatentCase) -> list[dict[str, float]]:
    """The per-layer distribution table for a loaded case, target label included."""
    table = layer_distribution_table(
        case.clean_features,
        case.backdoor_features,
        case.clean_labels,
        case.target_label,
    )
    return table
