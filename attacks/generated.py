"""Adapter for attacks whose trigger is a pregenerated per-sample image.

Sample-specific attacks such as SSBA produce their perturbation with a trained
generator that is impractical to reproduce here, and TrojanNN optimizes its patch
against a specific model. BackdoorBench distributes the resulting poisoned images.
This adapter serves those images by dataset index, so any precomputed poisoning
plugs into the same Attack interface as the deterministic attacks.

The directory is expected to hold index-named images, matching how the poisoned
test set is already stored, so the file 42.png is the poisoned version of the
clean sample at index 42.
"""

import glob
from dataclasses import dataclass
from pathlib import Path

import torchvision.transforms.v2 as transforms_v2
from PIL import Image

from poison import Attack


@dataclass(frozen=True)
class GeneratedConfig:
    poisoned_dir: str
    name: str = "generated"
    label_mode: str = "all_to_one"


def _index_to_path(poisoned_dir: str) -> dict[int, str]:
    paths = glob.glob(f"{poisoned_dir}/**/*.png", recursive=True)
    return {int(Path(path).stem): path for path in paths}


def build(config: GeneratedConfig, image_size: int, target_label: int) -> Attack:
    index_to_path = _index_to_path(config.poisoned_dir)
    to_tensor = transforms_v2.Compose(
        [transforms_v2.Resize((image_size, image_size)), transforms_v2.ToTensor()]
    )

    def apply_trigger(_image, index: int):
        # The stored image already carries the trigger, so the clean image passed
        # in is ignored and the pregenerated poisoned image is returned.
        stored = Image.open(index_to_path[index]).convert("RGB")
        return to_tensor(stored)

    return Attack(config.name, apply_trigger, config.label_mode, target_label)
