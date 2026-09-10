"""Reading the adversarially perturbed base images Label-Consistent stamps onto.

adversarial.py writes them in a separate offline step. This is the read side: a
dataset index maps to the image the attack should stamp instead of the clean one.

Loading is deferred until the first lookup, because build() runs on every consumer
of a checkpoint, including detection and analysis processes that only ever call
apply_trigger_eval and may run where the cache does not exist.
"""

import os

import torch

BASES_FILENAME = "bases.pt"


def read_adversarial_bases(directory: str) -> dict[int, torch.Tensor]:
    """The bases.pt in directory as an index-to-image map, or {} if absent."""
    if not directory:
        return {}
    path = os.path.join(directory, BASES_FILENAME)
    if not os.path.exists(path):
        return {}
    payload = torch.load(path, map_location="cpu")
    indices = payload["indices"].tolist()
    images = payload["images"]
    return {int(index): images[position] for position, index in enumerate(indices)}


def lazy_adversarial_lookup(directory: str, image_size: int):
    """A lookup from dataset index to base image (None on a miss) that loads lazily.

    The stored images are at the dataset's native resolution, the space triggers are
    applied in. A size mismatch means the cache was built for a different dataset,
    so it raises rather than resizing into a silently wrong experiment.
    """
    cache: dict[int, torch.Tensor] = {}
    state = {"loaded": False}

    def lookup(index: int):
        if not state["loaded"]:
            cache.update(read_adversarial_bases(directory))
            state["loaded"] = True
            for base in cache.values():
                if base.shape[-1] != image_size:
                    raise ValueError(
                        f"adversarial bases in {directory} are {base.shape[-1]}px "
                        f"but the attack is built for {image_size}px"
                    )
                break
        return cache.get(int(index))

    return lookup


def missing_adversarial_bases(config, poison_indices) -> list[int]:
    """Poisoned indices with no perturbed base, empty when the config asks for none.

    apply_trigger falls back to the clean image on a miss so that analysis code
    holding test indices does not crash. That fallback must never fire in training,
    where it would quietly train the patch-only variant while args.json records the
    adversarial variant, so training calls this first and refuses.
    """
    directory = getattr(config, "adversarial_dir", "")
    if not directory:
        return []
    available = set(read_adversarial_bases(directory))
    return sorted(index for index in poison_indices if int(index) not in available)


def adversarial_config_error(config) -> str | None:
    """Why the config's adversarial settings are incoherent, or None if they are fine.

    An epsilon with no directory is the shape a dropped command-line override
    produces. Nothing else notices it, since it is in range and well typed. The
    attack would fall back to its patch-only variant while the recorded metadata
    says otherwise.
    """
    if getattr(config, "adversarial_epsilon", 0.0) and not getattr(
        config, "adversarial_dir", ""
    ):
        return (
            "adversarial_epsilon is set but adversarial_dir is empty, so no perturbed "
            "bases would be used. Pass both in ONE --attack-override flag."
        )
    return None
