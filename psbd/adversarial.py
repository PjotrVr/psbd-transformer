"""Adversarially perturbed base images for the Label-Consistent attack.

Turner et al. (2019) poison only target-class images and keep their labels, so
the model can still learn the class from the untouched picture and has no reason
to prefer the trigger. Their fix is to first destroy the natural evidence: run an
untargeted attack on each base image against a model trained on clean data, so
the image no longer supports its own label and the trigger becomes the only cue
that reliably does. The PSBD paper we are porting used exactly this, taking
precomputed adversarial images from the original authors and from BackdoorBench.

The perturbation is a property of the poisoned DATASET, not of the victim, so a
single surrogate serves every architecture we then train. That is also the
faithful threat model: the attacker publishes images, not a model.

Everything here works at the dataset's NATIVE resolution in 0-to-1 pixel space,
because that is where triggers are applied. `train_backdoor.base_transform` stops
at ToTensor, the trigger goes on, and normalization comes last, so the surrogate
is fed `normalize(x)` and its Resize-to-224 wrapper is inside the graph.
"""

import hashlib
import json
import os

import torch
import torch.nn.functional as F

BASES_FILENAME = "bases.pt"
MANIFEST_FILENAME = "manifest.json"


def epsilon_tag(epsilon: float) -> str:
    """`16` for 16/255, so a directory name says the strength in the usual units."""
    return str(round(epsilon * 255))


def bases_directory(
    results_dir: str, dataset: str, target_label: int, epsilon: float
) -> str:
    return os.path.join(
        results_dir,
        "lc_adversarial",
        f"{dataset}_tl{target_label}_eps{epsilon_tag(epsilon)}",
    )


def pgd_perturb(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    normalize,
    epsilon: float,
    steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Untargeted L-inf PGD, maximizing the loss on each image's own label.

    original (Madry et al., 2018):
        x^{t+1} = Proj_{B_eps(x) ∩ [0,1]} ( x^t + alpha * sign( grad_x L(f(x^t), y) ) )
    simplified: start at a random point of the epsilon-ball, take `steps` steps of
    size 2.5 * epsilon / steps along the sign of the gradient, and after each step
    clip back into the ball and into the valid pixel range.

    The step size is Madry's rule: 2.5 * epsilon / steps gives enough total travel
    to reach the far side of the ball with room to turn around.
    """
    step_size = 2.5 * epsilon / steps
    noise = torch.empty_like(images).uniform_(-epsilon, epsilon, generator=generator)
    delta = (images + noise).clamp(0.0, 1.0) - images
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(normalize(images + delta)), labels)
        (gradient,) = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + step_size * gradient.sign()).clamp(-epsilon, epsilon)
        delta = (images + delta).clamp(0.0, 1.0) - images
    return (images + delta).detach()


@torch.no_grad()
def accuracy_on(model, images: torch.Tensor, labels: torch.Tensor, normalize) -> float:
    return float((model(normalize(images)).argmax(dim=1) == labels).float().mean())


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bases(
    directory: str, indices: list[int], images: torch.Tensor, manifest: dict
) -> None:
    """Write bases.pt and its manifest, tensor first, both atomically.

    The tensor is regenerable and gitignored; the manifest is tracked, and it is
    what makes a poisoned checkpoint traceable to the exact bases it trained on.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, BASES_FILENAME)
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(
        {"indices": torch.tensor(indices, dtype=torch.long), "images": images},
        temporary,
    )
    os.replace(temporary, path)

    manifest = dict(manifest, bases_sha256=file_digest(path), n_images=len(indices))
    manifest_path = os.path.join(directory, MANIFEST_FILENAME)
    temporary = f"{manifest_path}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary, manifest_path)
