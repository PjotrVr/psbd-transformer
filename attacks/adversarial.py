"""Generating the adversarially perturbed bases the Label-Consistent attack needs.

Turner et al. (2019) poison only target-class images and keep their labels, so the
model can still learn the class from the untouched picture and has no reason to
prefer the trigger. Their fix is to destroy the natural evidence first: an untargeted
attack on each base image against a clean model, so the image stops supporting its
own label and the trigger becomes the only reliable cue. cli.lc_bases runs this
and bases.py reads the result.

The perturbation is a property of the poisoned dataset, not the victim, so 1
surrogate serves every architecture trained afterwards. That is also the faithful
threat model, since the attacker publishes images rather than a model.

Everything works at the dataset's native resolution in 0-to-1 pixel space, because
that is where triggers are applied. The surrogate is fed normalize(x) and its
Resize-to-224 wrapper sits inside the graph.
"""

import hashlib
import json
import os

import torch
import torch.nn.functional as F

BASES_FILENAME = "bases.pt"
MANIFEST_FILENAME = "manifest.json"


def epsilon_tag(epsilon: float) -> str:
    """The strength in the usual units, 16 for 16/255, for use in a directory name."""
    return str(round(epsilon * 255))


def bases_directory(
    results_dir: str, dataset: str, target_label: int, epsilon: float
) -> str:
    """Where the bases for this dataset, target and strength live under results_dir."""
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

    original form (Madry et al., 2018)
        x^{t+1} = Proj_{B_eps(x) ∩ [0,1]} ( x^t + alpha * sign( grad_x L(f(x^t), y) ) )
    descriptive form
        start at a random point of the epsilon ball, take steps of size
        2.5 * epsilon / steps along the sign of the gradient, and after each one
        clip back into the ball and into the valid pixel range

    The step size is Madry's rule, enough total travel to cross the ball with room
    to turn around.
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
    """Top-1 accuracy of the surrogate on these images, before or after perturbing."""
    return float((model(normalize(images)).argmax(dim=1) == labels).float().mean())


def file_digest(path: str) -> str:
    """The sha256 of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bases(
    directory: str, indices: list[int], images: torch.Tensor, manifest: dict
) -> None:
    """Write bases.pt and its manifest, tensor first, both atomically.

    The tensor is regenerable and gitignored. The manifest is tracked, and it is
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
