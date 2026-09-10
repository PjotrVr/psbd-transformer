"""Compute PSNR and SSIM between clean and poisoned images for each attack.

Measures trigger imperceptibility at native resolution (before the 224x224
upscale the model sees). Poison rate does not affect the trigger itself, so
metrics are computed once per (attack, dataset) pair.

That holds because this measures the EVAL trigger. Two attacks now vary their
training trigger per sample, Adaptive-Blend by planting a subset of its pattern
and Label-Consistent by substituting an adversarial base, so measuring
apply_trigger would make the result depend on which indices were sampled.

This duplicates psbd/stealth.py's PSNR and SSIM rather than calling it, which is
worth consolidating; the two are not folded together here because the library
also computes LPIPS and samples differently, so switching would move these
numbers for reasons unrelated to the trigger.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms_v2
from torchvision import datasets as tv_datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attacks import build_attack, default_config
from data.registry import DATASET_REGISTRY

ATTACKS = [
    "badnet_a2o",
    "blend",
    "sig",
    "wanet",
    "lf",
    "lc",
    "bpp",
    "adaptive_blend",
    "tact",
]

DATASETS = ["cifar10", "cifar100", "gtsrb", "tiny"]
MAX_IMAGES = 200
TARGET_LABEL = 0
SEED = 42


# -- image quality metrics ---------------------------------------------------


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = torch.outer(g, g)
    return g / g.sum()


def compute_psnr(clean: torch.Tensor, poisoned: torch.Tensor) -> float:
    """PSNR between two [0,1] CHW images.

    original: PSNR = 10 * log10(MAX^2 / MSE)
    simplified (MAX=1): PSNR = 10 * log10(1 / MSE)
    """
    mse = ((clean - poisoned) ** 2).mean().item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> float:
    """SSIM (Wang et al. 2004) between two [0,1] CHW images.

    original:
        SSIM(x,y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
                     / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
    where C1 = (K1*L)^2, C2 = (K2*L)^2, L = dynamic range, K1 = 0.01, K2 = 0.03
    """
    c1 = 0.01**2
    c2 = 0.03**2

    kernel = gaussian_kernel(window_size, sigma)
    channels = img1.shape[0]
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

    x = img1.unsqueeze(0)
    y = img2.unsqueeze(0)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x**2, kernel, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y**2, kernel, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    return (numerator / denominator).mean().item()


# -- dataset loading ----------------------------------------------------------


def load_raw_dataset(dataset_name: str, raw_data_dir: str = "raw_data"):
    """Load a dataset with Resize + ToTensor only (no normalization).

    Returns images in [0,1] at the attack's native resolution.
    """
    spec = DATASET_REGISTRY[dataset_name]
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((spec.image_size, spec.image_size)),
            transforms_v2.ToTensor(),
        ]
    )

    root = str(Path(raw_data_dir) / dataset_name)

    if spec.loader_kind == "gtsrb":
        return tv_datasets.GTSRB(
            root=root, split="test", download=False, transform=transform
        )

    if spec.loader_kind == "image_folder":
        return tv_datasets.ImageFolder(Path(root) / "val", transform=transform)

    torchvision_cls = {
        "cifar10": tv_datasets.CIFAR10,
        "cifar100": tv_datasets.CIFAR100,
    }[spec.loader_kind]
    return torchvision_cls(root=root, train=False, download=False, transform=transform)


def sample_indices(dataset_size: int, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_size, size=min(n, dataset_size), replace=False)
    return sorted(int(i) for i in indices)


# -- main loop ---------------------------------------------------------------


def measure_attack(
    attack_name: str, dataset_name: str, dataset, image_size: int, indices: list[int]
) -> dict:
    """Compute PSNR and SSIM for one (attack, dataset) pair."""
    config = default_config(attack_name)
    attack = build_attack(attack_name, config, image_size, TARGET_LABEL)

    # Stealth is what a defender sees at inference, so it is the eval trigger.
    # It also keeps these TEST indices away from any train-indexed lookup a
    # training-time trigger holds: Adaptive-Blend plants a per-sample subset of
    # its pattern while training and the whole pattern at eval, and
    # Label-Consistent substitutes an adversarial base only while training.
    # Measuring apply_trigger here reported a different trigger from the one
    # psbd/stealth.py reports, for the same attack.
    plant = attack.apply_trigger_eval or attack.apply_trigger

    psnr_values = []
    ssim_values = []

    for idx in indices:
        clean_img, _ = dataset[idx]
        poisoned_img = plant(clean_img, idx)

        psnr_values.append(compute_psnr(clean_img, poisoned_img))
        ssim_values.append(compute_ssim(clean_img, poisoned_img))

    psnr_arr = np.array(psnr_values)
    ssim_arr = np.array(ssim_values)

    # filter inf PSNR for mean/std (identical images)
    finite_psnr = psnr_arr[np.isfinite(psnr_arr)]

    return {
        "psnr_mean": float(np.mean(finite_psnr))
        if len(finite_psnr) > 0
        else float("inf"),
        "psnr_std": float(np.std(finite_psnr)) if len(finite_psnr) > 0 else 0.0,
        "ssim_mean": float(np.mean(ssim_arr)),
        "ssim_std": float(np.std(ssim_arr)),
        "n_images": len(indices),
        "n_inf_psnr": int(np.sum(~np.isfinite(psnr_arr))),
    }


def main():
    results = {}

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        spec = DATASET_REGISTRY[dataset_name]
        dataset = load_raw_dataset(dataset_name)
        indices = sample_indices(len(dataset), MAX_IMAGES, SEED)

        results[dataset_name] = {}

        for attack_name in ATTACKS:
            print(f"  {attack_name:20s} ... ", end="", flush=True)
            try:
                row = measure_attack(
                    attack_name, dataset_name, dataset, spec.image_size, indices
                )
                results[dataset_name][attack_name] = row
                print(
                    f"PSNR={row['psnr_mean']:.2f} +/- {row['psnr_std']:.2f}  "
                    f"SSIM={row['ssim_mean']:.4f} +/- {row['ssim_std']:.4f}"
                )
            except Exception as e:
                print(f"FAILED: {e}")
                results[dataset_name][attack_name] = {"error": str(e)}

    # save raw results as JSON for later use
    out_path = Path(__file__).resolve().parent / "stealth_metrics_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {out_path}")

    # print summary
    print(f"\n{'=' * 60}")
    print("Summary (averaged across datasets)")
    print(f"{'=' * 60}")
    print(f"{'Attack':20s} {'PSNR':>10s} {'SSIM':>10s}")
    print("-" * 42)
    for attack_name in ATTACKS:
        psnr_vals = []
        ssim_vals = []
        for dataset_name in DATASETS:
            row = results[dataset_name].get(attack_name, {})
            if "error" not in row:
                psnr_vals.append(row["psnr_mean"])
                ssim_vals.append(row["ssim_mean"])
        if psnr_vals:
            print(
                f"{attack_name:20s} {np.mean(psnr_vals):10.2f} {np.mean(ssim_vals):10.4f}"
            )


if __name__ == "__main__":
    main()
