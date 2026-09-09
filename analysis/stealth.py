"""Stealth metrics: how visible an attack's trigger is, in pixel space.

PSNR, SSIM, and LPIPS between each clean test image and its triggered self,
measured before normalization (the range where pixel-space triggers are defined,
0 to 1). These answer a different question than ASR or clean accuracy: not how
well the trigger fools the model, but how hard it is for a human to notice, so
they need no model and no checkpoint, only an attack and the raw test images.

Stealth depends only on (dataset_name, attack_name) for the attacks in scope,
because this measures the EVAL trigger, which is built purely from the attack
config and image size, never from poison_rate, target_label, seed, architecture,
or SAM/rho. So the same trigger stamped on the same images yields byte-identical
numbers across all 8 to 16 checkpoints that share a (dataset, attack) pair.
cached_stealth_metrics computes each pair once per process rather than rerunning
LPIPS's AlexNet forward pass once per checkpoint, which is where nearly all of the
cost sits.

The eval trigger is load-bearing here, not incidental. Adaptive-Blend plants a
per-sample SUBSET of its pattern while training and the whole pattern at eval, and
Label-Consistent substitutes an adversarial base only while training. Reading
apply_trigger would make this cache key false for both, since the result would then
depend on which sample indices were drawn.
"""

import lpips
import torch
from torchmetrics.functional import structural_similarity_index_measure

from data.loading import limit_dataset
from evaluation.loaders import load_test_base
from attacks.poisoning import Attack

# Image-quality metrics are precision-sensitive (PSNR is an MSE ratio in dB), so
# they run in float32 regardless of the bfloat16 used for model evaluation.
STEALTH_DTYPE = torch.float32

# One LPIPS network per backbone, reused across every (dataset, attack) pair in a
# run. Building it loads pretrained weights from disk, not worth repeating.
_lpips_models: dict[str, lpips.LPIPS] = {}

_stealth_cache: dict[tuple[str, str], dict] = {}


def build_clean_triggered_pairs(
    dataset_name: str,
    attack: Attack,
    raw_data_dir: str = "raw_data",
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index-aligned (clean, triggered) image pairs, both (N, C, H, W) in 0 to 1.

    Uses load_test_base's pre-normalize tensors directly, and pairs every sampled
    image with its own triggered self, not through AttackSuccessSet's eligibility
    filter: stealth is pixel visibility to a human, which does not depend on an
    image's class or whether the label mode would poison it, so the pool is wider
    and simpler than the ASR-eligible one.
    """
    test_base, _spec = load_test_base(dataset_name, raw_data_dir)
    test_base = limit_dataset(test_base, max_samples, seed)

    clean = torch.stack([test_base[i][0] for i in range(len(test_base))])
    # Stealth is what a defender sees at inference, which is the eval trigger.
    # It also keeps these test indices away from any train-indexed lookup an
    # attack's training-time trigger may hold (Label-Consistent's adversarial bases).
    plant = attack.apply_trigger_eval or attack.apply_trigger
    triggered = torch.stack([plant(test_base[i][0], i) for i in range(len(test_base))])
    assert clean.shape == triggered.shape, (
        "stealth needs index-aligned pairs. A trigger that changes the image shape "
        "would make every per-image metric compare different pixels"
    )
    return clean, triggered


def per_image_psnr(clean: torch.Tensor, triggered: torch.Tensor) -> torch.Tensor:
    """PSNR in dB per image, (N,), data_range 1.0, infinite where the pair is identical.

    original form
        PSNR = 10 * log10(MAX^2 / MSE),  with MAX = 1 in this pixel range
    simplified form
        psnr = 10 * log10(1 / mean_squared_error_over_pixels)
    """
    mse = ((clean - triggered) ** 2).flatten(start_dim=1).mean(dim=1)  # (N,)

    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr


def per_image_ssim(clean: torch.Tensor, triggered: torch.Tensor) -> torch.Tensor:
    """SSIM per image, (N,), higher meaning more similar."""
    ssim = structural_similarity_index_measure(
        triggered, clean, data_range=1.0, reduction="none"
    )
    return ssim


def _lpips_model(backbone: str, device) -> lpips.LPIPS:
    if backbone not in _lpips_models:
        _lpips_models[backbone] = lpips.LPIPS(net=backbone).to(device).eval()
    return _lpips_models[backbone]


def per_image_lpips(
    clean: torch.Tensor, triggered: torch.Tensor, device, backbone: str
) -> torch.Tensor:
    """LPIPS per pair, (N,), lower meaning more similar. Inputs are mapped to -1 to 1.

    The pretrained LPIPS backbones expect that range, so the 0-to-1 pixel tensors
    every trigger works in have to be rescaled here rather than at the call site.
    """
    model = _lpips_model(backbone, device)

    with torch.no_grad():
        distances = model(triggered * 2 - 1, clean * 2 - 1)  # (N, 1, 1, 1)

    per_pair = distances.flatten()  # (N,)
    return per_pair


def _mean_std(values: torch.Tensor) -> tuple[float, float]:
    return float(values.mean()), float(values.std())


def compute_stealth_metrics(
    clean: torch.Tensor,
    triggered: torch.Tensor,
    device,
    batch_size: int = 64,
    lpips_backbone: str = "alex",
    max_samples: int | None = None,
    seed: int = 0,
) -> dict:
    """PSNR/SSIM/LPIPS between pair-aligned clean and triggered images, pixel space.

    Per-image reductions (SSIM reduction='none', PSNR and LPIPS computed one image
    at a time) so the reported std is a real spread over images, not the std of
    batch means. Batched to bound memory on large test sets like Tiny ImageNet.
    """
    clean = clean.to(STEALTH_DTYPE)
    triggered = triggered.to(STEALTH_DTYPE)

    psnr_chunks, ssim_chunks, lpips_chunks = [], [], []
    for start in range(0, len(clean), batch_size):
        clean_batch = clean[start : start + batch_size].to(device)
        triggered_batch = triggered[start : start + batch_size].to(device)
        psnr_chunks.append(per_image_psnr(clean_batch, triggered_batch).cpu())
        ssim_chunks.append(per_image_ssim(clean_batch, triggered_batch).cpu())
        lpips_chunks.append(
            per_image_lpips(clean_batch, triggered_batch, device, lpips_backbone).cpu()
        )

    psnr = torch.cat(psnr_chunks)  # (N,)
    ssim = torch.cat(ssim_chunks)  # (N,)
    lpips_values = torch.cat(lpips_chunks)  # (N,)

    psnr_mean, psnr_std = _mean_std(psnr)
    ssim_mean, ssim_std = _mean_std(ssim)
    lpips_mean, lpips_std = _mean_std(lpips_values)
    metrics = {
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "lpips_backbone": lpips_backbone,
        "data_range": 1.0,
        "n_pairs": int(len(clean)),
        "max_samples": max_samples,
        "seed": seed,
    }
    return metrics


def cached_stealth_metrics(
    dataset_name: str,
    attack_name: str,
    attack: Attack,
    raw_data_dir: str,
    device,
    batch_size: int = 64,
    lpips_backbone: str = "alex",
    max_samples: int | None = None,
    seed: int = 0,
) -> dict:
    """compute_stealth_metrics memoized on (dataset_name, attack_name).

    Safe as a module-level cache because the evaluation sweep is a single process,
    single pass, no concurrency, exiting when the run finishes.
    """
    key = (dataset_name, attack_name)
    if key not in _stealth_cache:
        clean, triggered = build_clean_triggered_pairs(
            dataset_name, attack, raw_data_dir, max_samples=max_samples, seed=seed
        )
        _stealth_cache[key] = compute_stealth_metrics(
            clean,
            triggered,
            device,
            batch_size=batch_size,
            lpips_backbone=lpips_backbone,
            max_samples=max_samples,
            seed=seed,
        )

    return _stealth_cache[key]
