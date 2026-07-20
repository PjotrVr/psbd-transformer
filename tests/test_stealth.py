"""Stealth metrics: agreement with skimage, sanity direction, and cache behaviour.

PSNR and SSIM are validated against skimage.metrics (the reference the
literature reports), the identity trigger pins the "no change at all" corner
(PSNR infinite, SSIM 1, LPIPS 0), and the cache test confirms a repeated
(dataset, attack) key does not rerun the LPIPS forward pass.
"""

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

import stealth
from poison import Attack
from stealth import (
    _per_image_psnr,
    _per_image_ssim,
    cached_stealth_metrics,
    compute_stealth_metrics,
)


@pytest.fixture
def pairs():
    generator = torch.Generator().manual_seed(0)
    clean = torch.rand(5, 3, 32, 32, generator=generator)
    triggered = (clean + 0.05 * torch.rand(5, 3, 32, 32, generator=generator)).clamp(
        0, 1
    )
    return clean, triggered


def test_psnr_agrees_with_skimage(pairs):
    clean, triggered = pairs
    ours = _per_image_psnr(clean, triggered)
    for i in range(len(clean)):
        reference = skimage_psnr(clean[i].numpy(), triggered[i].numpy(), data_range=1.0)
        assert ours[i].item() == pytest.approx(reference, rel=1e-4)


def test_ssim_agrees_with_skimage(pairs):
    clean, triggered = pairs
    ours = _per_image_ssim(clean, triggered)
    for i in range(len(clean)):
        reference = skimage_ssim(
            clean[i].numpy(),
            triggered[i].numpy(),
            data_range=1.0,
            channel_axis=0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        assert ours[i].item() == pytest.approx(reference, abs=1e-3)


def test_identical_images_are_maximally_similar():
    clean = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    metrics = compute_stealth_metrics(clean, clean.clone(), torch.device("cpu"))
    assert np.isinf(metrics["psnr_mean"])
    assert metrics["ssim_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["lpips_mean"] == pytest.approx(0.0, abs=1e-4)


def test_identity_trigger_leaves_images_unchanged():
    identity = Attack("identity", lambda image, index: image, "all_to_one", 0)
    clean = torch.rand(3, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    triggered = torch.stack(
        [identity.apply_trigger(clean[i], i) for i in range(len(clean))]
    )
    metrics = compute_stealth_metrics(triggered, clean, torch.device("cpu"))
    assert np.isinf(metrics["psnr_mean"])
    assert metrics["ssim_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["lpips_mean"] == pytest.approx(0.0, abs=1e-4)


def test_cache_does_not_recompute_for_repeated_key(monkeypatch):
    stealth._stealth_cache.clear()
    calls = {"count": 0}
    marker = {"psnr_mean": 42.0}

    def fake_build(*args, **kwargs):
        return torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)

    def fake_compute(*args, **kwargs):
        calls["count"] += 1
        return dict(marker)

    monkeypatch.setattr(stealth, "build_clean_triggered_pairs", fake_build)
    monkeypatch.setattr(stealth, "compute_stealth_metrics", fake_compute)

    attack = Attack("identity", lambda image, index: image, "all_to_one", 0)
    first = cached_stealth_metrics(
        "cifar10", "badnet", attack, "raw_data", torch.device("cpu")
    )
    second = cached_stealth_metrics(
        "cifar10", "badnet", attack, "raw_data", torch.device("cpu")
    )
    assert first == marker
    assert second == marker
    assert calls["count"] == 1
