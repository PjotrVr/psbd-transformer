"""Mechanical checks on the adaptive-evasion loss before any training job.

The expensive failure this prevents: a penalty that is silently disconnected from
the graph trains for hours, reports a falling loss, and changes nothing.

Ported from a standalone script that pytest never collected, so none of these
ran in CI before.
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

from attacks.evasion import evasion_penalty, evasive_update, psu_for_batch

PROBE = {
    "position": "before_attention_norm",
    "operator": "dropout",
    "rate": 0.3,
    "architecture": "vit",
}


class TinyViT(nn.Module):
    """A stand-in with the module names the ViT position registry resolves."""

    def __init__(self):
        super().__init__()
        self.inner = VisionTransformer(
            image_size=32,
            patch_size=16,
            num_layers=2,
            num_heads=2,
            hidden_dim=32,
            mlp_dim=64,
            num_classes=4,
        )

    def forward(self, x):
        return self.inner(x)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 4, (8,))
    is_poisoned = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])
    return images, labels, is_poisoned


def test_psu_is_differentiable_and_reaches_the_model(batch):
    images, _, _ = batch
    torch.manual_seed(0)
    model = TinyViT()

    model.zero_grad()
    psu, _ = psu_for_batch(model, images, PROBE, passes=3)
    assert psu.shape == (8,)
    assert psu.requires_grad

    psu.sum().backward()
    total = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    assert total > 0, f"sum|grad| = {total:.4f}"


def test_probe_is_removed_after_use(batch):
    images, _, _ = batch
    torch.manual_seed(0)
    model = TinyViT()

    before = {name: module.training for name, module in model.named_modules()}
    psu_for_batch(model, images, PROBE, passes=2)
    after = {name: module.training for name, module in model.named_modules()}
    assert before == after

    model.eval()
    with torch.no_grad():
        first, second = model(images), model(images)
    assert torch.equal(first, second)


def test_hinge_is_positive_when_the_poisoned_shift_is_lower(batch):
    _, _, is_poisoned = batch
    psu_gap = torch.tensor([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert float(evasion_penalty(psu_gap, is_poisoned)) > 0


def test_hinge_is_zero_once_the_poisoned_shift_exceeds_clean(batch):
    _, _, is_poisoned = batch
    psu_ok = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert float(evasion_penalty(psu_ok, is_poisoned)) == 0.0


def test_hinge_is_zero_when_a_group_is_absent():
    psu_gap = torch.tensor([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    all_clean = torch.zeros(8, dtype=torch.long)
    assert float(evasion_penalty(psu_gap, all_clean)) == 0.0


def test_optimising_the_penalty_closes_an_active_gap():
    """The hinge falls when a real optimizer is pointed at it.

    A toy ViT cannot host a realistic backdoor: a few hundred steps on 8 samples
    saturates the softmax and PSU collapses to about 1e-4 for every sample, so any
    hinge built on an "implanted" backdoor here is vacuously zero. What can be
    tested honestly is the optimisation itself. The group assignment below is
    chosen so the hinge is active by construction, which is the condition the real
    attacker faces. Whether a real backdoor produces that condition is a question
    for the GPU run, not for a unit test.
    """
    torch.manual_seed(0)
    model = TinyViT()
    model.train()
    images = torch.randn(64, 3, 32, 32)
    labels = torch.randint(0, 4, (64,))
    criterion = nn.CrossEntropyLoss()
    strong_probe = dict(PROBE, rate=0.7)

    # PSU is identically zero at random init: a uniform softmax barely moves under
    # dropout, so P_c is the same with and without it. The statistic only exists
    # once the model is confident, which is why PSBD reads a late-stage model. Warm
    # up to that regime before measuring anything.
    warmup = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(60):
        warmup.zero_grad(set_to_none=True)
        criterion(model(images), labels).backward()
        warmup.step()

    with torch.no_grad():
        initial, _ = psu_for_batch(model, images, strong_probe, passes=5)
    assert float(initial.std()) > 1e-3, f"PSU std {float(initial.std()):.4f}"

    lowest = torch.argsort(initial)[:16]
    flags = torch.zeros(64, dtype=torch.long)
    flags[lowest] = 1

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    first, last = None, None
    for step in range(40):
        _, stats = evasive_update(
            model,
            images,
            labels,
            flags,
            criterion,
            optimizer,
            strong_probe,
            weight=100.0,
            passes=5,
        )
        if step == 0:
            first = stats
        last = stats

    assert first["penalty"] > 0, f"hinge inactive at step 0: {first['penalty']:.4f}"
    assert last["penalty"] < first["penalty"], (
        f"{first['penalty']:.4f} -> {last['penalty']:.4f}"
    )


def test_weight_zero_matches_plain_cross_entropy(batch):
    images, labels, is_poisoned = batch
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(1)
    evasive_model = TinyViT()
    torch.manual_seed(1)
    plain_model = TinyViT()
    assert all(
        torch.equal(p, q)
        for p, q in zip(evasive_model.parameters(), plain_model.parameters())
    )

    evasive_optimizer = torch.optim.SGD(evasive_model.parameters(), lr=0.1)
    plain_optimizer = torch.optim.SGD(plain_model.parameters(), lr=0.1)

    evasive_update(
        evasive_model,
        images,
        labels,
        is_poisoned,
        criterion,
        evasive_optimizer,
        PROBE,
        0.0,
        3,
    )
    plain_optimizer.zero_grad(set_to_none=True)
    criterion(plain_model(images), labels).backward()
    plain_optimizer.step()

    drift = max(
        float((p - q).abs().max())
        for p, q in zip(evasive_model.parameters(), plain_model.parameters())
    )
    assert drift < 1e-6, f"max drift {drift:.2e}"
