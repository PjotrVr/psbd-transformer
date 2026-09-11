"""Mechanical checks on the adaptive-evasion loss before any training job.

The expensive failure this prevents: a penalty that is silently disconnected from
the graph trains for hours, reports a falling loss and changes nothing.

Ported from a standalone script that pytest never collected, so none of these
ran in CI before.
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

from attacks.evasion import (
    absolute_psu_for_batch,
    evasion_penalty,
    evasive_update,
    multi_probe_evasion_penalty,
    multi_probe_psu_for_batch,
    paper_adaptive_penalty,
    psu_for_batch,
)

PROBE = {
    "position": "before_attention_norm",
    "operator": "dropout",
    "rate": 0.3,
    "architecture": "vit",
}

PROBE_PAPER = {**PROBE, "objective": "psbd_paper"}


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

    A toy ViT cannot host a realistic backdoor: 300-plus steps on 8 samples
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


def test_absolute_psu_is_the_unnormalised_fractional_psu(batch):
    """absolute_psu_for_batch and psu_for_batch share _probe_confidences.

    psu_ratio = (base - dropped) / base, so multiplying the fractional form back
    by base recovers the absolute form exactly, up to the floor clamp on base
    that guards the division.
    """
    images, _, _ = batch
    torch.manual_seed(0)
    model = TinyViT()

    torch.manual_seed(1)
    fractional, logits = psu_for_batch(model, images, PROBE, passes=3)
    torch.manual_seed(1)
    absolute, logits_again = absolute_psu_for_batch(model, images, PROBE, passes=3)

    assert torch.equal(logits, logits_again)
    base = (
        torch.softmax(logits, dim=1)
        .gather(1, logits.argmax(dim=1, keepdim=True))
        .squeeze(1)
    )
    reconstructed = fractional * base.clamp_min(1e-6)
    assert torch.allclose(absolute, reconstructed, atol=1e-6)


def test_absolute_psu_is_differentiable_and_reaches_the_model(batch):
    images, _, _ = batch
    torch.manual_seed(0)
    model = TinyViT()

    model.zero_grad()
    psu, _ = absolute_psu_for_batch(model, images, PROBE, passes=3)
    assert psu.shape == (8,)
    assert psu.requires_grad

    psu.sum().backward()
    total = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    assert total > 0, f"sum|grad| = {total:.4f}"


def test_paper_penalty_is_the_batch_mean_and_ignores_is_poisoned():
    """L_ada = mean over D^c union D^b: the whole batch, poisoned flag irrelevant."""
    psu = torch.tensor([0.1, 0.3, 0.5, 0.7])
    assert float(paper_adaptive_penalty(psu)) == pytest.approx(0.4)


def test_paper_objective_matches_the_convex_combination(batch):
    """evasive_update's psbd_paper branch reproduces (1 - alpha) CE + alpha L_ada by hand."""
    images, labels, is_poisoned = batch
    criterion = nn.CrossEntropyLoss()
    alpha = 0.5

    torch.manual_seed(2)
    model = TinyViT()
    torch.manual_seed(2)
    reference = TinyViT()
    assert all(
        torch.equal(p, q) for p, q in zip(model.parameters(), reference.parameters())
    )

    torch.manual_seed(3)
    psu, logits = absolute_psu_for_batch(reference, images, PROBE_PAPER, passes=3)
    expected_loss = (1 - alpha) * criterion(logits, labels) + alpha * psu.mean()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    torch.manual_seed(3)
    batch_loss, _ = evasive_update(
        model, images, labels, is_poisoned, criterion, optimizer, PROBE_PAPER, alpha, 3
    )
    assert float(batch_loss) == pytest.approx(float(expected_loss), abs=1e-5)


def test_paper_weight_zero_matches_plain_cross_entropy(batch):
    """alpha = 0 must reduce to plain cross-entropy, just like the hinge at weight 0."""
    images, labels, is_poisoned = batch
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(1)
    evasive_model = TinyViT()
    torch.manual_seed(1)
    plain_model = TinyViT()

    evasive_optimizer = torch.optim.SGD(evasive_model.parameters(), lr=0.1)
    plain_optimizer = torch.optim.SGD(plain_model.parameters(), lr=0.1)

    evasive_update(
        evasive_model,
        images,
        labels,
        is_poisoned,
        criterion,
        evasive_optimizer,
        PROBE_PAPER,
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


def test_paper_weight_one_drops_cross_entropy_entirely(batch):
    """alpha = 1 must zero out the classification term, unlike the hinge's additive form."""
    images, labels, is_poisoned = batch
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(4)
    model = TinyViT()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    torch.manual_seed(5)
    batch_loss, _ = evasive_update(
        model, images, labels, is_poisoned, criterion, optimizer, PROBE_PAPER, 1.0, 3
    )

    torch.manual_seed(5)
    psu, _ = absolute_psu_for_batch(model, images, PROBE_PAPER, passes=3)
    assert float(batch_loss) == pytest.approx(float(psu.mean()), abs=1e-5)


def test_unknown_objective_raises(batch):
    images, labels, is_poisoned = batch
    model = TinyViT()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    bad_probe = {**PROBE, "objective": "not_a_real_objective"}
    with pytest.raises(ValueError, match="unknown evasion objective"):
        evasive_update(
            model,
            images,
            labels,
            is_poisoned,
            nn.CrossEntropyLoss(),
            optimizer,
            bad_probe,
            1.0,
            3,
        )


def test_optimising_the_paper_penalty_lowers_mean_psu():
    """The psbd_paper objective, run on a real optimizer, pushes PSU down.

    Mirrors test_optimising_the_penalty_closes_an_active_gap: the toy ViT is
    warmed up first, since PSU is identically 0 at random init (a uniform
    softmax barely moves under the probe). Once warmed up, a large alpha makes
    L_ada dominate the loss, so mean PSU absolute must fall step over step.
    """
    torch.manual_seed(0)
    model = TinyViT()
    model.train()
    images = torch.randn(64, 3, 32, 32)
    labels = torch.randint(0, 4, (64,))
    is_poisoned = torch.zeros(64, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    strong_probe = {**PROBE_PAPER, "rate": 0.7}

    warmup = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(60):
        warmup.zero_grad(set_to_none=True)
        criterion(model(images), labels).backward()
        warmup.step()

    with torch.no_grad():
        initial, _ = absolute_psu_for_batch(model, images, strong_probe, passes=5)
    assert float(initial.mean()) > 1e-3, f"mean PSU {float(initial.mean()):.4f}"

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    first, last = None, None
    for step in range(40):
        _, stats = evasive_update(
            model,
            images,
            labels,
            is_poisoned,
            criterion,
            optimizer,
            strong_probe,
            weight=0.9,
            passes=5,
        )
        if step == 0:
            first = stats
        last = stats

    first_psu = first["psu_clean"]
    last_psu = last["psu_clean"]
    assert last_psu < first_psu, f"{first_psu:.4f} -> {last_psu:.4f}"


PROBE_B = {
    "position": "before_attention_norm",
    "operator": "token_mask",
    "rate": 0.5,
    "architecture": "vit",
}


def test_multi_probe_penalty_equals_single_probe_hinge_for_1_probe(batch):
    """multi_probe_evasion_penalty over a 1-element list is evasion_penalty."""
    _, _, is_poisoned = batch
    psu_gap = torch.tensor([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

    single = evasion_penalty(psu_gap, is_poisoned)
    multi = multi_probe_evasion_penalty([psu_gap], is_poisoned)
    assert float(multi) == pytest.approx(float(single))


def test_multi_probe_penalty_is_the_mean_over_probes():
    """The formula is a mean, not a sum, of the per-probe hinges (module docstring)."""
    is_poisoned = torch.tensor([1, 1, 0, 0])
    psu_a = torch.tensor([0.9, 0.9, 0.1, 0.1])  # relu(0.0 - 0.9), hinge 0.0
    psu_b = torch.tensor([0.1, 0.1, 0.9, 0.9])  # relu(0.9 - 0.1), hinge 0.8

    combined = multi_probe_evasion_penalty([psu_a, psu_b], is_poisoned)
    expected = (
        float(evasion_penalty(psu_a, is_poisoned))
        + float(evasion_penalty(psu_b, is_poisoned))
    ) / 2
    assert float(combined) == pytest.approx(expected)


def test_multi_probe_penalty_decreases_when_every_probes_gap_decreases():
    is_poisoned = torch.tensor([1, 1, 0, 0])
    wide_a = torch.tensor([0.1, 0.1, 0.9, 0.9])
    wide_b = torch.tensor([0.2, 0.2, 0.8, 0.8])
    narrow_a = torch.tensor([0.4, 0.4, 0.6, 0.6])
    narrow_b = torch.tensor([0.35, 0.35, 0.65, 0.65])

    wide_penalty = multi_probe_evasion_penalty([wide_a, wide_b], is_poisoned)
    narrow_penalty = multi_probe_evasion_penalty([narrow_a, narrow_b], is_poisoned)
    assert float(narrow_penalty) < float(wide_penalty)


def test_multi_probe_psu_shares_the_base_forward_pass(batch):
    """multi_probe_psu_for_batch's shared logits equal a lone psu_for_batch's logits."""
    images, _, _ = batch
    torch.manual_seed(0)
    model = TinyViT()

    torch.manual_seed(7)
    psu_per_probe, shared_logits = multi_probe_psu_for_batch(
        model, images, [PROBE, PROBE_B], passes=3
    )
    assert len(psu_per_probe) == 2
    for psu in psu_per_probe:
        assert psu.shape == (8,)
        assert psu.requires_grad

    torch.manual_seed(7)
    solo_psu, solo_logits = psu_for_batch(model, images, PROBE, passes=3)
    assert torch.equal(shared_logits, solo_logits)
    assert torch.equal(psu_per_probe[0], solo_psu)


def test_evasive_update_with_a_1_element_list_matches_a_plain_probe_dict(batch):
    """evasive_update([PROBE]) must be bit-for-bit evasive_update(PROBE)."""
    images, labels, is_poisoned = batch
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(1)
    list_model = TinyViT()
    torch.manual_seed(1)
    dict_model = TinyViT()

    list_optimizer = torch.optim.SGD(list_model.parameters(), lr=0.1)
    dict_optimizer = torch.optim.SGD(dict_model.parameters(), lr=0.1)

    torch.manual_seed(2)
    list_loss, list_stats = evasive_update(
        list_model,
        images,
        labels,
        is_poisoned,
        criterion,
        list_optimizer,
        [PROBE],
        1.0,
        3,
    )
    torch.manual_seed(2)
    dict_loss, dict_stats = evasive_update(
        dict_model,
        images,
        labels,
        is_poisoned,
        criterion,
        dict_optimizer,
        PROBE,
        1.0,
        3,
    )

    assert float(list_loss) == pytest.approx(float(dict_loss), abs=1e-6)
    assert list_stats["penalty"] == pytest.approx(dict_stats["penalty"], abs=1e-6)
    drift = max(
        float((p - q).abs().max())
        for p, q in zip(list_model.parameters(), dict_model.parameters())
    )
    assert drift < 1e-6, f"max drift {drift:.2e}"


def test_evasive_update_rejects_multi_probe_with_the_paper_objective(batch):
    images, labels, is_poisoned = batch
    model = TinyViT()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    probes = [PROBE_PAPER, {**PROBE_B, "objective": "psbd_paper"}]
    with pytest.raises(ValueError, match="only supports the hinge objective"):
        evasive_update(
            model,
            images,
            labels,
            is_poisoned,
            nn.CrossEntropyLoss(),
            optimizer,
            probes,
            1.0,
            3,
        )


def test_optimising_the_multi_probe_penalty_closes_every_active_gap():
    """The multi-probe hinge falls under a real optimizer, and so does each probe's own gap.

    Mirrors test_optimising_the_penalty_closes_an_active_gap, with 2 probes at
    the same position but different operators so their PSU statistics differ.
    """
    torch.manual_seed(0)
    model = TinyViT()
    model.train()
    images = torch.randn(64, 3, 32, 32)
    labels = torch.randint(0, 4, (64,))
    criterion = nn.CrossEntropyLoss()
    probe_a = dict(PROBE, rate=0.7)
    probe_b = dict(PROBE_B, rate=0.7)

    warmup = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(60):
        warmup.zero_grad(set_to_none=True)
        criterion(model(images), labels).backward()
        warmup.step()

    with torch.no_grad():
        initial_a, _ = psu_for_batch(model, images, probe_a, passes=5)
    lowest = torch.argsort(initial_a)[:16]
    flags = torch.zeros(64, dtype=torch.long)
    flags[lowest] = 1

    def gaps():
        with torch.no_grad():
            psu_a, _ = psu_for_batch(model, images, probe_a, passes=5)
            psu_b, _ = psu_for_batch(model, images, probe_b, passes=5)
        poisoned = flags.bool()
        gap_a = float(psu_a[~poisoned].mean() - psu_a[poisoned].mean())
        gap_b = float(psu_b[~poisoned].mean() - psu_b[poisoned].mean())
        return gap_a, gap_b

    first_gap_a, first_gap_b = gaps()

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
            [probe_a, probe_b],
            weight=100.0,
            passes=5,
        )
        if step == 0:
            first = stats
        last = stats

    last_gap_a, last_gap_b = gaps()

    assert first["penalty"] > 0, (
        f"multi-probe hinge inactive at step 0: {first['penalty']:.4f}"
    )
    assert last["penalty"] < first["penalty"], (
        f"{first['penalty']:.4f} -> {last['penalty']:.4f}"
    )
    assert last_gap_a < first_gap_a, (
        f"probe a gap {first_gap_a:.4f} -> {last_gap_a:.4f}"
    )
    assert last_gap_b < first_gap_b, (
        f"probe b gap {first_gap_b:.4f} -> {last_gap_b:.4f}"
    )
