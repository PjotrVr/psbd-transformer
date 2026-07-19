"""Fast unit tests for the attack implementations and the poisoning pipeline.

These run in milliseconds with no model, no GPU, and no dataset download, so they
are safe on every change and in continuous integration. They check three things:
that each trigger has the structural property its paper defines, that the label
and index policy is correct, and that the poisoning datasets relabel and trigger
the right samples.

Run with pytest from the repo root: pytest tests/test_attacks.py.
"""

import pytest
import torch

from attacks import ATTACK_NAMES, build_attack, default_config
from poison import (
    Attack,
    AttackSuccessSet,
    CoverPoisonedTrainingSet,
    PoisonedTrainingSet,
    attack_success_label,
    choose_indices_with_cover,
    choose_poison_indices,
    is_eval_poisonable,
    is_poisonable,
    poisoned_label,
)

# "generated" needs a --poisoned-dir of pre-generated images on disk, not
# something a unit test can synthesize, so it is exercised elsewhere.
TESTABLE_ATTACK_NAMES = tuple(name for name in ATTACK_NAMES if name != "generated")

SIZE = 32
IDENTITY = lambda image: image


def _mid_gray() -> torch.Tensor:
    # A flat image keeps additive triggers off the 0 and 1 clamps, so the
    # difference reflects the trigger exactly rather than saturated pixels.
    return torch.full((3, SIZE, SIZE), 0.5)


def _gradient() -> torch.Tensor:
    ramp = torch.linspace(0.1, 0.9, SIZE)
    return ramp.view(1, 1, SIZE).expand(3, SIZE, SIZE).clone()


def _random(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, SIZE, SIZE, generator=generator)


def _built(name: str) -> Attack:
    return build_attack(name, default_config(name), SIZE, target_label=0)


def _assert_static_trigger(attack: Attack, image: torch.Tensor) -> None:
    first = attack.apply_trigger(image, 0)
    assert torch.equal(first, attack.apply_trigger(image, 0)), f"{attack.name} is not deterministic"
    assert torch.equal(first, attack.apply_trigger(image, 5)), f"{attack.name} depends on the index"
    assert first.min() >= 0.0 and first.max() <= 1.0, f"{attack.name} left the 0 to 1 range"


def test_badnet_patch_locality():
    attack = _built("badnet_a2o")
    image = _gradient()
    _assert_static_trigger(attack, image)
    poisoned = attack.apply_trigger(image, 0)
    size = default_config("badnet_a2o").patch_size

    clean_outside = image.clone()
    poisoned_outside = poisoned.clone()
    clean_outside[:, -size:, -size:] = 0.0
    poisoned_outside[:, -size:, -size:] = 0.0
    assert torch.equal(clean_outside, poisoned_outside), "badnet changed pixels outside the patch"
    assert not torch.equal(poisoned[:, -size:, -size:], image[:, -size:, -size:]), "patch not applied"


def test_badnet_label_modes():
    assert default_config("badnet_a2o").label_mode == "all_to_one"
    assert default_config("badnet_a2a").label_mode == "all_to_all"


def test_blend_is_bounded_and_changes_image():
    attack = _built("blend")
    image = _mid_gray()
    _assert_static_trigger(attack, image)
    assert not torch.equal(attack.apply_trigger(image, 0), image), "blend produced no change"


def test_sig_signal_is_column_only():
    attack = _built("sig")
    image = _mid_gray()
    _assert_static_trigger(attack, image)
    difference = attack.apply_trigger(image, 0) - image
    first_row = difference[:, 0, :]
    assert torch.allclose(difference, first_row.unsqueeze(1).expand_as(difference), atol=1e-6), (
        "sig signal must depend on the column only, so every row matches"
    )
    assert first_row.std() > 0, "sig signal must vary across columns"


def test_wanet_warps_the_image():
    attack = _built("wanet")
    image = _gradient()
    _assert_static_trigger(attack, image)
    assert not torch.equal(attack.apply_trigger(image, 0), image), "wanet produced no warp"


def test_lf_energy_is_low_frequency():
    attack = _built("lf")
    image = _mid_gray()
    _assert_static_trigger(attack, image)
    difference = attack.apply_trigger(image, 0) - image
    spectrum = torch.fft.fftshift(torch.fft.fft2(difference), dim=(-2, -1)).abs()
    center = SIZE // 2
    cutoff = default_config("lf").cutoff
    low_band = spectrum[:, center - cutoff:center + cutoff + 1, center - cutoff:center + cutoff + 1]
    assert (low_band.sum() / spectrum.sum().clamp_min(1e-8)) > 0.5, "lf energy is not low frequency"


def test_lc_four_corners_clean_label():
    attack = _built("lc")
    assert attack.label_mode == "clean_label"
    image = _gradient()
    poisoned = attack.apply_trigger(image, 0)
    size = default_config("lc").patch_size

    corners = [
        (slice(0, size), slice(0, size)),
        (slice(0, size), slice(SIZE - size, SIZE)),
        (slice(SIZE - size, SIZE), slice(0, size)),
        (slice(SIZE - size, SIZE), slice(SIZE - size, SIZE)),
    ]
    for rows, columns in corners:
        assert not torch.equal(poisoned[:, rows, columns], image[:, rows, columns]), "corner not stamped"
    center = slice(size, SIZE - size)
    assert torch.equal(poisoned[:, center, center], image[:, center, center]), "lc changed the center"


def test_bpp_quantizes_to_grid():
    attack = _built("bpp")
    image = _random(3)
    _assert_static_trigger(attack, image)
    levels = 2 ** default_config("bpp").bit_depth
    scaled = attack.apply_trigger(image, 0) * (levels - 1)
    assert torch.allclose(scaled, torch.round(scaled), atol=1e-5), "bpp output is not on the quantization grid"


def test_adaptive_blend_has_cover_rate():
    config = default_config("adaptive_blend")
    assert config.cover_rate > 0.0, "adaptive blend needs cover samples"
    _assert_static_trigger(_built("adaptive_blend"), _mid_gray())


def test_tact_has_sources_and_cover():
    config = default_config("tact")
    assert len(config.source_classes) > 0, "tact needs source classes"
    assert config.cover_rate > 0.0, "tact needs cover samples"
    _assert_static_trigger(_built("tact"), _gradient())


def test_label_policy():
    assert is_poisonable("all_to_one", 1, 0)
    assert not is_poisonable("all_to_one", 0, 0)
    assert is_poisonable("all_to_all", 0, 0)
    assert is_poisonable("clean_label", 0, 0)
    assert not is_poisonable("clean_label", 1, 0)
    assert poisoned_label("all_to_one", 5, 0, 10) == 0
    assert poisoned_label("all_to_all", 9, 0, 10) == 0
    assert poisoned_label("all_to_all", 3, 0, 10) == 4
    assert poisoned_label("clean_label", 7, 0, 10) == 7


def _marker_attack(label_mode: str = "all_to_one", target: int = 0) -> Attack:
    # A trigger that blanks the image to all ones, easy to detect in a dataset.
    return Attack("marker", lambda image, index: torch.ones_like(image), label_mode, target)


def test_choose_poison_indices_all_to_one():
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    indices = choose_poison_indices(labels, _marker_attack(), 0.5, seed=0)
    assert len(indices) == 4, "count should be round(rate times dataset size)"
    assert all(labels[i] != 0 for i in indices), "all_to_one must skip the target class"
    assert indices == choose_poison_indices(labels, _marker_attack(), 0.5, seed=0), "must be reproducible"


def test_cover_indices_disjoint_and_non_target():
    labels = [0, 1, 2, 3, 0, 1, 2, 3, 1, 2]
    poison, cover = choose_indices_with_cover(
        labels, _marker_attack(), poison_rate=0.2, cover_rate=0.2, source_classes=None, seed=0
    )
    assert poison.isdisjoint(cover), "poison and cover must not overlap"
    assert all(labels[i] != 0 for i in poison), "all_to_one poison must skip the target"
    assert all(labels[i] != 0 for i in cover), "cover must not be the target class"


def test_cover_indices_source_specific():
    labels = [0, 1, 2, 3, 0, 1, 2, 3, 1, 2]
    poison, cover = choose_indices_with_cover(
        labels, _marker_attack(), poison_rate=0.2, cover_rate=0.2, source_classes=(1,), seed=0
    )
    assert all(labels[i] == 1 for i in poison), "source-specific poison must be the source class"
    assert all(labels[i] not in (0, 1) for i in cover), "cover excludes target and source"


def _fake_base(count: int):
    return [(torch.zeros(3, 4, 4), index % 4) for index in range(count)]


def _fake_base_at_size(count: int, size: int):
    return [(torch.zeros(3, size, size), index % 4) for index in range(count)]


def test_poisoned_training_set_relabels_and_triggers():
    base = _fake_base(8)
    dataset = PoisonedTrainingSet(base, _marker_attack(), {1, 3}, IDENTITY, num_classes=4)
    image_one, label_one = dataset[1]
    assert torch.equal(image_one, torch.ones(3, 4, 4)) and label_one == 0, "poisoned sample wrong"
    image_zero, label_zero = dataset[0]
    assert torch.equal(image_zero, torch.zeros(3, 4, 4)) and label_zero == base[0][1], "clean sample changed"


def test_cover_sample_keeps_label_but_triggers():
    base = _fake_base(8)
    dataset = CoverPoisonedTrainingSet(base, _marker_attack(), {1}, {2}, IDENTITY, num_classes=4)
    cover_image, cover_label = dataset[2]
    assert torch.equal(cover_image, torch.ones(3, 4, 4)), "cover sample must be triggered"
    assert cover_label == base[2][1], "cover sample must keep its label"


def test_fully_poisoned_test_set_drops_target():
    base = _fake_base(8)
    labels = [item[1] for item in base]
    dataset = AttackSuccessSet(base, labels, _marker_attack(), IDENTITY, num_classes=4)
    assert len(dataset) == sum(1 for y in labels if y != 0), "target-class samples should be dropped"
    for position in range(len(dataset)):
        _, target = dataset[position]
        assert target == 0, "every ASR sample should carry the target label"


def test_attack_success_set_clean_label_selects_non_target_only():
    # Clean-label training poisons only target-class images, but attack success
    # must be measured on non-target images fooled into predicting the target,
    # the opposite eligibility from training. AttackSuccessSet must ask that
    # question, not the training-time one.
    base = _fake_base(8)
    labels = [item[1] for item in base]
    dataset = AttackSuccessSet(base, labels, _marker_attack(label_mode="clean_label"), IDENTITY, num_classes=4)
    assert len(dataset) == sum(1 for y in labels if y != 0), "only non-target images should be eligible"
    for position in range(len(dataset)):
        _, target = dataset[position]
        assert target == 0, "every eligible sample should carry the target label"


@pytest.mark.parametrize("name", TESTABLE_ATTACK_NAMES)
def test_every_attack_trigger_is_deterministic_and_bounded(name):
    attack = _built(name)
    first = attack.apply_trigger(_gradient(), 0)
    assert torch.equal(first, attack.apply_trigger(_gradient(), 0)), f"{name} is not deterministic"
    assert torch.equal(first, attack.apply_trigger(_gradient(), 5)), f"{name} depends on the index"
    assert first.min() >= 0.0 and first.max() <= 1.0, f"{name} left the 0 to 1 range"


@pytest.mark.parametrize("name", TESTABLE_ATTACK_NAMES)
def test_every_attack_success_set_matches_its_label_mode(name):
    # Uses each attack's real trigger and real label_mode, unlike the marker-
    # attack tests above, so a future attack registered with the wrong
    # label_mode in its default_config would show up here.
    attack = _built(name)
    base = _fake_base_at_size(8, SIZE)  # matches the image_size _built configured the attack for
    labels = [item[1] for item in base]
    dataset = AttackSuccessSet(base, labels, attack, IDENTITY, num_classes=4)

    expected_positions = [
        position for position, label in enumerate(labels)
        if is_eval_poisonable(attack.label_mode, label, attack.target_label)
    ]
    assert len(dataset) == len(expected_positions)
    for index, position in enumerate(expected_positions):
        _, target = dataset[index]
        expected = attack_success_label(attack.label_mode, labels[position], attack.target_label, num_classes=4)
        assert target == expected


