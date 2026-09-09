"""Fast unit tests for the attack implementations and the poisoning pipeline.

These run in milliseconds with no model, no GPU, and no dataset download, so they
are safe on every change and in continuous integration. They check three things:
that each trigger has the structural property its paper defines, that the label
and index policy is correct, and that the poisoning datasets relabel and trigger
the right samples.

Run with pytest from the repo root: pytest tests/test_attacks.py.
"""

from dataclasses import fields, replace

import pytest
import torch

from attacks import ATTACK_NAMES, build_attack, default_config
from attacks.poisoning import (
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


def IDENTITY(image):
    return image


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


def _assert_static_trigger_fn(plant, name: str, image: torch.Tensor) -> None:
    first = plant(image, 0)
    assert torch.equal(first, plant(image, 0)), f"{name} is not deterministic"
    assert torch.equal(first, plant(image, 5)), f"{name} depends on the index"
    assert first.min() >= 0.0 and first.max() <= 1.0, f"{name} left the 0 to 1 range"


def _assert_static_trigger(attack: Attack, image: torch.Tensor) -> None:
    _assert_static_trigger_fn(attack.apply_trigger, attack.name, image)


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
    assert torch.equal(clean_outside, poisoned_outside), (
        "badnet changed pixels outside the patch"
    )
    assert not torch.equal(poisoned[:, -size:, -size:], image[:, -size:, -size:]), (
        "patch not applied"
    )


def test_badnet_label_modes():
    assert default_config("badnet_a2o").label_mode == "all_to_one"
    assert default_config("badnet_a2a").label_mode == "all_to_all"


def test_blend_is_bounded_and_changes_image():
    attack = _built("blend")
    image = _mid_gray()
    _assert_static_trigger(attack, image)
    assert not torch.equal(attack.apply_trigger(image, 0), image), (
        "blend produced no change"
    )


def test_sig_signal_is_column_only():
    attack = _built("sig")
    image = _mid_gray()
    _assert_static_trigger(attack, image)
    difference = attack.apply_trigger(image, 0) - image
    first_row = difference[:, 0, :]
    assert torch.allclose(
        difference, first_row.unsqueeze(1).expand_as(difference), atol=1e-6
    ), "sig signal must depend on the column only, so every row matches"
    assert first_row.std() > 0, "sig signal must vary across columns"


def test_wanet_warps_the_image():
    attack = _built("wanet")
    image = _gradient()
    _assert_static_trigger(attack, image)
    assert not torch.equal(attack.apply_trigger(image, 0), image), (
        "wanet produced no warp"
    )


def test_lf_energy_is_low_frequency():
    attack = _built("lf")
    image = _mid_gray()
    _assert_static_trigger(attack, image)
    difference = attack.apply_trigger(image, 0) - image
    spectrum = torch.fft.fftshift(torch.fft.fft2(difference), dim=(-2, -1)).abs()
    center = SIZE // 2
    cutoff = default_config("lf").cutoff
    low_band = spectrum[
        :, center - cutoff : center + cutoff + 1, center - cutoff : center + cutoff + 1
    ]
    assert (low_band.sum() / spectrum.sum().clamp_min(1e-8)) > 0.5, (
        "lf energy is not low frequency"
    )


def test_lc_four_corners_clean_label():
    attack = _built("lc")
    assert attack.label_mode == "clean_label"
    image = _gradient()
    # The eval trigger is the patch alone by definition; apply_trigger additionally
    # substitutes an adversarial base when one was generated, which is covered below.
    poisoned = attack.apply_trigger_eval(image, 0)
    size = default_config("lc").patch_size

    corners = [
        (slice(0, size), slice(0, size)),
        (slice(0, size), slice(SIZE - size, SIZE)),
        (slice(SIZE - size, SIZE), slice(0, size)),
        (slice(SIZE - size, SIZE), slice(SIZE - size, SIZE)),
    ]
    for rows, columns in corners:
        assert not torch.equal(poisoned[:, rows, columns], image[:, rows, columns]), (
            "corner not stamped"
        )
    center = slice(size, SIZE - size)
    assert torch.equal(poisoned[:, center, center], image[:, center, center]), (
        "lc changed the center"
    )


def test_lc_adversarial_base_is_training_only(tmp_path):
    """Turner's attack perturbs the base image before stamping, at TRAIN time only.

    Without that step the model can still read the class off the untouched image and
    never has to use the trigger, which is why the patch-only variant needs nearly the
    whole target class to implant. At eval time attack success is measured on non-target
    images, which have no adversarial base by construction, so only the patch applies.
    """
    directory = tmp_path / "bases"
    directory.mkdir()
    base = torch.zeros(3, SIZE, SIZE)
    torch.save(
        {"indices": torch.tensor([7]), "images": base.unsqueeze(0)},
        directory / "bases.pt",
    )
    config = replace(default_config("lc"), adversarial_dir=str(directory))
    attack = build_attack("lc", config, SIZE, 0)
    image = _gradient()

    with_base = attack.apply_trigger(image, 7)
    without_base = attack.apply_trigger(image, 8)
    eval_only = attack.apply_trigger_eval(image, 7)
    size = config.patch_size
    center = slice(size, SIZE - size)

    assert not torch.equal(with_base[:, center, center], image[:, center, center]), (
        "the adversarial base must replace the image the patch is stamped on"
    )
    assert torch.equal(with_base[:, center, center], base[:, center, center]), (
        "the stamped image should be the stored base outside the corners"
    )
    assert torch.equal(eval_only[:, center, center], image[:, center, center]), (
        "the eval trigger must be the patch alone"
    )
    assert torch.equal(without_base, eval_only), (
        "an index with no base falls back to the clean image, matching the eval trigger"
    )


def test_lc_default_config_is_the_patch_only_variant():
    """No adversarial_dir means byte-identical behaviour to before the field existed.

    Every checkpoint trained before this change rebuilds through default_config, so a
    drift here would silently re-interpret them.
    """
    attack = build_attack("lc", default_config("lc"), SIZE, 0)
    image = _gradient()
    assert torch.equal(
        attack.apply_trigger(image, 3), attack.apply_trigger_eval(image, 3)
    )
    assert torch.equal(attack.apply_trigger(image, 3), attack.apply_trigger(image, 99))


def test_bpp_quantizes_to_grid():
    attack = _built("bpp")
    image = _random(3)
    _assert_static_trigger(attack, image)
    levels = 2 ** default_config("bpp").bit_depth
    scaled = attack.apply_trigger(image, 0) * (levels - 1)
    assert torch.allclose(scaled, torch.round(scaled), atol=1e-5), (
        "bpp output is not on the quantization grid"
    )


def test_adaptive_blend_has_cover_rate():
    config = default_config("adaptive_blend")
    assert config.cover_rate > 0.0, "adaptive blend needs cover samples"


def test_adaptive_blend_trigger_is_asymmetric():
    """Train plants a per-sample SUBSET of the pattern; eval plants all of it.

    This is the mechanism, not an implementation detail: training on a subset forces the
    model to generalise over the pattern, so the full pattern at test time lands well
    inside the learned region. So unlike every other attack here, the training trigger is
    deliberately index-dependent, and only the eval trigger is static.
    """
    attack = _built("adaptive_blend")
    image = _mid_gray()

    _assert_static_trigger_fn(attack.apply_trigger_eval, "adaptive_blend eval", image)

    first = attack.apply_trigger(image, 0)
    assert torch.equal(first, attack.apply_trigger(image, 0)), (
        "the training trigger must be reproducible for a given sample"
    )
    assert not torch.equal(first, attack.apply_trigger(image, 5)), (
        "the training trigger must vary by sample, which is the asymmetry"
    )
    assert first.min() >= 0.0 and first.max() <= 1.0

    full = attack.apply_trigger_eval(image, 0)
    changed_train = (first - image).abs().sum(0) > 1e-6
    changed_eval = (full - image).abs().sum(0) > 1e-6
    assert changed_eval.sum() > changed_train.sum(), (
        "the eval trigger must cover more of the image than a training subset"
    )


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
    return Attack(
        "marker", lambda image, index: torch.ones_like(image), label_mode, target
    )


def test_choose_poison_indices_all_to_one():
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    indices = choose_poison_indices(labels, _marker_attack(), 0.5, seed=0)
    assert len(indices) == 4, "count should be round(rate times dataset size)"
    assert all(labels[i] != 0 for i in indices), "all_to_one must skip the target class"
    assert indices == choose_poison_indices(labels, _marker_attack(), 0.5, seed=0), (
        "must be reproducible"
    )


def test_cover_indices_disjoint_and_non_target():
    labels = [0, 1, 2, 3, 0, 1, 2, 3, 1, 2]
    poison, cover = choose_indices_with_cover(
        labels,
        _marker_attack(),
        poison_rate=0.2,
        cover_rate=0.2,
        source_classes=None,
        seed=0,
    )
    assert poison.isdisjoint(cover), "poison and cover must not overlap"
    assert all(labels[i] != 0 for i in poison), "all_to_one poison must skip the target"
    assert all(labels[i] != 0 for i in cover), "cover must not be the target class"


def test_cover_indices_source_specific():
    labels = [0, 1, 2, 3, 0, 1, 2, 3, 1, 2]
    poison, cover = choose_indices_with_cover(
        labels,
        _marker_attack(),
        poison_rate=0.2,
        cover_rate=0.2,
        source_classes=(1,),
        seed=0,
    )
    assert all(labels[i] == 1 for i in poison), (
        "source-specific poison must be the source class"
    )
    assert all(labels[i] not in (0, 1) for i in cover), (
        "cover excludes target and source"
    )


def _fake_base(count: int):
    return [(torch.zeros(3, 4, 4), index % 4) for index in range(count)]


def _fake_base_at_size(count: int, size: int):
    return [(torch.zeros(3, size, size), index % 4) for index in range(count)]


def test_poisoned_training_set_relabels_and_triggers():
    base = _fake_base(8)
    dataset = PoisonedTrainingSet(
        base, _marker_attack(), {1, 3}, IDENTITY, num_classes=4
    )
    image_one, label_one = dataset[1]
    assert torch.equal(image_one, torch.ones(3, 4, 4)) and label_one == 0, (
        "poisoned sample wrong"
    )
    image_zero, label_zero = dataset[0]
    assert torch.equal(image_zero, torch.zeros(3, 4, 4)) and label_zero == base[0][1], (
        "clean sample changed"
    )


def test_cover_sample_keeps_label_but_triggers():
    base = _fake_base(8)
    dataset = CoverPoisonedTrainingSet(
        base, _marker_attack(), {1}, {2}, IDENTITY, num_classes=4
    )
    cover_image, cover_label = dataset[2]
    assert torch.equal(cover_image, torch.ones(3, 4, 4)), (
        "cover sample must be triggered"
    )
    assert cover_label == base[2][1], "cover sample must keep its label"


def test_fully_poisoned_test_set_drops_target():
    base = _fake_base(8)
    labels = [item[1] for item in base]
    dataset = AttackSuccessSet(base, labels, _marker_attack(), IDENTITY, num_classes=4)
    assert len(dataset) == sum(1 for y in labels if y != 0), (
        "target-class samples should be dropped"
    )
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
    dataset = AttackSuccessSet(
        base, labels, _marker_attack(label_mode="clean_label"), IDENTITY, num_classes=4
    )
    assert len(dataset) == sum(1 for y in labels if y != 0), (
        "only non-target images should be eligible"
    )
    for position in range(len(dataset)):
        _, target = dataset[position]
        assert target == 0, "every eligible sample should carry the target label"


@pytest.mark.parametrize("name", TESTABLE_ATTACK_NAMES)
def test_every_attack_trigger_is_deterministic_and_bounded(name):
    """The trigger a sample is SCORED with must be static and reproducible.

    Read through apply_trigger_eval, because that is what AttackSuccessSet plants and it
    defaults to apply_trigger for every attack that does not distinguish the two.
    Adaptive-Blend does distinguish them on purpose: its training trigger is a per-sample
    subset of the pattern, which is the asymmetry the attack depends on, so requiring
    index-independence of apply_trigger would forbid a correct implementation.
    See test_adaptive_blend_trigger_is_asymmetric.
    """
    attack = _built(name)
    plant = attack.apply_trigger_eval or attack.apply_trigger
    first = plant(_gradient(), 0)
    assert torch.equal(first, plant(_gradient(), 0)), f"{name} is not deterministic"
    assert torch.equal(first, plant(_gradient(), 5)), f"{name} depends on the index"
    assert first.min() >= 0.0 and first.max() <= 1.0, f"{name} left the 0 to 1 range"


@pytest.mark.parametrize("name", TESTABLE_ATTACK_NAMES)
def test_every_attack_success_set_matches_its_label_mode(name):
    # Uses each attack's real trigger and real label_mode, unlike the marker-
    # attack tests above, so a future attack registered with the wrong
    # label_mode in its default_config would show up here.
    attack = _built(name)
    base = _fake_base_at_size(
        8, SIZE
    )  # matches the image_size _built configured the attack for
    labels = [item[1] for item in base]
    # all_to_m maps onto (y + 1) mod m, so the class count has to cover m or the
    # label map runs off the end of the logits. Every other attack keeps 4.
    num_classes = max(4, attack.num_targets or 0)
    dataset = AttackSuccessSet(base, labels, attack, IDENTITY, num_classes=num_classes)

    expected_positions = [
        position
        for position, label in enumerate(labels)
        if is_eval_poisonable(
            attack.label_mode, label, attack.target_label, attack.num_targets
        )
    ]
    if attack.source_classes is not None:
        # A source-specific attack only claims to flip its source classes, so the
        # success set is the intersection, not the whole eval-poisonable pool.
        # Measuring over the pool divides the true ASR by the class count.
        sources = set(attack.source_classes)
        expected_positions = [
            position for position in expected_positions if labels[position] in sources
        ]
    assert len(dataset) == len(expected_positions)
    for index, position in enumerate(expected_positions):
        _, target = dataset[index]
        expected = attack_success_label(
            attack.label_mode,
            labels[position],
            attack.target_label,
            num_classes,
            attack.num_targets,
        )
        assert target == expected


def test_adversarial_config_without_bases_is_refused():
    """An epsilon with no directory is incoherent and must be caught before training.

    This is the state a dropped `--attack-override` produced: nothing is out of
    range and no type is violated, so the attack silently reverts to its patch-only
    variant while the recorded metadata claims the adversarial one.
    """
    from attacks import adversarial_config_error

    default = default_config("lc")
    assert adversarial_config_error(default) is None, "the patch-only default is valid"

    both_set = replace(default, adversarial_dir="bases/", adversarial_epsilon=0.0627)
    assert adversarial_config_error(both_set) is None, "a complete config is valid"

    epsilon_only = replace(default, adversarial_epsilon=0.0627)
    message = adversarial_config_error(epsilon_only)
    assert message is not None and "adversarial_dir" in message


def test_missing_adversarial_bases_names_the_gap(tmp_path):
    """Training must refuse when a poisoned index has no perturbed base."""
    from attacks import missing_adversarial_bases

    directory = tmp_path / "bases"
    directory.mkdir()
    torch.save(
        {"indices": torch.tensor([1, 2]), "images": torch.zeros(2, 3, SIZE, SIZE)},
        directory / "bases.pt",
    )
    config = replace(default_config("lc"), adversarial_dir=str(directory))

    assert missing_adversarial_bases(config, [1, 2]) == []
    assert missing_adversarial_bases(config, [1, 2, 7, 9]) == [7, 9]
    # An empty directory setting means the patch-only variant, which needs no bases.
    assert missing_adversarial_bases(default_config("lc"), [1, 2, 7]) == []


def test_multi_target_clean_label_widens_both_pools_as_complements():
    """More targets means a bigger training pool and a smaller eval pool.

    A clean-label attack can only poison images that already carry a target
    label, so a single target caps it at 1/K of a balanced training set: 1% on
    CIFAR-100 and 0.5% on Tiny. The 2 pools stay exact complements however wide
    the set gets, because eval asks the opposite question of training.
    """
    from attacks.poisoning import (
        clean_label_target_set,
        is_eval_poisonable,
        is_poisonable,
    )

    labels = list(range(10))
    for num_targets in (1, 2, 3):
        targets = clean_label_target_set(0, num_targets)
        assert len(targets) == num_targets

        train = {
            y for y in labels if is_poisonable("clean_label_multi", y, 0, num_targets)
        }
        evaluate = {
            y
            for y in labels
            if is_eval_poisonable("clean_label_multi", y, 0, num_targets)
        }
        assert train == set(targets)
        assert train.isdisjoint(evaluate)
        assert train | evaluate == set(labels)


def test_multi_target_keeps_labels_which_is_what_makes_it_clean_label():
    """The training label is never changed, however many targets there are."""
    from attacks.poisoning import poisoned_label

    for y in (0, 1, 2, 7):
        assert poisoned_label("clean_label_multi", y, 0, 10, 3) == y


def test_single_target_clean_label_is_untouched_by_the_multi_target_work():
    """num_targets=1 must reproduce the original mode exactly.

    Every clean-label checkpoint trained before num_targets existed rebuilds
    through the default config, so a drift here silently reinterprets them.
    """
    from attacks.poisoning import is_eval_poisonable, is_poisonable

    config = default_config("sig")
    assert config.num_targets == 1
    assert build_attack("sig", config, SIZE, 0).label_mode == "clean_label"

    for y in (0, 1, 5):
        assert is_poisonable("clean_label_multi", y, 0, 1) == is_poisonable(
            "clean_label", y, 0
        )
        assert is_eval_poisonable("clean_label_multi", y, 0, 1) == is_eval_poisonable(
            "clean_label", y, 0
        )


@pytest.mark.parametrize("name", TESTABLE_ATTACK_NAMES)
def test_builder_propagates_every_config_field_the_attack_record_also_carries(name):
    """A builder must not silently drop a config field that Attack can hold.

    TaCT's source_classes is the case that motivated this. A builder that forgets
    it still returns a working Attack, and the trigger pixels stay bit-identical,
    so a test comparing only name, label mode, target and pixels passes while ASR
    is quietly measured over every non-target class instead of the source classes.
    Comparing the fields the two dataclasses share catches the whole family.
    """
    config = default_config(name)
    attack = build_attack(name, config, 32, 0)

    shared = {field.name for field in fields(type(config))} & {
        field.name for field in fields(Attack)
    }
    assert shared, f"{name}: no shared fields, so this test would be vacuous"

    for field_name in sorted(shared):
        assert getattr(attack, field_name) == getattr(config, field_name), (
            f"{name}: build() dropped {field_name}"
        )
