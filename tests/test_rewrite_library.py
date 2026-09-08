"""Equivalence of the psbd/ library rewrite against the modules it replaces.

The second half of the cutover: attacks, analysis, evaluation, evasion, the
baseline detectors, stealth, and training provenance. As with the first half, the
old modules stay the source of truth until every call site moves, so the new ones
have to agree with them exactly rather than merely look right.

Two areas carry the most risk and get the most scrutiny.

  1. The attacks. A trigger is the definition of the experiment, and 5 of the 10
     attacks now build their pattern from a shared helper rather than from a
     private copy. Every attack is compared bit-for-bit at several image sizes,
     and the 2 deduplicated helpers are additionally compared against all 5
     originals directly, so a drift cannot hide behind a passing end-to-end test.
  2. The ASR measurement. Attack success is counted over AttackSuccessSet with
     eval-time eligibility, which for a clean-label attack is the opposite
     population from training-time eligibility. The loader and evaluation tests
     compare the served samples, not just the final number.

The checkpoint-backed and dataset-backed tests skip when raw_data/ or the
checkpoint is absent. Everything else is pure and always runs.
"""

import dataclasses
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from PIL import Image
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.models.vision_transformer import VisionTransformer

import adaptive_evasion as old_evasion
import analysis.cka as old_cka
import analysis.direction as old_direction
import analysis.features as old_features
import analysis.lipschitz as old_lipschitz
import attacks as old_attacks
import backdoor_data as old_backdoorbench
import defences.baselines as old_baselines
import defences.detection as old_detection
import evaluate as old_evaluate
import loaders as old_loaders
import stealth as old_stealth
import train as old_training
from psbd import analysis as _new_analysis_package  # noqa: F401
from psbd import attacks as new_attacks
from psbd import backdoorbench as new_backdoorbench
from psbd import baselines as new_baselines
from psbd import eval_loaders as new_loaders
from psbd import evaluation as new_evaluation
from psbd import evasion as new_evasion
from psbd import models as new_models
from psbd import stealth as new_stealth
from psbd import training as new_training
from psbd.analysis import cka as new_cka
from psbd.analysis import direction as new_direction
from psbd.analysis import features as new_features
from psbd.analysis import lipschitz as new_lipschitz
from psbd.attacks import _patterns as new_patterns

RAW_DATA_DIR = "raw_data"

# A dirty-label ViT checkpoint on the smallest dataset, so the real-model tests
# stay a few seconds even on a shared login node.
EVAL_CHECKPOINT = "checkpoints/vit_cifar10_badnet_a2o_0_01/attack_result.pt"
EVAL_DATASET = "cifar10"
EVAL_ATTACK = "badnet_a2o"
EVAL_MAX_SAMPLES = 256

# "generated" reads pregenerated PNGs from disk and has no default config, so it
# is compared separately against a fixture directory.
COMPARED_ATTACK_NAMES = tuple(
    name for name in new_attacks.ATTACK_NAMES if name != "generated"
)

# 32 and 64 are the 2 native trigger resolutions in the dataset registry. 17 is
# odd and not a multiple of any patch size, which is where an off-by-one in a
# corner stamp or an fft shift would show up.
TRIGGER_IMAGE_SIZES = (32, 64, 17)
TRIGGER_INDICES = (0, 1, 7, 123)

SYNTHETIC_CLASSES = 4


def _skip_unless_present(*paths: str) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        pytest.skip(f"not on disk: {missing}")


def _images(image_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, image_size, image_size, generator=generator)


@pytest.fixture(scope="module")
def synthetic_classifier() -> nn.Module:
    """A deterministic tiny classifier, so a detector's arithmetic can be compared alone."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, SYNTHETIC_CLASSES)).eval()


@pytest.fixture(scope="module")
def synthetic_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(24, 3, 8, 8, generator=generator)
    labels = torch.randint(0, SYNTHETIC_CLASSES, (24,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)


@pytest.fixture(scope="module")
def tiny_vit() -> nn.Module:
    """A ViT with the real block structure but tiny dimensions.

    The probe registry resolves positions by module type and dotted name, so a
    2-block VisionTransformer exercises exactly the same plug path as ViT-B/16 at
    a fraction of the cost.
    """
    torch.manual_seed(0)
    network = VisionTransformer(
        image_size=32,
        patch_size=16,
        num_layers=2,
        num_heads=2,
        hidden_dim=16,
        mlp_dim=32,
        num_classes=SYNTHETIC_CLASSES,
    )

    # torchvision zero-initializes the classification head, so an untrained
    # VisionTransformer returns the same all-zero logits for every input. Any
    # assertion made on its logits then holds no matter what the code under test
    # did, including a probe-leak check that can never fail. Giving the head real
    # weights is what makes those assertions able to detect anything.
    nn.init.normal_(network.heads.head.weight, std=0.05)
    nn.init.normal_(network.heads.head.bias, std=0.05)

    return nn.Sequential(transforms_v2.Resize((32, 32)), network)


def test_the_shared_fixture_is_not_degenerate(tiny_vit):
    """A zero head makes every logit assertion in this file vacuous.

    torchvision zero-initializes VisionTransformer's head, so an untrained model
    returns identical all-zero logits for any input. Tests that compare logits, or
    that check a probe left no trace by re-running the model, then pass no matter
    what the code under test did. This asserts the fixture is a real function of
    its input, so those tests can fail.
    """
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(4, 3, 32, 32, generator=generator)

    with torch.no_grad():
        logits = tiny_vit(images)

    assert logits.abs().max() > 0, "the head is zero, so logit assertions cannot fail"
    assert not torch.allclose(logits[0], logits[1]), (
        "different images must produce different logits"
    )


def test_attack_names_match_exactly():
    assert new_attacks.ATTACK_NAMES == old_attacks.ATTACK_NAMES


@pytest.mark.parametrize("attack_name", COMPARED_ATTACK_NAMES)
def test_default_config_fields_match(attack_name):
    new_config = new_attacks.default_config(attack_name)
    old_config = old_attacks.default_config(attack_name)

    # The 2 config classes are distinct dataclasses, so dataclass equality is
    # False by construction; the contract is that their fields agree.
    assert dataclasses.asdict(new_config) == dataclasses.asdict(old_config)
    assert type(new_config).__name__ == type(old_config).__name__


def test_generated_needs_an_explicit_config_in_both():
    with pytest.raises(ValueError):
        new_attacks.default_config("generated")
    with pytest.raises(ValueError):
        old_attacks.default_config("generated")


@pytest.mark.parametrize("attack_name", COMPARED_ATTACK_NAMES)
@pytest.mark.parametrize("image_size", TRIGGER_IMAGE_SIZES)
def test_trigger_is_bit_identical_to_the_original(attack_name, image_size):
    new_attack = new_attacks.build_attack(
        attack_name, new_attacks.default_config(attack_name), image_size, target_label=3
    )
    old_attack = old_attacks.build_attack(
        attack_name, old_attacks.default_config(attack_name), image_size, target_label=3
    )

    assert new_attack.name == old_attack.name
    assert new_attack.label_mode == old_attack.label_mode
    assert new_attack.target_label == old_attack.target_label

    for seed, index in enumerate(TRIGGER_INDICES):
        image = _images(image_size, seed)
        new_triggered = new_attack.apply_trigger(image.clone(), index)
        old_triggered = old_attack.apply_trigger(image.clone(), index)
        assert torch.equal(new_triggered, old_triggered), (
            attack_name,
            image_size,
            index,
        )


def test_bpp_dithered_trigger_is_bit_identical():
    """The dithering branch is off by default, so the parametrized sweep never reaches it."""
    new_attack = new_attacks.build_attack(
        attack_name="bpp",
        config=new_attacks.bpp.BppConfig(dither=True),
        image_size=16,
        target_label=0,
    )
    old_attack = old_attacks.build_attack(
        "bpp", old_attacks.bpp.BppConfig(dither=True), 16, 0
    )
    image = _images(16, seed=9)

    assert torch.equal(
        new_attack.apply_trigger(image.clone(), 0),
        old_attack.apply_trigger(image.clone(), 0),
    )


@pytest.mark.parametrize("patch_size", (1, 2, 3, 5, 8))
def test_shared_checkerboard_reproduces_all_3_originals(patch_size):
    """The deduplication proof for the patch builder badnet, lc and tact each had."""
    shared = new_patterns.checkerboard_patch(patch_size)

    assert torch.equal(shared, old_attacks.badnet._checkerboard(patch_size))
    assert torch.equal(shared, old_attacks.lc._corner_pattern(patch_size))
    assert torch.equal(shared, old_attacks.tact._patch(patch_size))


@pytest.mark.parametrize("image_size", TRIGGER_IMAGE_SIZES)
@pytest.mark.parametrize("seed", (0, 1, 42))
def test_shared_random_pattern_reproduces_both_originals(image_size, seed):
    """The deduplication proof for the pattern blend and adaptive_blend each had."""
    shared = new_patterns.seeded_random_pattern(image_size, seed)

    assert torch.equal(shared, old_attacks.blend._random_pattern(image_size, seed))
    assert torch.equal(
        shared, old_attacks.adaptive_blend._random_pattern(image_size, seed)
    )


def test_generated_adapter_matches_on_a_fixture_directory(tmp_path):
    poisoned_dir = tmp_path / "poisoned"
    poisoned_dir.mkdir()
    for index in (0, 1, 2):
        array = ((np.arange(16 * 16 * 3) + index) % 256).astype(np.uint8)
        Image.fromarray(array.reshape(16, 16, 3)).save(poisoned_dir / f"{index}.png")

    new_attack = new_attacks.build_attack(
        "generated",
        new_attacks.generated.GeneratedConfig(poisoned_dir=str(poisoned_dir)),
        image_size=16,
        target_label=0,
    )
    old_attack = old_attacks.build_attack(
        "generated",
        old_attacks.generated.GeneratedConfig(poisoned_dir=str(poisoned_dir)),
        16,
        0,
    )

    for index in (0, 1, 2):
        clean = _images(16, seed=index)
        assert torch.equal(
            new_attack.apply_trigger(clean.clone(), index),
            old_attack.apply_trigger(clean.clone(), index),
        )


def test_counting_primitives_match(synthetic_classifier, synthetic_loader):
    device = torch.device("cpu")

    assert new_evaluation.prediction_accuracy(
        synthetic_classifier, synthetic_loader, device, use_bfloat16=False
    ) == old_detection._prediction_accuracy(
        synthetic_classifier, synthetic_loader, device, use_bfloat16=False
    )
    assert new_evaluation.clean_accuracy(
        synthetic_classifier, synthetic_loader, device, use_bfloat16=False
    ) == old_detection.clean_accuracy(
        synthetic_classifier, synthetic_loader, device, use_bfloat16=False
    )
    assert new_evaluation.attack_success_rate(
        synthetic_classifier, synthetic_loader, device, use_bfloat16=False
    ) == old_detection.attack_success_rate(
        synthetic_classifier, synthetic_loader, device, use_bfloat16=False
    )

    new_correct, new_total = new_evaluation.class_correct_and_total(
        synthetic_classifier, synthetic_loader, device, SYNTHETIC_CLASSES, False
    )
    old_correct, old_total = old_detection.class_correct_and_total(
        synthetic_classifier, synthetic_loader, device, SYNTHETIC_CLASSES, False
    )
    assert torch.equal(new_correct, old_correct)
    assert torch.equal(new_total, old_total)

    assert new_evaluation.accuracy_by_class_from_counts(
        new_correct, new_total
    ) == old_detection.accuracy_by_class_from_counts(old_correct, old_total)
    assert new_evaluation.pooled_accuracy_from_counts(
        new_correct, new_total
    ) == old_detection.pooled_accuracy_from_counts(old_correct, old_total)
    assert new_evaluation.clean_accuracy_by_class(
        synthetic_classifier, synthetic_loader, device, SYNTHETIC_CLASSES, False
    ) == old_detection.clean_accuracy_by_class(
        synthetic_classifier, synthetic_loader, device, SYNTHETIC_CLASSES, False
    )


def test_threshold_and_detection_helpers_match():
    torch.manual_seed(0)
    validation = torch.randn(500)
    clean = torch.randn(300)
    backdoor = torch.randn(300) - 1.0

    for quantile in (0.01, 0.1, 0.25):
        new_threshold = new_evaluation.threshold_from_validation(validation, quantile)
        old_threshold = old_detection.threshold_from_validation(validation, quantile)
        assert new_threshold == old_threshold
        assert new_evaluation.detection_rates(
            clean, backdoor, new_threshold
        ) == old_detection.detection_rates(clean, backdoor, old_threshold)

    assert new_evaluation.auroc(clean, backdoor) == old_detection.auroc(clean, backdoor)


# badnet_a2o is dirty-label and sig is clean-label. The clean-label case is the
# one whose eval eligibility is the opposite of its training eligibility, so it is
# where mixing up the 2 function pairs would change which samples are served.
@pytest.mark.parametrize("attack_name", ("badnet_a2o", "sig"))
def test_eval_loaders_serve_identical_samples(attack_name):
    _skip_unless_present(RAW_DATA_DIR)

    new_attack = new_attacks.build_attack(
        attack_name, new_attacks.default_config(attack_name), 32, target_label=0
    )
    old_attack = old_attacks.build_attack(
        attack_name, old_attacks.default_config(attack_name), 32, target_label=0
    )

    new_clean = new_loaders.build_clean_loader(
        EVAL_DATASET, RAW_DATA_DIR, batch_size=32, num_workers=0, max_samples=64
    )
    old_clean = old_loaders.build_clean_loader(
        EVAL_DATASET, RAW_DATA_DIR, batch_size=32, num_workers=0, max_samples=64
    )
    new_poisoned = new_loaders.build_poisoned_loader(
        EVAL_DATASET, new_attack, RAW_DATA_DIR, 32, num_workers=0, max_samples=64
    )
    old_poisoned = old_loaders.build_poisoned_loader(
        EVAL_DATASET, old_attack, RAW_DATA_DIR, 32, num_workers=0, max_samples=64
    )

    for new_loader, old_loader in (
        (new_clean, old_clean),
        (new_poisoned, old_poisoned),
    ):
        assert len(new_loader.dataset) == len(old_loader.dataset)
        for (new_images, new_labels), (old_images, old_labels) in zip(
            new_loader, old_loader
        ):
            assert torch.equal(new_images, old_images)
            assert torch.equal(torch.as_tensor(new_labels), torch.as_tensor(old_labels))

    # The ASR population is the whole point of the poisoned loader. Both label
    # modes here drop the already-target-class images at eval time, so the
    # poisoned loader must be strictly shorter than the clean one. A clean-label
    # attack served through the TRAINING-time pair would instead keep only the
    # target class, which is the mistake this comparison catches.
    assert len(new_poisoned.dataset) < len(new_clean.dataset)
    assert new_poisoned.dataset.indices == old_poisoned.dataset.indices
    served_labels = [int(new_clean.dataset[i][1]) for i in new_poisoned.dataset.indices]
    assert all(label != 0 for label in served_labels)


def test_evaluation_matches_on_a_real_checkpoint():
    _skip_unless_present(RAW_DATA_DIR, EVAL_CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_models.load_checkpoint("vit", EVAL_CHECKPOINT, device)

    new_metrics = new_evaluation.evaluate_attack(
        model,
        EVAL_DATASET,
        EVAL_ATTACK,
        new_attacks.default_config(EVAL_ATTACK),
        0,
        device,
        RAW_DATA_DIR,
        64,
        max_samples=EVAL_MAX_SAMPLES,
        seed=0,
    )
    old_metrics = old_evaluate.evaluate_attack(
        model,
        EVAL_DATASET,
        EVAL_ATTACK,
        old_attacks.default_config(EVAL_ATTACK),
        0,
        device,
        RAW_DATA_DIR,
        64,
        max_samples=EVAL_MAX_SAMPLES,
        seed=0,
    )
    assert new_metrics == old_metrics

    # A backdoored checkpoint under its own trigger: a near-0 ASR would mean the
    # eval set was built wrong, and both implementations would agree on nothing.
    assert new_metrics["asr"] > 0.5
    assert new_metrics["clean_accuracy"] > 0.5

    new_benign = new_evaluation.evaluate_benign(
        model, EVAL_DATASET, device, RAW_DATA_DIR, 64, EVAL_MAX_SAMPLES, 0
    )
    old_benign = old_evaluate.evaluate_benign(
        model, EVAL_DATASET, device, RAW_DATA_DIR, 64, EVAL_MAX_SAMPLES, 0
    )
    assert new_benign == old_benign


def test_baseline_detectors_match(synthetic_classifier, synthetic_loader):
    device = torch.device("cpu")

    new_confidence = new_baselines.confidence_scores(
        synthetic_classifier, synthetic_loader, device, use_bfloat16=False
    )
    old_confidence = old_baselines.confidence_scores(
        synthetic_classifier, synthetic_loader, device, use_bfloat16=False
    )
    assert torch.equal(new_confidence, old_confidence)

    new_overlays = new_baselines.collect_overlay_batch(synthetic_loader, 8, seed=0)
    old_overlays = old_baselines.collect_overlay_batch(synthetic_loader, 8, seed=0)
    assert torch.equal(new_overlays, old_overlays)

    mean = (0.5, 0.5, 0.5)
    std = (0.25, 0.25, 0.25)
    new_strip = new_baselines.strip_scores(
        synthetic_classifier,
        synthetic_loader,
        new_overlays,
        mean,
        std,
        device,
        False,
        seed=0,
    )
    old_strip = old_baselines.strip_scores(
        synthetic_classifier,
        synthetic_loader,
        old_overlays,
        mean,
        std,
        device,
        False,
        seed=0,
    )
    assert torch.equal(new_strip, old_strip)

    # STRIP's entropy is NOT negated, so it must be non-negative. A negated copy
    # is the sign error that once produced AUROC 0.000.
    assert (new_strip >= 0).all()


@pytest.mark.parametrize(
    "poisoned_flags",
    (
        [0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],  # no poisoned sample in the batch
        [1, 1, 1, 1, 1, 1],  # no clean sample in the batch
    ),
)
def test_evasion_penalty_matches(poisoned_flags):
    torch.manual_seed(0)
    psu = torch.randn(len(poisoned_flags), requires_grad=True)
    is_poisoned = torch.tensor(poisoned_flags)

    new_penalty = new_evasion.evasion_penalty(psu, is_poisoned)
    old_penalty = old_evasion.evasion_penalty(psu, is_poisoned)

    assert torch.equal(new_penalty.detach(), old_penalty.detach())
    assert new_penalty.requires_grad == old_penalty.requires_grad


def test_psu_for_batch_matches(tiny_vit):
    probe = {
        "position": "pre_residual",
        "operator": "dropout",
        "architecture": "vit",
        "rate": 0.3,
    }
    images = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(2))

    torch.manual_seed(7)
    new_psu, new_logits = new_evasion.psu_for_batch(tiny_vit, images, probe, passes=3)
    torch.manual_seed(7)
    old_psu, old_logits = old_evasion.psu_for_batch(tiny_vit, images, probe, passes=3)

    assert torch.equal(new_logits.detach(), old_logits.detach())
    assert torch.equal(new_psu.detach(), old_psu.detach())
    assert new_psu.requires_grad

    # The probe must be gone afterwards, or it leaks into the next clean forward.
    torch.manual_seed(0)
    first = tiny_vit(images)
    torch.manual_seed(0)
    second = tiny_vit(images)
    assert torch.equal(first, second)


def _synthetic_features(seed: int, num_samples: int = 40, dim: int = 12):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_samples, dim, generator=generator)


def test_cka_variants_match():
    features_x = _synthetic_features(0)
    features_y = _synthetic_features(1)

    assert new_cka.linear_cka(features_x, features_y) == old_cka.linear_cka(
        features_x, features_y
    )
    assert new_cka.debiased_linear_cka(
        features_x, features_y
    ) == old_cka.debiased_linear_cka(features_x, features_y)
    assert new_cka.rbf_cka(features_x, features_y) == old_cka.rbf_cka(
        features_x, features_y
    )
    assert new_cka.debiased_rbf_cka(features_x, features_y) == old_cka.debiased_rbf_cka(
        features_x, features_y
    )

    by_layer_a = {layer: _synthetic_features(10 + layer) for layer in range(3)}
    by_layer_b = {layer: _synthetic_features(20 + layer) for layer in range(3)}
    assert torch.equal(
        new_cka.layerwise_cka_matrix(by_layer_a, by_layer_b),
        old_cka.layerwise_cka_matrix(by_layer_a, by_layer_b),
    )
    assert torch.equal(
        new_cka.layerwise_cka_matrix(by_layer_a, by_layer_b, debiased=False),
        old_cka.layerwise_cka_matrix(by_layer_a, by_layer_b, debiased=False),
    )

    with pytest.raises(ValueError):
        new_cka.unbiased_hsic(torch.eye(3).double(), torch.eye(3).double())


def test_direction_tac_and_weight_tools_match():
    clean = _synthetic_features(2)
    backdoor = clean + 0.3 * _synthetic_features(3)

    new_dir = new_direction.backdoor_direction(clean, backdoor)
    old_dir = old_direction.backdoor_direction(clean, backdoor)
    assert torch.equal(new_dir, old_dir)

    assert torch.equal(
        new_direction.trigger_activated_change(clean, backdoor),
        old_direction.trigger_activated_change(clean, backdoor),
    )
    assert torch.equal(
        new_direction.project_onto_direction(clean, new_dir),
        old_direction.project_onto_direction(clean, old_dir),
    )
    assert torch.equal(
        new_direction.outlier_dimensions(new_dir.abs(), sensitivity=0.5),
        old_direction.outlier_dimensions(old_dir.abs(), sensitivity=0.5),
    )

    weight = _synthetic_features(4, num_samples=12, dim=6)
    assert torch.equal(
        new_direction.orthogonalize_weight(weight, new_dir),
        old_direction.orthogonalize_weight(weight, old_dir),
    )

    hidden = torch.randn(2, 5, 12, generator=torch.Generator().manual_seed(5))
    new_hook = new_direction.make_steering_hook(new_dir, scale=0.5)
    old_hook = old_direction.make_steering_hook(old_dir, scale=0.5)
    assert torch.equal(new_hook(None, None, hidden), old_hook(None, None, hidden))


def test_layer_features_match_after_the_vit_core_rename(tiny_vit):
    """psbd.analysis resolves the network through network_core, not the old vit_core."""
    generator = torch.Generator().manual_seed(4)
    images = torch.rand(6, 3, 32, 32, generator=generator)
    labels = torch.zeros(6, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=3, shuffle=False)
    device = torch.device("cpu")

    for reduction in ("cls", "mean", "flatten"):
        new_by_layer = new_features.extract_layer_features(
            tiny_vit, loader, device, use_bfloat16=False, reduction=reduction
        )
        old_by_layer = old_features.extract_layer_features(
            tiny_vit, loader, device, use_bfloat16=False, reduction=reduction
        )
        assert new_by_layer.keys() == old_by_layer.keys()
        for layer in new_by_layer:
            assert torch.equal(new_by_layer[layer], old_by_layer[layer])

    # The hooks must leave nothing behind, or the next forward is not the model's.
    assert not any(block._forward_hooks for block in tiny_vit[1].encoder.layers)


def test_lipschitz_tools_match(tiny_vit):
    new_by_block = new_lipschitz.mlp_output_channel_lipschitz(tiny_vit)
    old_by_block = old_lipschitz.mlp_output_channel_lipschitz(tiny_vit)
    assert new_by_block.keys() == old_by_block.keys()
    for block in new_by_block:
        assert torch.equal(new_by_block[block], old_by_block[block])

    weight = tiny_vit[1].encoder.layers[0].mlp[0].weight
    assert new_lipschitz.spectral_norm(weight) == old_lipschitz.spectral_norm(weight)
    assert torch.equal(
        new_lipschitz.linear_channel_lipschitz(weight),
        old_lipschitz.linear_channel_lipschitz(weight),
    )

    new_alignment = new_lipschitz.head_weight_alignment(tiny_vit, 2, 0.9)
    old_alignment = old_lipschitz.head_weight_alignment(tiny_vit, 2, 0.9)
    assert torch.equal(new_alignment, old_alignment)
    assert new_lipschitz.alignment_outlier_score(
        new_alignment
    ) == old_lipschitz.alignment_outlier_score(old_alignment)


def test_stealth_metrics_match():
    generator = torch.Generator().manual_seed(0)
    clean = torch.rand(6, 3, 32, 32, generator=generator)
    triggered = (clean + 0.05 * torch.rand(6, 3, 32, 32, generator=generator)).clamp(
        0, 1
    )
    device = torch.device("cpu")

    assert torch.equal(
        new_stealth.per_image_psnr(clean, triggered),
        old_stealth._per_image_psnr(clean, triggered),
    )
    assert torch.equal(
        new_stealth.per_image_ssim(clean, triggered),
        old_stealth._per_image_ssim(clean, triggered),
    )
    assert new_stealth.compute_stealth_metrics(
        clean, triggered, device, batch_size=4
    ) == old_stealth.compute_stealth_metrics(clean, triggered, device, batch_size=4)


def _labelled_dataset(num_samples: int, image_size: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(num_samples, 3, image_size, image_size, generator=generator)
    labels = torch.arange(num_samples) % SYNTHETIC_CLASSES
    return TensorDataset(images, labels)


def test_backdoorbench_splitting_matches():
    clean = _labelled_dataset(80, 8, seed=0)
    backdoor = _labelled_dataset(80, 8, seed=1)

    new_val, new_eval, new_bd = new_backdoorbench.split_validation_and_eval(
        clean, backdoor, clean_val_size=20, seed=0
    )
    old_val, old_eval, old_bd = old_backdoorbench.split_validation_and_eval(
        clean, backdoor, clean_val_size=20, seed=0
    )
    assert new_val.indices == old_val.indices
    assert new_eval.indices == old_eval.indices
    assert new_bd.indices == old_bd.indices

    new_balanced_clean, new_balanced_bd = new_backdoorbench.balance_by_class(
        new_eval, new_bd, examples_per_class=3, seed=0
    )
    old_balanced_clean, old_balanced_bd = old_backdoorbench.balance_by_class(
        old_eval, old_bd, examples_per_class=3, seed=0
    )
    assert new_balanced_clean.indices == old_balanced_clean.indices
    assert new_balanced_bd.indices == old_balanced_bd.indices


@pytest.mark.parametrize("label_mode", ("all_to_one", "all_to_all", "clean_label"))
def test_png_backed_backdoor_splits_match(tmp_path, label_mode):
    """The PNG path must apply the same eval-time eligibility as the in-memory path."""
    clean = _labelled_dataset(16, 8, seed=2)
    png_dir = tmp_path / "weights" / "folder" / "bd_test_dataset"
    png_dir.mkdir(parents=True)
    for index in range(len(clean)):
        array = (clean[index][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(array).save(png_dir / f"{index}.png")

    transform = transforms_v2.Compose(
        [transforms_v2.Resize((8, 8)), transforms_v2.ToTensor()]
    )
    arguments = (
        "folder",
        clean,
        transform,
        str(tmp_path / "weights"),
        label_mode,
        1,
        SYNTHETIC_CLASSES,
    )
    new_bd, new_counterparts = new_backdoorbench.load_backdoor_splits(*arguments)
    old_bd, old_counterparts = old_backdoorbench.load_backdoor_splits(*arguments)

    assert new_bd.eligible_positions == old_bd.eligible_positions
    assert new_counterparts.indices == old_counterparts.indices
    assert len(new_bd) == len(old_bd)
    for position in range(len(new_bd)):
        new_image, new_label = new_bd[position]
        old_image, old_label = old_bd[position]
        assert torch.equal(new_image, old_image)
        assert new_label == old_label


METADATA_ARGUMENTS = {
    "dataset": "cifar10",
    "attack": "badnet_a2o",
    "label_mode": "all_to_one",
    "target_label": 0,
    "poison_rate": 0.01,
    "cover_rate": 0.0,
    "realized_poison_rate": 0.008,
    "architecture": "vit",
    "rho": 0.1,
    "epochs": 15,
    "seed": 0,
    "max_samples": None,
    "clean_accuracy": 0.95,
    "asr": 0.99,
    "started_at": "2026-01-01T00:00:00+00:00",
    "ended_at": "2026-01-01T01:00:00+00:00",
    "evasion": {"weight": 1.0, "passes": 2},
    "model_dropout": 0.0,
}


@pytest.mark.parametrize("use_sam", (True, False))
def test_checkpoint_metadata_matches(use_sam):
    new_metadata = new_training.checkpoint_metadata(
        use_sam=use_sam, **METADATA_ARGUMENTS
    )
    old_metadata = old_training.checkpoint_metadata(
        use_sam=use_sam, **METADATA_ARGUMENTS
    )

    assert list(new_metadata) == list(old_metadata)
    assert new_metadata == old_metadata
    assert "realized_poison_rate" in new_metadata
    assert new_metadata["optimizer"] == ("sam" if use_sam else "adam")
    assert new_metadata["rho"] == (0.1 if use_sam else None)


def test_build_model_and_optimizer_match_the_original():
    new_model = new_training.build_model("vit", SYNTHETIC_CLASSES)
    old_model = old_training.build_model("vit", SYNTHETIC_CLASSES)
    assert new_model.state_dict().keys() == old_model.state_dict().keys()

    with pytest.raises(ValueError):
        new_training.build_model("resnet", SYNTHETIC_CLASSES)
    with pytest.raises(ValueError):
        new_training.build_model("swin", SYNTHETIC_CLASSES, model_dropout=0.1)

    plain = new_training.build_optimizer(new_model, False, 1e-4, 1e-4, 0.1)
    sharpness_aware = new_training.build_optimizer(new_model, True, 1e-4, 1e-4, 0.1)
    assert isinstance(plain, torch.optim.Adam)
    assert hasattr(sharpness_aware, "first_step")
    assert hasattr(sharpness_aware, "second_step")


def test_training_update_helpers_match_on_a_tiny_model():
    """Both updates must move the weights identically to the originals'."""

    def fresh_model() -> nn.Module:
        torch.manual_seed(0)
        return nn.Sequential(nn.Flatten(), nn.Linear(12, SYNTHETIC_CLASSES))

    images = torch.rand(4, 3, 2, 2, generator=torch.Generator().manual_seed(3))
    labels = torch.tensor([0, 1, 2, 3])
    criterion = nn.CrossEntropyLoss()

    for new_update, old_update, use_sam in (
        (new_training.plain_update, old_training._plain_update, False),
        (new_training.sam_update, old_training._sam_update, True),
    ):
        new_model = fresh_model()
        old_model = fresh_model()
        new_optimizer = new_training.build_optimizer(new_model, use_sam, 1e-2, 0.0, 0.1)
        old_optimizer = old_training.build_optimizer(old_model, use_sam, 1e-2, 0.0, 0.1)

        new_loss = new_update(new_model, images, labels, criterion, new_optimizer)
        old_loss = old_update(old_model, images, labels, criterion, old_optimizer)

        assert torch.equal(new_loss.detach(), old_loss.detach())
        for new_parameter, old_parameter in zip(
            new_model.parameters(), old_model.parameters()
        ):
            assert torch.equal(new_parameter, old_parameter)


def test_subset_helper_keeps_paired_alignment():
    """Both split helpers must return Subsets, which is what preserves the pairing."""
    clean = _labelled_dataset(40, 8, seed=6)
    backdoor = _labelled_dataset(40, 8, seed=7)
    val, clean_eval, backdoor_eval = new_backdoorbench.split_validation_and_eval(
        clean, backdoor, clean_val_size=8, seed=0
    )

    assert isinstance(val, Subset)
    assert clean_eval.indices == backdoor_eval.indices
    assert set(val.indices).isdisjoint(clean_eval.indices)
