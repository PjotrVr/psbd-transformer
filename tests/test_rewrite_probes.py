"""Equivalence tests for the psbd/ probe rewrite against the defences/ originals.

The rewrite splits the old defences package along one seam: psbd.positions owns
WHERE a probe attaches, psbd.operators owns WHAT it does, and psbd.scores and
psbd.decision split the old psbd_metrics into "what number a sample gets" and
"what that number means". A split like that is only worth doing if it changes
nothing measurable, so every test here compares the new implementation against
the old one on the same inputs.

Three properties carry the most risk and get the most scrutiny:

  1. Plugging must remain a pure attachment. Both implementations must register
     the same number of handles per position, and unplugging must leave the
     loaded state_dict byte-identical, on real checkpoints of both architectures.
  2. The cache layout is a fixed on-disk contract. Hundreds of thousands of
     cached tensors already exist, so every path string must match exactly,
     including the %g rate tag that keeps 0.01 through 0.09 distinct.
  3. Scores computed from real cached tensors must agree with the originals.
"""

import glob
import os

import pytest
import torch
import torch.nn as nn

import defences.dropout as old_positions
import defences.perturbations as old_operators
import defences.psbd_cache as old_cache
import defences.psbd_metrics as old_metrics
import psbd.cache as new_cache
import psbd.decision as new_decision
import psbd.operators as new_operators
import psbd.positions as new_positions
import psbd.scores as new_scores
from models import load_checkpoint

CHECKPOINT_ROOT = "checkpoints"
RESULTS_ROOT = "results"

# ViT carries (batch, tokens, channels), Swin carries (batch, H, W, channels),
# and the head-axis operators see (batch, heads, tokens, dim).
TOKEN_SHAPE = (3, 7, 16)
SPATIAL_SHAPE = (3, 4, 4, 16)
HEAD_SHAPE = (3, 4, 7, 8)

TOKEN_LAYOUT_OPERATORS = (
    "dropout",
    "channel_mask",
    "token_mask",
    "droppath",
    "gaussian",
    "gain_scale",
)
HEAD_LAYOUT_OPERATORS = ("head_mask",)


def _find_checkpoint(architecture: str) -> str | None:
    """The first real checkpoint of an architecture, by folder-name convention."""
    pattern = os.path.join(CHECKPOINT_ROOT, f"{architecture}_*", "attack_result.pt")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


@pytest.fixture(scope="module")
def vit_checkpoint() -> nn.Module:
    path = _find_checkpoint("vit")
    if path is None:
        pytest.skip("no ViT checkpoint on disk")
    return load_checkpoint("vit", path, torch.device("cpu"))


@pytest.fixture(scope="module")
def swin_checkpoint() -> nn.Module:
    path = _find_checkpoint("swin")
    if path is None:
        pytest.skip("no Swin checkpoint on disk")
    return load_checkpoint("swin", path, torch.device("cpu"))


def _state_dict_snapshot(model: nn.Module) -> dict:
    return {key: value.clone() for key, value in model.state_dict().items()}


def _assert_state_dict_identical(before: dict, after: dict) -> None:
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} changed under plug/unplug"


def _plug_and_count(module, model: nn.Module, architecture: str, position: str) -> int:
    handles = module.plug_dropout(model, architecture, (position,), {}, rate=0.3)
    count = len(handles)
    module.unplug_dropout(handles)
    return count


@pytest.mark.parametrize("position", sorted(old_positions.VIT_POSITIONS))
def test_vit_position_attaches_identically(vit_checkpoint, position):
    before = _state_dict_snapshot(vit_checkpoint)

    old_count = _plug_and_count(old_positions, vit_checkpoint, "vit", position)
    new_count = _plug_and_count(new_positions, vit_checkpoint, "vit", position)

    assert new_count == old_count
    _assert_state_dict_identical(before, _state_dict_snapshot(vit_checkpoint))


@pytest.mark.parametrize("position", sorted(old_positions.SWIN_POSITIONS))
def test_swin_position_attaches_identically(swin_checkpoint, position):
    before = _state_dict_snapshot(swin_checkpoint)

    old_count = _plug_and_count(old_positions, swin_checkpoint, "swin", position)
    new_count = _plug_and_count(new_positions, swin_checkpoint, "swin", position)

    assert new_count == old_count
    _assert_state_dict_identical(before, _state_dict_snapshot(swin_checkpoint))


def test_registries_match_the_originals():
    assert new_positions.VIT_POSITIONS.keys() == old_positions.VIT_POSITIONS.keys()
    assert new_positions.SWIN_POSITIONS.keys() == old_positions.SWIN_POSITIONS.keys()
    for name, spec in old_positions.VIT_POSITIONS.items():
        ported = new_positions.VIT_POSITIONS[name]
        assert (ported.submodule_name, ported.hook_type, ported.scope) == (
            spec.submodule_name,
            spec.hook_type,
            spec.scope,
        )
    for name, spec in old_positions.SWIN_POSITIONS.items():
        ported = new_positions.SWIN_POSITIONS[name]
        assert (ported.submodule_name, ported.hook_type, ported.scope) == (
            spec.submodule_name,
            spec.hook_type,
            spec.scope,
        )
    assert new_positions.DROPOUT_CONFIGS == old_positions.DROPOUT_CONFIGS
    assert new_positions.SINGLE_POSITION_NAMES == old_positions.SINGLE_POSITION_NAMES
    assert (
        new_positions.STRUCTURED_POSITION_NAMES
        == old_positions.STRUCTURED_POSITION_NAMES
    )
    assert new_positions.PORTED_POSITION_NAMES == old_positions.PORTED_POSITION_NAMES


def _sample_input(name: str, layout: str) -> torch.Tensor:
    if name in HEAD_LAYOUT_OPERATORS:
        return torch.randn(HEAD_SHAPE)
    shape = TOKEN_SHAPE if layout == "token" else SPATIAL_SHAPE
    return torch.randn(shape)


@pytest.mark.parametrize("name", sorted(new_operators.PERTURBATIONS))
def test_operator_is_identity_at_rate_zero(name):
    operator = new_operators.build_perturbation(name)(0.0)
    operator.train()
    x = _sample_input(name, "token")

    assert torch.equal(operator(x), x)


def test_scale_up_is_identity_at_rate_zero():
    operator = new_operators.scale_up(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))(0.0)
    operator.train()
    x = torch.randn(2, 3, 8, 8)

    assert torch.equal(operator(x), x)


@pytest.mark.parametrize("name", TOKEN_LAYOUT_OPERATORS)
@pytest.mark.parametrize("layout", ("token", "spatial"))
def test_operator_preserves_shape(name, layout):
    operator = new_operators.build_perturbation(name)(0.3)
    operator.train()
    x = _sample_input(name, layout)

    assert operator(x).shape == x.shape


@pytest.mark.parametrize("name", HEAD_LAYOUT_OPERATORS)
def test_head_operator_preserves_shape(name):
    operator = new_operators.build_perturbation(name)(0.3)
    operator.train()
    x = torch.randn(HEAD_SHAPE)

    assert operator(x).shape == x.shape


# Operators the rewrite ADDED, which therefore have no original to match. Listed
# explicitly rather than derived, so adding one silently drops it out of the
# equivalence sweep without anyone noticing.
OPERATORS_WITHOUT_AN_ORIGINAL = frozenset({"rademacher"})

PORTED_OPERATORS = sorted(
    set(new_operators.PERTURBATIONS) - OPERATORS_WITHOUT_AN_ORIGINAL
)


def test_the_added_operators_are_exactly_the_ones_declared():
    """A new operator must be declared here, not discovered by a failing sweep."""
    added = set(new_operators.PERTURBATIONS) - set(old_operators.PERTURBATIONS)

    assert added == OPERATORS_WITHOUT_AN_ORIGINAL, (
        f"the new tree adds {sorted(added)}; update OPERATORS_WITHOUT_AN_ORIGINAL "
        "so the equivalence sweep stays honest about what it does not cover"
    )
    assert set(old_operators.PERTURBATIONS) <= set(new_operators.PERTURBATIONS), (
        "the rewrite must not drop an operator the old tree had"
    )


@pytest.mark.parametrize("name", PORTED_OPERATORS)
@pytest.mark.parametrize("layout", ("token", "spatial"))
def test_operator_output_matches_the_original(name, layout):
    if name in HEAD_LAYOUT_OPERATORS and layout == "spatial":
        pytest.skip("head masking is only defined on the per-head layout")

    x = _sample_input(name, layout)
    old_operator = old_operators.build_perturbation(name)(0.3)
    new_operator = new_operators.build_perturbation(name)(0.3)
    old_operator.train()
    new_operator.train()

    # Same seed before each call, so both draw the same Bernoulli or Gaussian
    # sample and any difference left is a real change in the arithmetic.
    torch.manual_seed(0)
    old_out = old_operator(x)
    torch.manual_seed(0)
    new_out = new_operator(x)

    assert torch.equal(old_out, new_out)


def test_gaussian_noise_std_is_per_sample():
    """The confound fix: one sample's noise level must not depend on its batch."""
    operator = new_operators.GaussianNoise(0.1)
    operator.train()

    quiet = torch.full((1, 7, 16), 0.01)
    loud = torch.full((1, 7, 16), 100.0)

    torch.manual_seed(0)
    alone = operator(quiet)
    torch.manual_seed(0)
    batched = operator(torch.cat([quiet, loud], dim=0))[:1]

    assert torch.equal(alone, batched)


def test_deterministic_perturbations_need_one_pass():
    assert new_operators.DETERMINISTIC_PERTURBATIONS == frozenset(
        {"gain_scale", "scale_up"}
    )
    assert new_operators.effective_forward_passes("gain_scale", 20) == 1
    assert new_operators.effective_forward_passes("scale_up", 20) == 1
    assert new_operators.effective_forward_passes("dropout", 20) == 20


CACHE_RATES = (0.005, 0.01, 0.02, 0.05, 0.09, 0.1, 0.15, 0.25, 0.5, 0.9, 1.0)


@pytest.mark.parametrize("rate", CACHE_RATES)
@pytest.mark.parametrize("split", ("validation", "clean", "backdoor"))
def test_dropout_pass_path_is_byte_identical(rate, split):
    old_path = old_cache.dropout_pass_path("results/x/psbd", "before_mlp", rate, split)
    new_path = new_cache.dropout_pass_path("results/x/psbd", "before_mlp", rate, split)

    assert new_path == old_path


def test_rate_tags_stay_distinct_below_one_tenth():
    tags = {new_cache._rate_tag(rate / 100) for rate in range(1, 10)}

    assert len(tags) == 9


@pytest.mark.parametrize("split", ("validation", "clean", "backdoor"))
def test_baseline_and_manifest_paths_are_byte_identical(split):
    assert new_cache.baseline_path("results/x/psbd", split) == old_cache.baseline_path(
        "results/x/psbd", split
    )
    assert new_cache.manifest_path("results/x/psbd") == old_cache.manifest_path(
        "results/x/psbd"
    )


def _find_cached_sweep() -> tuple[str, str, float] | None:
    """The first (psbd_dir, position_config, rate) with all 3 splits on disk."""
    for psbd_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*", "psbd"))):
        if not os.path.exists(os.path.join(psbd_dir, "baseline_validation.pt")):
            continue
        for config in sorted(os.listdir(psbd_dir)):
            if not os.path.isdir(os.path.join(psbd_dir, config)):
                continue
            rates = old_metrics.complete_rates(psbd_dir, config)
            if rates:
                return psbd_dir, config, rates[0]
    return None


@pytest.fixture(scope="module")
def cached_sweep() -> dict:
    located = _find_cached_sweep()
    if located is None:
        pytest.skip("no complete cached PSBD sweep on disk")

    psbd_dir, config, rate = located
    split_data = {}
    for split in ("validation", "clean", "backdoor"):
        probs, labels, loader_labels = old_cache.load_baseline(
            old_cache.baseline_path(psbd_dir, split)
        )
        per_pass, argmax = old_cache.load_dropout_pass_probs(
            old_cache.dropout_pass_path(psbd_dir, config, rate, split)
        )
        split_data[split] = (probs, labels, loader_labels, per_pass, argmax)
    return {"dir": psbd_dir, "config": config, "rate": rate, "splits": split_data}


@pytest.mark.parametrize("split", ("validation", "clean", "backdoor"))
def test_psu_matches_the_original_on_real_tensors(cached_sweep, split):
    probs, labels, _, per_pass, _ = cached_sweep["splits"][split]

    old_psu = old_metrics.psu_from_cache(probs, labels, per_pass)
    new_psu = new_scores.psu_from_cache(probs, labels, per_pass)

    assert torch.allclose(new_psu, old_psu, atol=1e-6)
    assert torch.equal(new_psu, old_psu)


@pytest.mark.parametrize("split", ("validation", "clean", "backdoor"))
def test_psu_ratio_matches_the_original_on_real_tensors(cached_sweep, split):
    probs, labels, _, per_pass, _ = cached_sweep["splits"][split]

    old_ratio = old_metrics.psu_ratio_from_cache(probs, labels, per_pass)
    new_ratio = new_scores.psu_ratio_from_cache(probs, labels, per_pass)

    assert torch.allclose(new_ratio, old_ratio, atol=1e-6)
    assert torch.equal(new_ratio, old_ratio)


@pytest.mark.parametrize("split", ("validation", "clean", "backdoor"))
def test_shift_ratio_matches_the_original_on_real_tensors(cached_sweep, split):
    _, labels, _, _, argmax = cached_sweep["splits"][split]

    assert new_scores.shift_ratio(labels, argmax) == old_metrics.shift_ratio(
        labels, argmax
    )


def test_detection_report_matches_the_original_on_real_tensors(cached_sweep):
    psu = {}
    for split, (probs, labels, _, per_pass, _) in cached_sweep["splits"].items():
        psu[split] = new_scores.psu_from_cache(probs, labels, per_pass)

    for quantile in new_decision.PSBD_QUANTILES:
        old_report = old_metrics.detection_report(
            psu["validation"], psu["clean"], psu["backdoor"], quantile
        )
        new_report = new_decision.detection_report(
            psu["validation"], psu["clean"], psu["backdoor"], quantile
        )
        assert new_report == old_report


def test_detection_report_matches_the_original_on_synthetic_scores():
    torch.manual_seed(0)
    validation = torch.randn(2000)
    clean = torch.randn(1500)
    backdoor = torch.randn(1500) - 1.0

    for quantile in new_decision.PSBD_QUANTILES:
        assert new_decision.detection_report(
            validation, clean, backdoor, quantile
        ) == old_metrics.detection_report(validation, clean, backdoor, quantile)


def test_attack_success_mask_and_pairing_match_the_original(cached_sweep):
    _, labels, loader_labels, _, _ = cached_sweep["splits"]["backdoor"]
    old_mask = old_metrics.attack_success_mask(labels, loader_labels)
    new_mask = new_decision.attack_success_mask(labels, loader_labels)

    if old_mask is None:
        assert new_mask is None
    else:
        assert torch.equal(new_mask, old_mask)

    manifest = old_cache.read_split_manifest(cached_sweep["dir"])
    clean_probs, clean_labels, _, clean_per_pass, _ = cached_sweep["splits"]["clean"]
    clean_psu = new_scores.psu_from_cache(clean_probs, clean_labels, clean_per_pass)

    assert torch.equal(
        new_decision.pair_clean_to_backdoor(clean_psu, manifest),
        old_metrics.pair_clean_to_backdoor(clean_psu, manifest),
    )


def test_complete_rates_matches_the_original(cached_sweep):
    assert new_decision.complete_rates(
        cached_sweep["dir"], cached_sweep["config"]
    ) == old_metrics.complete_rates(cached_sweep["dir"], cached_sweep["config"])


def test_critical_rate_from_disk_matches_the_original(cached_sweep):
    old_p_star = old_metrics.load_critical_rate_from_disk(
        cached_sweep["dir"], cached_sweep["config"], "clean"
    )
    new_p_star = new_decision.load_critical_rate_from_disk(
        cached_sweep["dir"], cached_sweep["config"], "clean"
    )

    if old_p_star is None:
        assert new_p_star is None
    else:
        assert torch.equal(new_p_star, old_p_star)


def _uniform_probes(count: int, size: int, shift: float, seed: int) -> list:
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(size, generator=generator) - shift for _ in range(count)]


@pytest.mark.parametrize("k", (2, 3, 5))
@pytest.mark.parametrize("target_fpr", (0.05, 0.25))
def test_multi_probe_calibrated_rule_hits_the_target_fpr(k, target_fpr):
    """Uniform, independent probes: the calibrated rule must land on target."""
    validation = _uniform_probes(k, 8000, 0.0, seed=1)
    clean = _uniform_probes(k, 8000, 0.0, seed=2)
    backdoor = _uniform_probes(k, 8000, 0.3, seed=3)

    report = new_decision.multi_probe_detection(
        validation, clean, backdoor, target_fpr=target_fpr
    )

    assert report["fpr"] == report["by_rule"]["calibrated"]["fpr"]
    assert abs(report["by_rule"]["calibrated"]["fpr"] - target_fpr) < 0.02
    assert report["by_rule"]["bonferroni"]["fpr"] <= target_fpr


@pytest.mark.parametrize("rule", ("calibrated", "bonferroni"))
def test_multi_probe_detection_matches_the_original(rule):
    validation = _uniform_probes(3, 4000, 0.0, seed=11)
    clean = _uniform_probes(3, 4000, 0.0, seed=12)
    backdoor = _uniform_probes(3, 4000, 0.3, seed=13)

    old_report = old_metrics.multi_probe_detection(
        validation, clean, backdoor, 0.25, rule
    )
    new_report = new_decision.multi_probe_detection(
        validation, clean, backdoor, 0.25, rule
    )

    # Fields the rewrite ADDED. "reduction" records min against median, which the
    # old tree could not express because it only had min. AUROC is threshold free
    # so it is now repeated inside each rule block, where the old tree left None
    # and a caller reading a block found nothing. Listed explicitly so a future
    # addition has to be declared rather than quietly widening the exemption.
    ADDED_TOP_LEVEL = {"reduction"}
    ADDED_PER_RULE = {"auroc"}

    assert set(new_report) - set(old_report) == ADDED_TOP_LEVEL
    assert new_report["reduction"] == "min", "the ported default must stay min"

    for key in set(old_report) - {"by_rule"}:
        assert new_report[key] == old_report[key], f"{key} drifted from the original"

    for rule_name, old_block in old_report["by_rule"].items():
        new_block = new_report["by_rule"][rule_name]
        assert set(new_block) - set(old_block) == ADDED_PER_RULE
        for key in old_block:
            if old_block[key] is None:
                continue  # the old tree left auroc unset inside a rule block
            assert new_block[key] == old_block[key], f"{rule_name}.{key} drifted"
        assert new_block["auroc"] == new_report["auroc"]


def test_multi_probe_rejects_an_unknown_rule():
    probes = _uniform_probes(2, 100, 0.0, seed=21)
    with pytest.raises(ValueError):
        new_decision.multi_probe_detection(probes, probes, probes, 0.25, "wishful")


def test_rank_and_combined_score_match_the_original():
    torch.manual_seed(0)
    reference = torch.randn(500)
    values = torch.randn(300)

    assert torch.equal(
        new_scores.to_rank(values, reference), old_metrics.to_rank(values, reference)
    )

    probes = [torch.randn(300) for _ in range(3)]
    references = [torch.randn(500) for _ in range(3)]
    assert torch.equal(
        new_scores.multi_probe_score(probes, references),
        old_metrics.multi_probe_score(probes, references),
    )


def test_critical_rate_matches_the_original():
    torch.manual_seed(0)
    baseline_labels = torch.randint(0, 10, (200,))
    rates = [0.1, 0.2, 0.3]
    argmax_by_rate = {
        rate: torch.randint(0, 10, (4, 200), dtype=torch.int16) for rate in rates
    }

    assert torch.equal(
        new_scores.critical_rate(baseline_labels, rates, argmax_by_rate),
        old_metrics.critical_rate(baseline_labels, rates, argmax_by_rate),
    )


def test_shift_target_histogram_matches_the_original():
    torch.manual_seed(0)
    baseline_labels = torch.randint(0, 10, (200,))
    argmax = torch.randint(0, 10, (4, 200), dtype=torch.int16)

    assert new_scores.shift_target_histogram(
        baseline_labels, argmax, 10
    ) == old_metrics.shift_target_histogram(baseline_labels, argmax, 10)


def test_rate_selection_matches_the_original():
    shift_by_rate = {0.1: 0.2, 0.2: 0.55, 0.3: None, 0.4: 0.85}
    auroc_by_rate = {0.1: 0.6, 0.2: float("nan"), 0.3: 0.71}

    assert new_decision.select_rate_adaptively(
        shift_by_rate
    ) == old_metrics.select_rate_adaptively(shift_by_rate)
    for target in new_decision.SHIFT_MATCH_TARGETS:
        assert new_decision.select_rate_at_matched_shift(
            shift_by_rate, target
        ) == old_metrics.select_rate_at_matched_shift(shift_by_rate, target)
    assert new_decision.select_rate_by_oracle(
        auroc_by_rate
    ) == old_metrics.select_rate_by_oracle(auroc_by_rate)
    assert new_decision.select_rate_by_oracle({}) is None


def test_thresholds_and_constants_match_the_original():
    assert new_decision.PSBD_QUANTILES == old_metrics.PSBD_QUANTILES
    assert new_decision.HEADLINE_QUANTILE == old_metrics.HEADLINE_QUANTILE
    assert new_decision.SHIFT_MATCH_TARGETS == old_metrics.SHIFT_MATCH_TARGETS

    torch.manual_seed(0)
    validation = torch.randn(1000)
    for quantile in new_decision.PSBD_QUANTILES:
        assert new_decision.threshold_at_quantile(
            validation, quantile
        ) == old_metrics.threshold_at_quantile(validation, quantile)


def test_probe_perturbs_the_forward_pass_at_both_wrapper_positions(vit_checkpoint):
    """The 2 positions realized by a forward wrapper must actually take effect."""
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        clean_logits = vit_checkpoint(images)

    for position in ("after_attention_residual", "attention_heads"):
        handles = new_positions.plug_dropout(
            vit_checkpoint, "vit", (position,), {}, rate=0.5
        )
        with torch.inference_mode():
            perturbed_logits = vit_checkpoint(images)
        new_positions.unplug_dropout(handles)

        assert not torch.equal(perturbed_logits, clean_logits), position

        with torch.inference_mode():
            restored_logits = vit_checkpoint(images)
        assert torch.equal(restored_logits, clean_logits), position


def test_block_range_restricts_a_block_scope_position(vit_checkpoint):
    handles = new_positions.plug_dropout(
        vit_checkpoint, "vit", ("before_mlp",), {}, rate=0.3, block_range=(3, 6)
    )
    count = len(handles)
    new_positions.unplug_dropout(handles)

    assert count == 4

    with pytest.raises(ValueError):
        new_positions.plug_dropout(
            vit_checkpoint,
            "vit",
            ("after_embedding",),
            {},
            rate=0.3,
            block_range=(1, 2),
        )
