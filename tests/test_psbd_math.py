"""The PSBD math the structural tests do not touch: the per-pass probability
gather, the cache round-trips and idempotency, the rate tag, and the all-to-one
eligibility exclusion in the split manifest.

The core gather is checked against a controlled identity model whose logits are
the input itself, so every expected probability is hand-computable with a plain
softmax and no GPU. That isolates the arithmetic of compute_dropout_pass_probs
(which class it tracks, how it pairs batches to labels) from the transformer and
the hook machinery, which tests/test_dropout.py covers separately.
"""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, TensorDataset

from data.splits import build_psbd_loaders_from_checkpoint
from defences.inference import compute_dropout_pass_probs
from defences.cache import (
    _rate_tag,
    baseline_path,
    load_baseline,
    load_dropout_pass_probs,
    load_or_build_baseline,
    read_split_manifest,
    save_baseline,
    save_dropout_pass_probs,
    write_split_manifest,
)
from data.loading import extract_labels, load_clean_datasets

CPU = torch.device("cpu")
CHECKPOINT = "checkpoints/vit_cifar100_wanet_0_1/attack_result.pt"


class LogitPassthrough(nn.Module):
    """Returns its input unchanged, so the input tensor is the logits directly.

    forward_probs then softmaxes it, making every tracked probability something
    a test can compute by hand with no model weights involved.
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


def _logit_loader(logits: torch.Tensor, batch_size: int) -> DataLoader:
    # The second tensor is the loader's own label, which compute_dropout_pass_probs
    # ignores in favor of the baseline_labels argument, so a zero placeholder is fine.
    placeholder = torch.zeros(len(logits), dtype=torch.long)
    return DataLoader(
        TensorDataset(logits, placeholder), batch_size=batch_size, shuffle=False
    )


def _distinct_peak_logits(n: int, num_classes: int) -> torch.Tensor:
    """Row i peaks at a different class with a different height.

    Distinct argmax classes and distinct peak probabilities per row mean a
    batch-to-label misalignment gathers a visibly wrong number, so an off-by-one
    offset cannot pass silently.
    """
    logits = torch.zeros(n, num_classes)
    for row in range(n):
        logits[row, row % num_classes] = 1.0 + row
    return logits


def test_tracks_baseline_argmax_probability_across_uneven_batches():
    # 10 samples, batch 4, so batches of 4, 4, 2: the running offset must stay
    # aligned across the ragged final batch.
    logits = _distinct_peak_logits(10, num_classes=5)
    baseline_labels = logits.argmax(dim=1)
    loader = _logit_loader(logits, batch_size=4)

    per_pass, per_pass_argmax = compute_dropout_pass_probs(
        LogitPassthrough(), loader, baseline_labels, CPU, 3, use_bfloat16=False, seed=0
    )

    expected = (
        F.softmax(logits, dim=1).gather(1, baseline_labels.view(-1, 1)).squeeze(1)
    )
    assert per_pass.shape == (3, 10)
    assert per_pass_argmax.shape == (3, 10)
    assert per_pass_argmax.dtype == torch.int16
    # With no dropout plugged the identity model reproduces the baseline argmax
    # every pass, so sigma computed from this must be exactly 0.
    for pass_index in range(3):
        assert torch.equal(per_pass_argmax[pass_index].long(), baseline_labels)
    # No dropout is plugged, so the identity model is deterministic: every pass is
    # the baseline, and every row must equal the max softmax probability per sample.
    for pass_index in range(3):
        assert torch.allclose(per_pass[pass_index], expected, atol=1e-6)


def test_gathers_the_given_label_not_the_per_pass_argmax():
    # Hand the function labels that are deliberately NOT the argmax. It must track
    # exactly those classes, proving PSU follows the no-dropout prediction it is
    # given, never a fresh per-pass argmax.
    logits = _distinct_peak_logits(6, num_classes=5)
    wrong_labels = (logits.argmax(dim=1) + 1) % 5
    loader = _logit_loader(logits, batch_size=4)

    per_pass, _ = compute_dropout_pass_probs(
        LogitPassthrough(), loader, wrong_labels, CPU, 1, use_bfloat16=False, seed=0
    )

    expected = F.softmax(logits, dim=1).gather(1, wrong_labels.view(-1, 1)).squeeze(1)
    argmax_prob = F.softmax(logits, dim=1).max(dim=1).values
    assert torch.allclose(per_pass[0], expected, atol=1e-6)
    assert not torch.allclose(per_pass[0], argmax_prob, atol=1e-3)


def test_no_perturbation_gives_zero_psu():
    # PSU = baseline_prob(argmax) - mean_over_passes(dropout_prob(argmax)). With no
    # dropout plugged the passes equal the baseline, so PSU must be exactly zero,
    # the anchor the whole score is measured against.
    logits = _distinct_peak_logits(8, num_classes=4)
    baseline_labels = logits.argmax(dim=1)
    baseline_probs = F.softmax(logits, dim=1)
    loader = _logit_loader(logits, batch_size=3)

    per_pass, _ = compute_dropout_pass_probs(
        LogitPassthrough(), loader, baseline_labels, CPU, 3, use_bfloat16=False, seed=0
    )

    baseline_gathered = baseline_probs.gather(1, baseline_labels.view(-1, 1)).squeeze(1)
    psu = baseline_gathered - per_pass.mean(dim=0)
    assert torch.allclose(psu, torch.zeros(8), atol=1e-6)


def test_output_is_float32_even_without_bfloat16():
    logits = _distinct_peak_logits(4, num_classes=4)
    per_pass, _ = compute_dropout_pass_probs(
        LogitPassthrough(),
        _logit_loader(logits, 4),
        logits.argmax(dim=1),
        CPU,
        2,
        use_bfloat16=False,
        seed=0,
    )
    assert per_pass.dtype == torch.float32


def test_same_seed_reproduces_masks_paired_across_calls():
    # A hooked stochastic dropout makes the passes differ. Two calls at the same
    # seed must produce identical per-pass tensors, which is what keeps the
    # validation, clean, and backdoor masks paired in the real sweep.
    logits = _distinct_peak_logits(12, num_classes=5)
    baseline_labels = logits.argmax(dim=1)

    model = LogitPassthrough()
    stochastic = nn.Dropout(0.5)
    stochastic.train()
    handle = model.register_forward_hook(
        lambda module, args, output: stochastic(output)
    )

    first, _ = compute_dropout_pass_probs(
        model, _logit_loader(logits, 4), baseline_labels, CPU, 3, False, seed=7
    )
    second, _ = compute_dropout_pass_probs(
        model, _logit_loader(logits, 4), baseline_labels, CPU, 3, False, seed=7
    )
    other, _ = compute_dropout_pass_probs(
        model, _logit_loader(logits, 4), baseline_labels, CPU, 3, False, seed=8
    )
    handle.remove()

    assert torch.equal(first, second), "same seed must reproduce the mask sequence"
    assert not torch.equal(first, other), "a different seed should perturb differently"


def test_empty_loader_returns_empty_shaped_tensor():
    logits = torch.zeros(0, 4)
    per_pass, per_pass_argmax = compute_dropout_pass_probs(
        LogitPassthrough(),
        _logit_loader(logits, 4),
        torch.zeros(0, dtype=torch.long),
        CPU,
        3,
        use_bfloat16=False,
        seed=0,
    )
    assert per_pass.shape == (3, 0)
    assert per_pass_argmax.shape == (3, 0)


def test_baseline_round_trip(tmp_path):
    probs = torch.rand(20, 100)
    labels = probs.argmax(dim=1)
    loader_labels = torch.randint(0, 100, (20,))
    path = os.path.join(tmp_path, "baseline_clean.pt")
    save_baseline(path, probs, labels, loader_labels)
    loaded_probs, loaded_labels, loaded_targets = load_baseline(path)
    assert torch.equal(loaded_probs, probs)
    assert torch.equal(loaded_labels, labels)
    assert torch.equal(loaded_targets, loader_labels)


def test_dropout_pass_probs_round_trip(tmp_path):
    per_pass = torch.rand(3, 50)
    per_pass_argmax = torch.randint(0, 10, (3, 50), dtype=torch.int16)
    path = os.path.join(tmp_path, "before_attention", "rate_0_5_clean.pt")
    save_dropout_pass_probs(path, per_pass, per_pass_argmax)
    loaded_probs, loaded_argmax = load_dropout_pass_probs(path)
    assert torch.equal(loaded_probs, per_pass)
    assert torch.equal(loaded_argmax, per_pass_argmax)


def test_dropout_pass_probs_load_tolerates_a_file_without_argmax(tmp_path):
    # Files written before argmax was saved must still load, so a partial sweep
    # does not have to be thrown away. The argmax comes back empty and stage 2
    # reports sigma as unavailable rather than failing the checkpoint.
    per_pass = torch.rand(3, 8)
    path = os.path.join(tmp_path, "legacy", "rate_0_5_clean.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"per_pass_probs": per_pass}, path)
    loaded_probs, loaded_argmax = load_dropout_pass_probs(path)
    assert torch.equal(loaded_probs, per_pass)
    assert loaded_argmax.numel() == 0


def test_manifest_write_identical_is_idempotent(tmp_path):
    psbd_dir = str(tmp_path)
    manifest = {"seed": 0, "n_total": 10000, "n_heldout": 2000}
    write_split_manifest(psbd_dir, manifest)
    # Re-writing the identical split is a no-op, so a checkpoint's parallel jobs
    # all agree on the one manifest without racing on the bytes.
    write_split_manifest(psbd_dir, manifest)
    assert read_split_manifest(psbd_dir) == manifest


def test_manifest_write_rejects_a_conflicting_split(tmp_path):
    psbd_dir = str(tmp_path)
    write_split_manifest(psbd_dir, {"seed": 0, "n_total": 10000, "n_heldout": 2000})
    # A truncated max_samples or different-seed run wrote a different split here.
    # It must raise, not silently keep stale indices every tensor is ordered by.
    with pytest.raises(ValueError, match="different split"):
        write_split_manifest(psbd_dir, {"seed": 0, "n_total": 10000, "n_heldout": 128})


def test_load_or_build_baseline_reuses_without_recomputing(tmp_path):
    logits = _distinct_peak_logits(16, num_classes=4)
    loader = _logit_loader(logits, batch_size=4)
    psbd_dir = str(tmp_path)

    probs, labels, _ = load_or_build_baseline(
        psbd_dir, "clean", LogitPassthrough(), loader, CPU, use_bfloat16=False
    )
    assert os.path.exists(os.path.join(psbd_dir, "baseline_clean.pt"))

    # A second call whose model would compute different values must still return
    # the first result, which is only possible if it loaded the cache rather than
    # recomputing. The loader is the same size, so the integrity check passes.
    class ZeroLogits(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    reused_probs, reused_labels, _ = load_or_build_baseline(
        psbd_dir, "clean", ZeroLogits(), loader, CPU, use_bfloat16=False
    )
    assert torch.equal(reused_probs, probs)
    assert torch.equal(reused_labels, labels)


def test_load_or_build_baseline_rejects_a_stale_truncated_cache(tmp_path):
    psbd_dir = str(tmp_path)
    # A cache a truncated run left behind: 5 rows, but the real loader serves 16.
    save_baseline(
        baseline_path(psbd_dir, "clean"),
        torch.rand(5, 4),
        torch.zeros(5, dtype=torch.long),
        torch.zeros(5, dtype=torch.long),
    )
    loader = _logit_loader(_distinct_peak_logits(16, num_classes=4), batch_size=4)
    with pytest.raises(ValueError, match="rows but the loader serves"):
        load_or_build_baseline(
            psbd_dir, "clean", LogitPassthrough(), loader, CPU, use_bfloat16=False
        )


def test_rate_tag_is_injective_over_the_swept_rates():
    from cli.sweep import DROPOUT_RATES

    tags = [_rate_tag(rate) for rate in DROPOUT_RATES]
    assert len(set(tags)) == len(tags), "each swept rate must get a distinct tag"
    assert _rate_tag(0.1) == "0_1"
    assert _rate_tag(0.9) == "0_9"


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason="cifar100 wanet checkpoint not on disk"
)
def test_all_to_one_split_drops_the_whole_target_class_from_backdoor():
    """The CLAUDE.md eligibility rule, checked at the manifest level.

    wanet is all-to-one with target 0, so the backdoor eval set must contain
    every analysis image whose true label is not 0 and none whose label is 0.
    The existing split test only checks the backdoor indices are a subset of the
    analysis pool, which a wrong eligibility filter would also satisfy.
    """
    _, manifest = build_psbd_loaders_from_checkpoint(CHECKPOINT, batch_size=64)
    assert manifest["dataset"] == "cifar100"

    # extract_labels reads cifar100's target array directly without decoding
    # images, so the transform here is only to satisfy the loader signature.
    _, test_base = load_clean_datasets(
        "cifar100", transforms_v2.Compose([transforms_v2.ToImage()]), "raw_data"
    )
    labels = extract_labels(test_base)
    target_label = 0

    analysis = manifest["analysis_clean_indices"]
    backdoor = manifest["analysis_backdoor_indices"]
    expected_backdoor = [index for index in analysis if labels[index] != target_label]

    assert backdoor == expected_backdoor, (
        "backdoor set must be all non-target analysis images, in order"
    )
    assert all(labels[index] != target_label for index in backdoor)
    dropped = set(analysis) - set(backdoor)
    assert dropped == {index for index in analysis if labels[index] == target_label}
