"""Checkpoint save/load round-trips and metadata reading.

Covers the parts of the checkpoint pipeline that are easy to get subtly wrong
and hard to notice: a checkpoint that loads but with the wrong weights, an
args.json reader that silently accepts a malformed file, or a folder-name
parser that mis-tags one of the two known naming irregularities.
"""

import json
import os

import pytest
import torch

from checkpoint_eval import read_checkpoint_metadata
from models import build_vit, load_vit_checkpoint
from normalize_checkpoints import parse_folder_name
from train import save_checkpoint


def test_save_and_load_checkpoint_round_trip(tmp_path):
    model = build_vit(num_classes=4)
    path = os.path.join(tmp_path, "attack_result.pt")
    save_checkpoint(model, num_classes=4, path=path)

    reloaded = load_vit_checkpoint(path, device=torch.device("cpu"))
    for original, restored in zip(model.state_dict().values(), reloaded.state_dict().values()):
        assert torch.equal(original, restored), "reloaded weights must match exactly what was saved"


def test_save_checkpoint_writes_args_json_sidecar(tmp_path):
    model = build_vit(num_classes=4)
    path = os.path.join(tmp_path, "attack_result.pt")
    save_checkpoint(model, num_classes=4, path=path, metadata={"dataset": "cifar10", "attack": "benign"})

    args_path = os.path.join(tmp_path, "args.json")
    assert os.path.exists(args_path), "metadata must be written as an args.json sidecar, not merged into the .pt"
    with open(args_path) as handle:
        assert json.load(handle) == {"dataset": "cifar10", "attack": "benign"}

    # The .pt itself only ever carries the weights and the class count.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert set(checkpoint) == {"model", "num_classes"}


def test_read_checkpoint_metadata_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_checkpoint_metadata(os.path.join(tmp_path, "attack_result.pt"))


def test_read_checkpoint_metadata_missing_required_key(tmp_path):
    with open(os.path.join(tmp_path, "args.json"), "w") as handle:
        json.dump({"dataset": "cifar10"}, handle)  # missing "attack" and "target_label"

    with pytest.raises(KeyError):
        read_checkpoint_metadata(os.path.join(tmp_path, "attack_result.pt"))


def test_read_checkpoint_metadata_round_trip(tmp_path):
    metadata = {"dataset": "cifar10", "attack": "badnet_a2o", "target_label": 0, "poison_rate": 0.1}
    with open(os.path.join(tmp_path, "args.json"), "w") as handle:
        json.dump(metadata, handle)

    assert read_checkpoint_metadata(os.path.join(tmp_path, "attack_result.pt")) == metadata


# folder_name -> (architecture, dataset, attack_or_benign, poison_rate_tag, optimizer, rho)
FOLDER_NAME_CASES = [
    ("vit_cifar100_badnet_a2o_0_01", ("vit", "cifar100", "badnet_a2o", "0_01", "adam", None)),
    ("swin_cifar100_badnet_a2o_0_01_sam_rho_0_05", ("swin", "cifar100", "badnet_a2o", "0_01", "sam", 0.05)),
    # The two known irregular formats normalize_checkpoints.py had to unify:
    # underscore-separated rho for backdoor sweeps, no underscore for benign.
    ("vit_cifar100_badnet_a2o_0_01_sam_rho_0_15", ("vit", "cifar100", "badnet_a2o", "0_01", "sam", 0.15)),
    ("vit_cifar100_benign_sam_rho0_05", ("vit", "cifar100", "benign", None, "sam", 0.05)),
    # Architecture-token position differs: prefix for attacks, mid-name for benign.
    ("swin_cifar10_benign", ("swin", "cifar10", "benign", None, "adam", None)),
    ("cifar10_swin_benign_sam_rho0_1", ("swin", "cifar10", "benign", None, "sam", 0.1)),
]


@pytest.mark.parametrize("folder_name,expected", FOLDER_NAME_CASES)
def test_folder_name_parser_handles_known_formats(folder_name, expected):
    parsed = parse_folder_name(folder_name, checkpoints_dir="unused")
    architecture, dataset, attack_or_benign, poison_rate_tag, optimizer, rho = expected
    assert parsed.architecture == architecture
    assert parsed.dataset == dataset
    assert parsed.attack_or_benign == attack_or_benign
    assert parsed.poison_rate_tag == poison_rate_tag
    assert parsed.optimizer == optimizer
    assert parsed.rho == rho


def test_canonical_name_omits_adam_tag_and_underscores_sam_rho():
    adam = parse_folder_name("vit_cifar100_badnet_a2o_0_01", checkpoints_dir="unused")
    assert adam.canonical_name() == "vit_cifar100_badnet_a2o_0_01"

    sam = parse_folder_name("vit_cifar100_benign_sam_rho0_15", checkpoints_dir="unused")
    assert sam.canonical_name() == "vit_cifar100_benign_sam_rho_0_15"
