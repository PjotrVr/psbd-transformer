"""Checkpoint save/load round-trips and metadata reading.

Covers the parts of the checkpoint pipeline that are easy to get subtly wrong
and hard to notice: a checkpoint that loads but with the wrong weights, or an
args.json reader that silently accepts a malformed file. normalize_checkpoints.py's
folder-name parser is tested separately in scratch/test_normalize_checkpoints.py,
since that script itself lives in scratch/ now (a gitignored, one-off migration
tool, not part of the tracked package), and a tracked test importing from a
gitignored path would fail on a fresh clone.
"""

import json
import os

import pytest
import torch

from defences.checkpoint_eval import read_checkpoint_metadata
from models import build_vit, load_vit_checkpoint
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
