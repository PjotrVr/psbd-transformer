"""Wall-clock cost of PSBD against a plain forward pass, at k = 1, 3, 5, 10, 20.

The deployed placement runs PSBD at k = 3 forward passes per sample, the PSBD
paper's own value (sec/4_method.tex). This measures how much slower that is
against a plain unperturbed forward pass and against higher k, on the hardware
this project runs on: the login-node A100.

2 placements are timed at every k. token_mask at before_attention_norm is the
placement this project recommends. dropout at post_residual is the placement
the PSBD paper itself uses. A forward pass's cost is set almost entirely by k,
not by which position or operator is perturbed, since every operator here does
O(1) elementwise or per-token work next to a whole transformer block, so the 2
placements should read almost identically. Timing both is the check on that
claim rather than a second headline number.

512 clean CIFAR-100 test images come from the standardized PSBD validation
split (data.splits.build_psbd_loaders_from_checkpoint), moved to the GPU once
and reused unperturbed for every timed config, so no data-loading time enters
the measurement. 1 warm-up iteration is discarded, then 3 repeats are timed
with torch.cuda.synchronize() bracketing each one, batch size 128, fp32, no
autocast, torch.no_grad().

Example
    PYTHONPATH=. .venv/bin/python experiments/psbd_cost/measure.py
"""

import argparse
import json
import os
import time

import torch
from lightning import seed_everything

from data.splits import build_psbd_loaders_from_checkpoint
from defences.operators import build_operator, check_operator_position
from experiments._paths import experiment_result_path
from models.backbones import MODEL_INPUT_SIZE, load_checkpoint
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout

VIT_CHECKPOINT = "vit_cifar100_badnet_a2o_0_1"
SWIN_CHECKPOINT = "swin_cifar100_badnet_a2o_0_1"

NUM_IMAGES = 512
BATCH_SIZE = 128
PERTURBATION_RATE = 0.3
FORWARD_PASS_COUNTS = (1, 3, 5, 10, 20)
NUM_WARMUP = 1
NUM_REPEATS = 3
# Reseeds the mask draw before every config, so a rerun samples the identical
# masks. Timing does not depend on which mask was drawn, only on how many.
TIMING_SEED = 0

# (operator, position). before_attention_norm is a single atomic position, the
# site of RECOMMENDED_PLACEMENT (defences.decision). post_residual is a
# DROPOUT_CONFIGS key expanding to 2 positions, PUBLISHED_PLACEMENT, the PSBD
# paper's own ConvNet site.
PLACEMENTS = (
    ("token_mask", "before_attention_norm"),
    ("dropout", "post_residual"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--vit-checkpoint", default=VIT_CHECKPOINT)
    parser.add_argument("--swin-checkpoint", default=SWIN_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "torch_version": torch.__version__,
        "batch_size": BATCH_SIZE,
        "num_images": NUM_IMAGES,
        "model_resize_size": MODEL_INPUT_SIZE,
        "perturbation_rate": PERTURBATION_RATE,
        "forward_pass_counts": list(FORWARD_PASS_COUNTS),
        "num_warmup": NUM_WARMUP,
        "num_repeats": NUM_REPEATS,
        "architectures": {},
    }

    architecture_checkpoints = [("vit", args.vit_checkpoint)]
    swin_args_path = os.path.join(
        args.checkpoints_dir, args.swin_checkpoint, "args.json"
    )
    if os.path.exists(swin_args_path):
        architecture_checkpoints.append(("swin", args.swin_checkpoint))
    else:
        print(f"[skip] no args.json at {swin_args_path}, measuring ViT-B/16 only")

    for architecture, checkpoint_folder in architecture_checkpoints:
        print(f"{architecture} {checkpoint_folder}", flush=True)
        model, batches, manifest = load_timing_batches(
            checkpoint_folder,
            architecture,
            args.checkpoints_dir,
            args.raw_data_dir,
            device,
        )
        result["architectures"][architecture] = {
            "checkpoint": checkpoint_folder,
            "dataset": manifest["dataset"],
            "native_image_size": batches[0].shape[-1],
            **time_architecture(model, architecture, batches, device),
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    path = experiment_result_path("psbd_cost", "cost.json", args.results_dir)
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"written to {path}")


def load_timing_batches(
    checkpoint_folder: str,
    architecture: str,
    checkpoints_dir: str,
    raw_data_dir: str,
    device: torch.device,
) -> tuple[torch.nn.Module, list[torch.Tensor], dict]:
    """The loaded model and NUM_IMAGES clean validation images, chunked and on device.

    Built once per architecture and reused, unperturbed, for every timed config,
    so every measurement below pays 0 data-loading cost and differs only in what
    runs on the GPU. Images stay at the dataset's native resolution (32x32 for
    CIFAR-100): the model's own Sequential(Resize(MODEL_INPUT_SIZE), network)
    wrapper does the upscale internally, exactly as every other caller of
    load_checkpoint expects.
    """
    checkpoint_path = os.path.join(
        checkpoints_dir, checkpoint_folder, "attack_result.pt"
    )
    model = load_checkpoint(architecture, checkpoint_path, device)

    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        raw_data_dir=raw_data_dir,
        batch_size=BATCH_SIZE,
        max_samples=NUM_IMAGES,
    )
    images = torch.cat(
        [batch_images for batch_images, _ in loaders["validation"]], dim=0
    )  # (NUM_IMAGES, channels, native_size, native_size)
    assert images.shape[0] == NUM_IMAGES, (
        f"expected {NUM_IMAGES} validation images, got {images.shape[0]}"
    )

    images = images.to(device)  # (NUM_IMAGES, channels, native_size, native_size)
    batches = list(images.split(BATCH_SIZE, dim=0))  # NUM_IMAGES // BATCH_SIZE tensors
    return model, batches, manifest


def time_architecture(
    model: torch.nn.Module,
    architecture: str,
    batches: list[torch.Tensor],
    device: torch.device,
) -> dict:
    """The plain pass and every (placement, k) config, for 1 already-loaded model."""
    print("  plain forward pass", flush=True)
    plain = time_config(model, batches, forward_passes=1, device=device)

    placements = {}
    for operator, position in PLACEMENTS:
        position_names = DROPOUT_CONFIGS.get(position, (position,))
        for name in position_names:
            check_operator_position(operator, name)
        operator_factory = build_operator(operator)
        factory = {name: operator_factory for name in position_names}

        by_k = {}
        for forward_passes in FORWARD_PASS_COUNTS:
            print(f"  {operator} at {position}, k={forward_passes}", flush=True)
            seed_everything(TIMING_SEED)
            handles = plug_dropout(
                model, architecture, position_names, factory, PERTURBATION_RATE
            )
            try:
                by_k[str(forward_passes)] = time_config(
                    model, batches, forward_passes, device
                )
            finally:
                unplug_dropout(handles)

        placements[f"{operator}_{position}"] = {
            "operator": operator,
            "position": position,
            "rate": PERTURBATION_RATE,
            "by_k": by_k,
        }

    return {"plain": plain, "placements": placements}


def time_config(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    forward_passes: int,
    device: torch.device,
) -> dict:
    """Wall-clock seconds for 1 warm-up iteration (discarded) plus NUM_REPEATS timed ones.

    Each iteration runs `forward_passes` forward calls over every batch, so its
    elapsed time divided by NUM_IMAGES is the per-input cost of running PSBD at
    that pass count over this set of images.
    """
    for _ in range(NUM_WARMUP):
        run_forward_passes(model, batches, forward_passes)
    if device.type == "cuda":
        torch.cuda.synchronize()

    repeat_seconds = []
    for _ in range(NUM_REPEATS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run_forward_passes(model, batches, forward_passes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        repeat_seconds.append(time.perf_counter() - start)

    mean_seconds = sum(repeat_seconds) / len(repeat_seconds)
    timing = {
        "forward_passes": forward_passes,
        "repeat_seconds": repeat_seconds,
        "mean_seconds": mean_seconds,
        "seconds_per_input": mean_seconds / NUM_IMAGES,
        "images_per_second": NUM_IMAGES / mean_seconds,
    }
    return timing


def run_forward_passes(
    model: torch.nn.Module, batches: list[torch.Tensor], forward_passes: int
) -> None:
    """forward_passes forward calls over every batch, fp32, no_grad, output discarded."""
    with torch.no_grad():
        for batch in batches:
            for _ in range(forward_passes):
                model(batch)  # (batch, num_classes), discarded


if __name__ == "__main__":
    main()
