"""Is PSBD-TM's 0.459 AUROC on vit_cifar10_wanet_0_1 real, an artefact or checkpoint specific?

See README.md for the question and the answer. This script covers step 4 only: the
independent check against BackdoorBench's own reference WaNet checkpoint for CIFAR-10
at 10% poisoning (backdoor_bench_checkpoints/cifar10_wanet_0_1/), read through its PNG
test set rather than our own re-implemented trigger, so a bug specific to our WaNet
code cannot explain the result either way.

Mirrors cli.sweep's cache format exactly (defences.cache, models.positions,
defences.operators) so cli.analyze's reader, defences.scores and defences.decision
apply unchanged. The only thing this script does differently from cli.sweep is how
the model and the 3 loaders are built, because a BackdoorBench folder carries no
args.json and its poisoned test images are pre-rendered PNGs, not a trigger this
project's own attacks code can rebuild in memory.

Usage
    PYTHONPATH=. .venv/bin/python -m experiments.wanet_cifar10_audit.measure
"""

import json
import os

import torch
import torchvision.transforms.v2 as transforms_v2
from lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets

from data.backdoorbench import load_backdoor_splits, split_validation_and_eval
from data.registry import DATASET_REGISTRY
from defences.cache import (
    baseline_path,
    dropout_pass_path,
    load_dropout_pass_probs,
    run_provenance_path,
    save_baseline,
    save_dropout_pass_probs,
    write_split_manifest,
)
from defences.decision import (
    detection_report,
    pair_clean_to_backdoor,
    select_rate_adaptively,
    select_rate_at_matched_shift,
)
from defences.inference import build_baseline_cache, compute_dropout_pass_probs
from defences.operators import build_operator
from defences.scores import psu_from_cache, psu_ratio_from_cache, shift_ratio
from models.backbones import load_checkpoint
from models.positions import DROPOUT_CONFIGS, plug_dropout, unplug_dropout

DATASET_NAME = "cifar10"
BB_FOLDER = "cifar10_wanet_0_1"
WEIGHTS_DIR = "backdoor_bench_checkpoints"
LABEL_MODE = "all_to_one"
TARGET_LABEL = 0

RESULTS_DIR = "results"
CACHE_FOLDER = "bb_cifar10_wanet_0_1"

PSBD_SPLIT_SEED = 0
PSBD_HELDOUT_SIZE = 2000
FORWARD_PASSES = 3
BATCH_SIZE = 64
MASK_SEED = 0

# The 3 placements the audit compares: the recommended input-side token mask
# against the 2 dropout placements that read high on this checkpoint through our
# own re-implemented WaNet trigger.
PLACEMENTS = {
    "before_attention_norm_token_mask": {
        "position_names": ("before_attention_norm",),
        "operator": "token_mask",
        "rates": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    },
    "post_residual": {
        "position_names": DROPOUT_CONFIGS["post_residual"],
        "operator": "dropout",
        "rates": (
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.07,
            0.09,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),
    },
    "pre_residual": {
        "position_names": DROPOUT_CONFIGS["pre_residual"],
        "operator": "dropout",
        "rates": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    },
}


def build_transform(image_size: int, mean: tuple, std: tuple) -> transforms_v2.Compose:
    """Resize to the dataset's native trigger resolution, tensor, then normalize.

    The trigger is already baked into the BackdoorBench PNG, so unlike
    data.loading.base_image_transform there is no reason to keep the tensor
    unnormalized past this point: nothing here stamps a trigger afterward.
    """
    transform = transforms_v2.Compose(
        [
            transforms_v2.Resize((image_size, image_size)),
            transforms_v2.ToTensor(),
            transforms_v2.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def build_loaders() -> tuple[dict[str, DataLoader], dict]:
    """The (validation, clean, backdoor) loaders from BackdoorBench's own PNGs, plus a manifest.

    validation is the 2000-sample clean threshold set (PSBD's own convention,
    data.splits.PSBD_HELDOUT_SIZE), clean and backdoor are the remaining clean and
    triggered images, paired row for row: split_validation_and_eval reuses the same
    eval_idx for both, so backdoor row i and clean row i are the same original
    image with and without the trigger.
    """
    spec = DATASET_REGISTRY[DATASET_NAME]
    transform = build_transform(spec.image_size, spec.mean, spec.std)

    clean_test_dataset = tv_datasets.CIFAR10(
        root=os.path.join("raw_data", DATASET_NAME),
        train=False,
        download=False,
        transform=transform,
    )

    backdoor_test, eligible_counterparts = load_backdoor_splits(
        BB_FOLDER,
        clean_test_dataset,
        transform,
        WEIGHTS_DIR,
        LABEL_MODE,
        TARGET_LABEL,
        spec.num_classes,
    )

    clean_val, clean_eval, backdoor_eval = split_validation_and_eval(
        eligible_counterparts, backdoor_test, PSBD_HELDOUT_SIZE, PSBD_SPLIT_SEED
    )

    loaders = {
        "validation": DataLoader(clean_val, batch_size=BATCH_SIZE, shuffle=False),
        "clean": DataLoader(clean_eval, batch_size=BATCH_SIZE, shuffle=False),
        "backdoor": DataLoader(backdoor_eval, batch_size=BATCH_SIZE, shuffle=False),
    }

    # clean_eval and backdoor_eval share eval_idx (see split_validation_and_eval),
    # so row i of either loader is the same original image, with and without the
    # trigger. The pairing is already an identity mapping, unlike data.splits'
    # manifest, where the backdoor split can drop rows the clean split keeps.
    n_eval = len(clean_eval)
    manifest = {
        "seed": PSBD_SPLIT_SEED,
        "dataset": DATASET_NAME,
        "probe_attack": "wanet",
        "probe_target_label": TARGET_LABEL,
        "label_mode": LABEL_MODE,
        "source": f"{WEIGHTS_DIR}/{BB_FOLDER} (BackdoorBench reference checkpoint, PNG path)",
        "n_heldout": PSBD_HELDOUT_SIZE,
        "analysis_clean_indices": list(range(n_eval)),
        "analysis_backdoor_indices": list(range(n_eval)),
    }
    return loaders, manifest


def write_run_provenance(
    psbd_dir: str, placement: str, operator: str, rates: tuple
) -> None:
    """A minimal provenance record, in the same shape cli.sweep writes."""
    payload = {
        "position": placement,
        "operator": operator,
        "block_range": None,
        "dropout_rates": list(rates),
        "forward_passes": FORWARD_PASSES,
        "effective_forward_passes": FORWARD_PASSES,
        "mask_seed": MASK_SEED,
        "split_seed": PSBD_SPLIT_SEED,
        "batch_size": BATCH_SIZE,
        "max_samples": None,
        "use_bfloat16": True,
        "source": "experiments/wanet_cifar10_audit/measure.py",
    }
    path = run_provenance_path(psbd_dir, placement)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def sweep_placement(
    model: torch.nn.Module,
    placement: str,
    config: dict,
    loaders: dict,
    baselines: dict,
    psbd_dir: str,
    device: torch.device,
) -> None:
    """Every rate of 1 placement: plug, run the 3 splits, save, unplug.

    Mirrors cli.sweep.sweep_rates. The operator factory is the same for every
    position in a multi-position placement (post_residual, pre_residual), which
    matches how the basis config sweeps them.
    """
    position_names = config["position_names"]
    operator_name = config["operator"]
    factory = {name: build_operator(operator_name) for name in position_names}

    write_run_provenance(psbd_dir, placement, operator_name, config["rates"])

    for rate in config["rates"]:
        handles = plug_dropout(model, "vit", position_names, factory, rate)
        try:
            for split, loader in loaders.items():
                _, baseline_labels, _ = baselines[split]
                per_pass_probs, per_pass_argmax = compute_dropout_pass_probs(
                    model,
                    loader,
                    baseline_labels,
                    device,
                    FORWARD_PASSES,
                    use_bfloat16=True,
                    seed=MASK_SEED,
                )
                save_dropout_pass_probs(
                    dropout_pass_path(psbd_dir, placement, rate, split),
                    per_pass_probs,
                    per_pass_argmax,
                )
        finally:
            unplug_dropout(handles)
        print(f"[ok] {placement} rate={rate}", flush=True)


def analyze_placement(
    psbd_dir: str, placement: str, config: dict, baselines: dict, manifest: dict
) -> dict:
    """AUROC (fractional PSU, the headline statistic) at every rate, plus the adaptive pick."""
    by_rate = {}
    for rate in config["rates"]:
        psu_ratio = {}
        psu_abs = {}
        sigma = {}
        for split in ("validation", "clean", "backdoor"):
            probs, labels, _ = baselines[split]
            per_pass_probs, per_pass_argmax = load_dropout_pass_probs(
                dropout_pass_path(psbd_dir, placement, rate, split)
            )
            psu_ratio[split] = psu_ratio_from_cache(probs, labels, per_pass_probs)
            psu_abs[split] = psu_from_cache(probs, labels, per_pass_probs)
            sigma[split] = shift_ratio(labels, per_pass_argmax)

        paired_clean_ratio = pair_clean_to_backdoor(psu_ratio["clean"], manifest)
        paired_clean_abs = pair_clean_to_backdoor(psu_abs["clean"], manifest)
        ratio_report = detection_report(
            psu_ratio["validation"], paired_clean_ratio, psu_ratio["backdoor"], 0.25
        )
        abs_report = detection_report(
            psu_abs["validation"], paired_clean_abs, psu_abs["backdoor"], 0.25
        )
        by_rate[rate] = {
            "shift_ratio_validation": sigma["validation"],
            "auroc_psu_ratio": ratio_report["auroc"],
            "auroc_psu_absolute": abs_report["auroc"],
        }

    validation_sigma = {
        rate: row["shift_ratio_validation"] for rate, row in by_rate.items()
    }
    adaptive_rate = select_rate_adaptively(validation_sigma)
    matched_rate = select_rate_at_matched_shift(validation_sigma, 0.6)

    block = {
        "by_rate": by_rate,
        "adaptive_rate": adaptive_rate,
        "adaptive_auroc_psu_ratio": by_rate[adaptive_rate]["auroc_psu_ratio"]
        if adaptive_rate
        else None,
        "matched_rate_sigma0.6": matched_rate,
        "matched_auroc_psu_ratio_sigma0.6": by_rate[matched_rate]["auroc_psu_ratio"]
        if matched_rate
        else None,
    }
    return block


def main() -> None:
    seed_everything(PSBD_SPLIT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(WEIGHTS_DIR, BB_FOLDER, "attack_result.pt")
    model = load_checkpoint("vit", checkpoint_path, device)

    loaders, manifest = build_loaders()
    psbd_dir = os.path.join(RESULTS_DIR, CACHE_FOLDER, "psbd")
    write_split_manifest(psbd_dir, manifest)

    baselines = {}
    for split, loader in loaders.items():
        cache = build_baseline_cache(model, loader, device, use_bfloat16=True)
        probs = torch.cat([row["probs"] for row in cache])
        labels = torch.cat([row["labels"] for row in cache])
        loader_labels = torch.cat([row["loader_labels"] for row in cache])
        save_baseline(baseline_path(psbd_dir, split), probs, labels, loader_labels)
        baselines[split] = (probs, labels, loader_labels)
        print(f"[baseline] {split} n={probs.shape[0]}", flush=True)

    clean_acc = (baselines["clean"][1] == baselines["clean"][2]).float().mean().item()
    asr = (baselines["backdoor"][1] == baselines["backdoor"][2]).float().mean().item()
    print(f"BackdoorBench cifar10_wanet_0_1: clean_acc={clean_acc:.4f} asr={asr:.4f}")

    report = {"clean_accuracy": clean_acc, "asr": asr, "placements": {}}
    for placement, config in PLACEMENTS.items():
        sweep_placement(model, placement, config, loaders, baselines, psbd_dir, device)
        report["placements"][placement] = analyze_placement(
            psbd_dir, placement, config, baselines, manifest
        )
        adaptive = report["placements"][placement]["adaptive_auroc_psu_ratio"]
        print(f"[analyzed] {placement} adaptive_auroc_psu_ratio={adaptive}")

    summary_path = os.path.join(RESULTS_DIR, CACHE_FOLDER, "bb_reference_metrics.json")
    with open(summary_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
