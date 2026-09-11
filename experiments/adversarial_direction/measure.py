"""Do PGD adversarial examples ride the backdoor direction, the way Karayalcin et al. find.

"Backdoor Directions in Vision Transformers" (Karayalcin, Krcek, Chen and Picek,
arXiv 2603.10806, Section 6) runs untargeted PGD from clean CIFAR-100 test images
and separately from triggered ones, on ViT-B/16. Their Experimental Setup, quoted:

    "For both subsections, we use PGD with l_inf-norm and epsilon = 8/255. We
    start from either clean or backdoored test images and run 5 or 15 steps for
    clean and backdoored examples, respectively."

Table 2 reports the resulting share of adversarial examples landing on the
target class (from clean images, BPP 18.7 to 41.5%, WaNet and BadNet lower) and
on the original class (from triggered images, 20 to 50%). Section 6 defines the
per-image activation shift and its cosine similarity to the backdoor direction:

    original form
        v^l_i = a^l_i - x^l_i
        cos(v^l_i, r^l) = (v^l_i . r^l) / (||v^l_i|| ||r^l||)
    symbols
        x_i        a clean or triggered starting image
        a_i        the PGD adversarial example built from x_i
        x^l_i      the [CLS] token residual stream of x_i at layer l
        a^l_i      the [CLS] token residual stream of a_i at layer l
        v^l_i      the activation shift PGD caused at layer l
        r^l        the backdoor direction of layer l (analysis.directions, this
                   repo's port of their Eq. 1 estimator)

They find WaNet and BPP's shift has high cosine similarity to r^l in middle
layers and BadNet's does not, and that from triggered images the (negative)
cosine similarity is strongly negative in late layers whether or not the image
reverts to its original class.

Deviations from their stated protocol, since the step size is not given in the
paper: step size is epsilon / 4, and PGD starts at the clean or triggered image
itself with no random initialization, since neither is stated either. Untargeted
means maximizing cross entropy against the label the image currently carries:
the true label from a clean start, the trigger-induced target label from a
triggered start, which is what lets "reverts to the original class" be measured
on the far side. The clean-image share is compared against the same PGD run on
the benign checkpoint of the same dataset, at the same target class, since
Table 2 has no such baseline, and this comparison clarifies whether the
target-class pull is backdoor-specific or a generic PGD artifact of that class.

paired_rows, residual_stream and directions are imported from
experiments.whole_network_erasure.measure rather than reimplemented: they
already build the paired clean/triggered rows, capture every layer's tokens and
estimate the backdoor direction on the first 500 pairs exactly as this
experiment also needs. Every rate and cosine number below is read on the rows
after those first 500, so the direction is never validated on the data it came
from.

    PYTHONPATH=. python experiments/adversarial_direction/measure.py \
        --checkpoints-dir /lustre/home/pstika/projects/PSBD-ViT/checkpoints \
        --raw-data-dir /lustre/home/pstika/projects/PSBD-ViT/raw_data \
        --folders vit_cifar100_badnet_a2o_0_1 vit_cifar100_bpp_0_1 \
        --max-samples 1000
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from data.registry import DATASET_REGISTRY  # noqa: E402
from data.splits import (  # noqa: E402
    BENIGN_PROBE_ATTACK,
    BENIGN_PROBE_TARGET_LABEL,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from defences.inference import forward_logits, frozen_parameters  # noqa: E402
from detectors.strip import normalization_buffers  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from experiments.whole_network_erasure.measure import (  # noqa: E402
    ESTIMATE_PAIRS,
    NUM_LAYERS,
    directions,
    paired_rows,
    residual_stream,
)
from models.backbones import load_checkpoint  # noqa: E402

SLUG = "adversarial_direction"
EPSILON = 8.0 / 255.0
STEP_SIZE = EPSILON / 4.0  # the paper states no step size, so this is our choice
CLEAN_STEPS = 5
BACKDOOR_STEPS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--max-samples", type=int, default=1000, help="analysis rows read per split"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def benign_checkpoint_path(
    checkpoints_dir: str, architecture: str, dataset: str
) -> str:
    """Where this dataset's benign reference checkpoint lives, by naming convention."""
    folder = f"{architecture}_{dataset}_benign"
    path = os.path.join(checkpoints_dir, folder, "attack_result.pt")
    return path


def reference_model(
    metadata: dict,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    cache: dict,
) -> nn.Module:
    """The benign checkpoint of this dataset, a fresh load cached by dataset.

    The checkpoint under analysis stands in for its own reference when it is
    already the benign model, so the target-class share is compared against
    itself rather than loading the identical weights twice.
    """
    if metadata["attack"] == "benign":
        return model

    dataset = metadata["dataset"]
    if dataset in cache:
        return cache[dataset]

    path = benign_checkpoint_path(
        args.checkpoints_dir, metadata["architecture"], dataset
    )
    loaded = load_checkpoint(metadata["architecture"], path, device)
    cache[dataset] = loaded
    return loaded


def pgd_adversarial(
    model: nn.Module,
    device: torch.device,
    normalized_images: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    steps: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Untargeted L-inf PGD, batched, returning adversarial images and their predictions.

    original form
        a^{t+1} = Proj_{B_eps(x) cap [0, 1]} (a^t + alpha * sign(grad_a L(f(a^t), y)))
    symbols
        x          the starting image, in pixel space
        a^t        the adversarial iterate at step t, a^0 = x
        y          the label being maximized away from (labels)
        alpha      STEP_SIZE, epsilon      EPSILON
        L          cross entropy

    normalized_images is (n, channels, height, width) in the model's normalized
    input space, labels is (n,) long. PGD itself runs in pixel space, so every
    step denormalizes, perturbs and renormalizes, the round trip
    detectors.strip.normalization_buffers already does for STRIP's overlay.
    Returned "images" is (n, channels, height, width), back in normalized space
    so it feeds the reused residual_stream and prediction calls unchanged, and
    "predicted" is the (n,) argmax on the finished adversarial batch.
    """
    adv_chunks: list[torch.Tensor] = []
    predicted_chunks: list[torch.Tensor] = []
    with frozen_parameters(model):
        for start in range(0, len(normalized_images), batch_size):
            batch = normalized_images[start : start + batch_size].to(
                device
            )  # (batch, channels, height, width)
            batch_labels = labels[start : start + batch_size].to(device)  # (batch,)

            pixels = (batch * std + mean).clamp(0.0, 1.0)  # (batch, C, H, W)
            delta = torch.zeros_like(pixels)  # (batch, C, H, W)
            for _ in range(steps):
                delta.requires_grad_(True)
                perturbed_normalized = (pixels + delta - mean) / std
                logits = forward_logits(
                    model, perturbed_normalized, device, use_bfloat16=False
                )  # (batch, num_classes)
                loss = F.cross_entropy(logits, batch_labels)
                (gradient,) = torch.autograd.grad(loss, delta)  # (batch, C, H, W)

                # Project into the epsilon ball first and the pixel range second,
                # so delta always describes a valid image.
                delta = (delta.detach() + STEP_SIZE * gradient.sign()).clamp(
                    -EPSILON, EPSILON
                )
                delta = ((pixels + delta).clamp(0.0, 1.0) - pixels).detach()

            adversarial_pixels = (pixels + delta).clamp(0.0, 1.0)  # (batch, C, H, W)
            adversarial_normalized = (adversarial_pixels - mean) / std

            with torch.no_grad():
                adversarial_logits = forward_logits(
                    model, adversarial_normalized, device, use_bfloat16=False
                )  # (batch, num_classes)
                predicted = adversarial_logits.argmax(dim=1)  # (batch,)

            adv_chunks.append(adversarial_normalized.detach().cpu())
            predicted_chunks.append(predicted.cpu())

    result = {
        "images": torch.cat(adv_chunks),  # (n, C, H, W)
        "predicted": torch.cat(predicted_chunks),  # (n,)
    }
    return result


def layerwise_cosine(
    adversarial_stream: torch.Tensor,
    original_stream: torch.Tensor,
    layer_directions: torch.Tensor,
    random_direction: torch.Tensor,
) -> list[dict]:
    """Mean and median cosine similarity of the [CLS] shift to r^l and to the null.

    adversarial_stream and original_stream are both (n, 13, tokens, dim), the
    residual_stream output at every layer for the adversarial and the starting
    batch. layer_directions is (13, dim), the backdoor direction of every layer,
    and random_direction is (dim,), the same null vector reused across layers so
    it measures how much cosine similarity a layer produces by chance rather than
    by direction.
    """
    class_token_shift = (
        adversarial_stream[:, :, 0, :] - original_stream[:, :, 0, :]
    )  # (n, 13, dim)

    rows = []
    for layer in range(class_token_shift.shape[1]):
        shift = class_token_shift[:, layer, :]  # (n, dim)
        direction = layer_directions[layer].unsqueeze(0).expand_as(shift)  # (n, dim)
        null = random_direction.unsqueeze(0).expand_as(shift)  # (n, dim)

        direction_cosine = F.cosine_similarity(shift, direction, dim=1)  # (n,)
        null_cosine = F.cosine_similarity(shift, null, dim=1)  # (n,)

        rows.append(
            {
                "layer": layer,
                "direction_cosine_mean": float(direction_cosine.mean()),
                "direction_cosine_median": float(direction_cosine.median()),
                "random_cosine_mean": float(null_cosine.mean()),
                "random_cosine_median": float(null_cosine.median()),
            }
        )
    return rows


def analyse(
    folder: str, args: argparse.Namespace, device: torch.device, benign_cache: dict
) -> dict:
    checkpoint = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint)
    benign = metadata["attack"] == "benign"
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        max_samples=args.max_samples,
        probe_attack=BENIGN_PROBE_ATTACK if benign else None,
        probe_target_label=BENIGN_PROBE_TARGET_LABEL if benign else None,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint, device)
    pairs = paired_rows(loaders, manifest)
    found = directions(model, pairs, device, args.batch_size)  # {"cls": (13, dim), ...}
    reference = reference_model(metadata, model, args, device, benign_cache)

    dataset = metadata["dataset"]
    spec = DATASET_REGISTRY[dataset]
    mean, std = normalization_buffers(spec.mean, spec.std, device, torch.float32)
    target_label = manifest["probe_target_label"]

    evaluation = slice(ESTIMATE_PAIRS, None)
    clean_images = pairs["clean"][evaluation]  # (n_clean, C, H, W)
    true_label = pairs["true_label"][evaluation]  # (n_clean,)
    backdoor_images = pairs["backdoor"][evaluation]  # (n_backdoor, C, H, W)
    trigger_label = pairs["target"][
        evaluation
    ]  # (n_backdoor,), the label the trigger induces

    generator = torch.Generator().manual_seed(args.seed)
    embedding_dim = found["cls"].shape[-1]
    random_direction = torch.randn(embedding_dim, generator=generator)
    random_direction = random_direction / random_direction.norm()  # (dim,)

    clean_start = pgd_adversarial(
        model, device, clean_images, true_label, mean, std, CLEAN_STEPS, args.batch_size
    )
    backdoor_start = pgd_adversarial(
        model,
        device,
        backdoor_images,
        trigger_label,
        mean,
        std,
        BACKDOOR_STEPS,
        args.batch_size,
    )
    reference_clean_start = pgd_adversarial(
        reference,
        device,
        clean_images,
        true_label,
        mean,
        std,
        CLEAN_STEPS,
        args.batch_size,
    )

    clean_original_stream = residual_stream(
        model, clean_images, device, args.batch_size
    )
    clean_adversarial_stream = residual_stream(
        model, clean_start["images"], device, args.batch_size
    )
    backdoor_original_stream = residual_stream(
        model, backdoor_images, device, args.batch_size
    )
    backdoor_adversarial_stream = residual_stream(
        model, backdoor_start["images"], device, args.batch_size
    )

    report = {
        "folder": folder,
        "attack": metadata["attack"],
        "dataset": dataset,
        "poison_rate": metadata.get("poison_rate"),
        "target_label": target_label,
        "n_estimate": ESTIMATE_PAIRS,
        "n_eval_clean": int(len(clean_images)),
        "n_eval_backdoor": int(len(backdoor_images)),
        "epsilon": EPSILON,
        "step_size": STEP_SIZE,
        "clean_start_steps": CLEAN_STEPS,
        "backdoor_start_steps": BACKDOOR_STEPS,
        "clean_start_target_share_model": float(
            (clean_start["predicted"] == target_label).float().mean()
        ),
        "clean_start_target_share_benign_reference": float(
            (reference_clean_start["predicted"] == target_label).float().mean()
        ),
        "backdoor_start_reversion_share": float(
            (backdoor_start["predicted"] == true_label).float().mean()
        ),
        "clean_start_layerwise_cosine": layerwise_cosine(
            clean_adversarial_stream,
            clean_original_stream,
            found["cls"],
            random_direction,
        ),
        "backdoor_start_layerwise_cosine": layerwise_cosine(
            backdoor_adversarial_stream,
            backdoor_original_stream,
            found["cls"],
            random_direction,
        ),
    }
    return report


def summary_line(report: dict) -> str:
    """1 line: the 2 target-class shares, the reversion share and the peak middle-layer cosine."""
    middle_layers = report["clean_start_layerwise_cosine"][
        NUM_LAYERS // 2 : NUM_LAYERS + 1
    ]
    peak_middle = max(row["direction_cosine_mean"] for row in middle_layers)
    line = (
        f"{report['folder']:32s} clean->target model {report['clean_start_target_share_model']:.3f} "
        f"benign-ref {report['clean_start_target_share_benign_reference']:.3f} | "
        f"backdoor->original {report['backdoor_start_reversion_share']:.3f} | "
        f"peak mid-layer cosine {peak_middle:.3f}"
    )
    return line


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benign_cache: dict = {}
    for folder in args.folders:
        report = analyse(folder, args, device, benign_cache)
        path = experiment_result_path(SLUG, f"{folder}.json", args.results_dir)
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)
        print(summary_line(report), flush=True)


if __name__ == "__main__":
    main()
