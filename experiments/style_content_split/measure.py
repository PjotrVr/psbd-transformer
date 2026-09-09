"""Does a backdoor trigger ride in an image's STYLE or in its CONTENT?

The hypothesis: a trigger is a texture, and the class evidence is structure. If that is
right, a style-content decomposition puts the poison entirely on one side of the split, and
normalising the style is a purification defence.

The instrument is exact and needs no trained network. Fourier PHASE carries structure and
AMPLITUDE carries texture (Oppenheim and Lim, 1981), so for a suspect image x and a clean
donor d:

    mix(x, d, lam) = IFFT( [(1-lam)|FFT(x)| + lam|FFT(d)|] * exp(i * angle(FFT(x))) )

The suspect's PHASE, hence its structure, is always kept; its AMPLITUDE, hence its texture,
is interpolated toward a clean donor's by lam. lam = 0 is the original image and lam = 1 is
a full texture replacement. Exact, invertible up to the real projection, no learned decoder.

A BINARY swap was tried first and is uninterpretable: replacing the phase outright takes
clean accuracy from 0.972 to 0.060 on gtsrb badnet, so the image is destroyed and its ASR
drop says nothing. Interpolating is what the amplitude-mix augmentation literature (FACT,
APR) does for the same reason, and it turns a single number into the tradeoff curve a
defence is actually judged on.

Prediction, per attack, which is the point: this is a taxonomy, not one number.

| attack | trigger | rides in |
|---|---|---|
| badnet | a 3x3 corner patch | structure, so PHASE |
| blend, sig, lf, bpp | a global pattern, a sinusoid, a low band, a quantisation | texture, so AMPLITUDE |
| wanet | an elastic warp, which adds no pattern and moves content | NEITHER |

A result splitting the attacks that way is a finding. One that does not is a refutation.

THE CONTROL THAT DECIDES IT. A swapped image is not a natural image, so an ASR drop may
just be the image being wrecked. Clean accuracy under the SAME transform is therefore
measured on the same images, and an ASR drop only counts where clean accuracy survives.
Both transforms are applied to the clean split too, so every number has its own control.

    PYTHONPATH=. python experiments/style_content_split/measure.py \
        --checkpoint-folder vit_gtsrb_adaptive_blend_0_05
"""

import argparse
import json
import os

import torch

from data.splits import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models.backbones import load_checkpoint
from data.registry import DATASET_REGISTRY

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# How much clean accuracy a deployment will spend. 2 points is already generous.
CLEAN_BUDGET = 0.02


def normalization_buffers(mean, std, device, dtype):
    """Dataset statistics shaped to broadcast over (batch, C, H, W)."""
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean_tensor, std_tensor


def mix_amplitude(
    suspect: torch.Tensor, donor: torch.Tensor, weight: float
) -> torch.Tensor:
    """Keep the suspect's phase, pull its amplitude toward the donor's by `weight`.

    The inverse transform is complex only through numerical error, so the real part is
    taken and the result clamped back into the valid pixel range.
    """
    suspect_spectrum = torch.fft.fft2(suspect, dim=(-2, -1))
    donor_spectrum = torch.fft.fft2(donor, dim=(-2, -1))
    amplitude = (1.0 - weight) * suspect_spectrum.abs() + weight * donor_spectrum.abs()
    combined = amplitude * torch.exp(1j * suspect_spectrum.angle())
    return torch.fft.ifft2(combined, dim=(-2, -1)).real.clamp(0.0, 1.0)


@torch.inference_mode()
def donor_bank(loader, count: int, device: torch.device) -> torch.Tensor:
    """Clean validation images to donate a spectrum, in normalized space."""
    collected = []
    for images, _ in loader:
        collected.append(images)
        if sum(len(batch) for batch in collected) >= count:
            break
    return torch.cat(collected)[:count].to(device)


@torch.inference_mode()
def accuracy_under_mix(
    model, loader, donors, mean, std, device, weight: float
) -> float:
    """Fraction of samples the model still assigns the loader's own label.

    On the backdoor split the loader's label is the attack-success label, so this is ASR.
    On the clean split it is clean accuracy, which is this experiment's control.
    """
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if weight > 0.0:
            mean_tensor, std_tensor = normalization_buffers(
                mean, std, device, images.dtype
            )
            pixels = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
            index = torch.randint(0, len(donors), (images.size(0),), device=device)
            donor_pixels = (donors[index] * std_tensor + mean_tensor).clamp(0.0, 1.0)
            mixed = mix_amplitude(pixels, donor_pixels, weight)
            images = (mixed - mean_tensor) / std_tensor
        predicted = model(images).argmax(dim=1)
        correct += int((predicted == labels).sum())
        total += len(labels)
    return correct / total if total else float("nan")


def analyse(folder: str, args) -> dict:
    checkpoint_path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint_path)
    probe = "badnet_a2o" if metadata["attack"] == "benign" else None
    loaders, _ = build_psbd_loaders_from_checkpoint(
        checkpoint_path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        probe_attack=probe,
        probe_target_label=0 if probe else None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(metadata["architecture"], checkpoint_path, device)
    spec = DATASET_REGISTRY[metadata["dataset"]]
    torch.manual_seed(PSBD_SPLIT_SEED)
    donors = donor_bank(loaders["validation"], args.donors, device)

    report = {
        "folder_name": folder,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate"),
        "donors": args.donors,
    }
    curve = []
    for weight in WEIGHTS:
        curve.append(
            {
                "weight": weight,
                "asr": accuracy_under_mix(
                    model,
                    loaders["backdoor"],
                    donors,
                    spec.mean,
                    spec.std,
                    device,
                    weight,
                ),
                "clean_accuracy": accuracy_under_mix(
                    model, loaders["clean"], donors, spec.mean, spec.std, device, weight
                ),
            }
        )
    report["curve"] = curve
    # The deployable point: the largest ASR reduction available while clean accuracy stays
    # within CLEAN_BUDGET of the untransformed model. A purification that buys its ASR drop
    # with accuracy is not a defence, and this is where most of them fail.
    baseline = curve[0]
    affordable = [
        row
        for row in curve
        if row["clean_accuracy"] >= baseline["clean_accuracy"] - CLEAN_BUDGET
    ]
    best = min(affordable, key=lambda row: row["asr"]) if affordable else baseline
    report["operating_point"] = best
    report["baseline"] = baseline
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--donors", type=int, default=256)
    args = parser.parse_args()

    for folder in args.checkpoint_folder:
        try:
            report = analyse(folder, args)
        except Exception as error:  # noqa: BLE001
            print(f"[FAILED] {folder}: {type(error).__name__}: {error}")
            continue
        out_dir = os.path.join(args.results_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "style_content_split.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        base, best = report["baseline"], report["operating_point"]
        print(f"[ok] {folder}")
        print(
            "      lam  " + " ".join(f"{row['weight']:5.1f}" for row in report["curve"])
        )
        print("      ASR  " + " ".join(f"{row['asr']:5.3f}" for row in report["curve"]))
        print(
            "      CA   "
            + " ".join(f"{row['clean_accuracy']:5.3f}" for row in report["curve"])
        )
        print(
            f"      deployable (CA within {CLEAN_BUDGET:.0%}): lam={best['weight']:.1f} "
            f"ASR {base['asr']:.3f} -> {best['asr']:.3f}, "
            f"CA {base['clean_accuracy']:.3f} -> {best['clean_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
