"""How much of the trigger's push on the target logit lies along the mean shift?

Section 9 of H16 leaves a specific hole. Under Adam, removing the mean
clean-to-triggered direction destroys the backdoor. Under SAM at rho 0.2 it does
nothing, at any layer and at any rank, even though clean and triggered features stay
perfectly linearly separable. So the direction the representations differ along and
the direction the classifier decides along have come apart, and this measures by how
much.

Ablating the target class readout direction would answer nothing: it zeroes the
target logit for every input, so ASR falls to 0 trivially and clean accuracy on that
class with it. The informative quantity is a decomposition, not a deletion.

Work in POST-LayerNorm space, because that is where the head is linear:

    logit_j(x) = w_j . LN(h(x)) + b_j          exact
    delta_j    = w_j . (z_trig - z_clean)      the trigger's push on logit j

LayerNorm is per-token, so applying it to the CLS row alone is identical to applying
it to the sequence. Folding the LN gain into w and working pre-LN would NOT be
equivalent, because LN also recenters and rescales per sample.

Split each sample's shift into the part along the mean shift direction and the rest:

    d           = unit( mean over samples of (z_trig - z_clean) )
    along_i     = (delta_i . d) d
    residual_i  = delta_i - along_i

The obvious summary, the ratio of mean pushes, is worthless here and it is worth
recording why. The residual has **exactly zero mean** by construction, since
mean(delta) - |mean(delta)| d = 0, so its contribution to the MEAN target logit is
identically 0 and the ratio of means is 1 for every model, backdoored or not. That
is an algebraic identity, not a measurement.

So compare MAGNITUDES, which are not structurally fixed:

    residual_share = mean |w_t . residual_i| / mean |w_t . delta_i|

Near 0 means each sample's push on the target logit really is the common shift, so
deleting d deletes the attack. Near 1 means the push is per-sample and lives off d
entirely, so deleting d is beside the point however large d is.

Example
    PYTHONPATH=. python scripts/backdoor_neurons/logit_decomposition.py
"""

import argparse
import json
import os

import torch
from lightning import seed_everything

from analysis.features import extract_layer_features
from attacks import build_attack, default_config
from defences.checkpoint_eval import read_checkpoint_metadata, resolve_probe_attack
from models import load_checkpoint, vit_core
from scripts.backdoor_direction_layers.measure import build_paired_loaders
from utils.config import DATASET_REGISTRY

FINAL_BLOCK = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack",
        nargs="*",
        default=["badnet_a2o", "blend", "bpp", "lf"],
    )
    parser.add_argument("--rho", nargs="*", default=["", "0_1", "0_2"])
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--architecture", default="vit")
    parser.add_argument("--poison-tag", default="0_1")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layer", type=int, default=FINAL_BLOCK)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def measure(folder: str, args: argparse.Namespace, device) -> dict | None:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    if not os.path.exists(path):
        return None
    metadata = read_checkpoint_metadata(path)
    benign = metadata["attack"] == "benign"
    attack_name, target_label = resolve_probe_attack(
        metadata,
        "badnet_a2o" if benign else None,
        0 if benign else None,
    )
    spec = DATASET_REGISTRY[metadata["dataset"]]
    attack = build_attack(
        attack_name, default_config(attack_name), spec.image_size, target_label
    )
    clean_loader, triggered_loader, _ = build_paired_loaders(
        metadata["dataset"],
        attack,
        args.raw_data_dir,
        args.batch_size,
        args.samples,
        args.seed,
    )
    model = load_checkpoint(metadata["architecture"], path, device)
    core = vit_core(model)

    seed_everything(args.seed)
    clean = extract_layer_features(model, clean_loader, device, False, "cls")[
        args.layer
    ]
    seed_everything(args.seed)
    triggered = extract_layer_features(model, triggered_loader, device, False, "cls")[
        args.layer
    ]

    with torch.inference_mode():
        clean_z = core.encoder.ln(clean.to(device)).float().cpu()
        triggered_z = core.encoder.ln(triggered.to(device)).float().cpu()
    readout = core.heads.head.weight.detach().float().cpu()[target_label]

    shift = triggered_z - clean_z
    mean_direction = shift.mean(dim=0)
    unit = mean_direction / mean_direction.norm().clamp_min(1e-8)

    delta_logit = shift @ readout
    along = (shift @ unit).unsqueeze(1) * unit.unsqueeze(0)
    residual_logit = (shift - along) @ readout

    total_magnitude = float(delta_logit.abs().mean())
    residual_share = (
        float(residual_logit.abs().mean()) / total_magnitude
        if total_magnitude > 1e-8
        else None
    )

    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "rho": metadata.get("rho"),
        "target_label": target_label,
        "layer": args.layer,
        "samples": int(len(shift)),
        "mean_delta_target_logit": float(delta_logit.mean()),
        "mean_abs_delta_target_logit": total_magnitude,
        "residual_share_of_target_push": residual_share,
        # Spread of the per-sample push relative to its mean. A pure common shift
        # has a small ratio; a per-sample attack has a large one.
        "push_dispersion": float(
            delta_logit.std() / max(abs(delta_logit.mean()), 1e-8)
        ),
        "cosine_mean_direction_to_readout": float(
            torch.nn.functional.cosine_similarity(unit, readout, dim=0)
        ),
        "mean_shift_norm": float(shift.norm(dim=1).mean()),
        "mean_direction_norm": float(mean_direction.norm()),
        # How much of the per-sample shift is the common component at all, in the
        # representation rather than in the logit.
        "mean_direction_share_of_shift": float(
            mean_direction.norm() / shift.norm(dim=1).mean().clamp_min(1e-8)
        ),
    }


def rho_label(row: dict) -> str:
    rho = row.get("rho")
    return "adam" if not rho else f"sam {float(rho):g}"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("How much of the trigger's push on the TARGET logit lies along the mean")
    print("clean-to-triggered shift. Post-LayerNorm, where the head is linear.\n")
    print(
        f"{'checkpoint':40} {'d logit_t':>10} {'resid share':>12} "
        f"{'dispersion':>11} {'|d|/|shift|':>12} {'cos(d,w_t)':>11}"
    )
    print("-" * 100)

    rows = []
    for attack in args.attack:
        for rho in args.rho:
            stem = f"{args.architecture}_{args.dataset}_{attack}"
            if attack != "benign":
                stem += f"_{args.poison_tag}"
            folder = stem if rho == "" else f"{stem}_sam_rho_{rho}"
            try:
                row = measure(folder, args, device)
            except Exception as error:
                print(f"{folder:40} FAILED {type(error).__name__}: {error}", flush=True)
                continue
            if row is None:
                continue
            rows.append(row)
            share = row["residual_share_of_target_push"]
            print(
                f"{folder:40} {row['mean_delta_target_logit']:>10.3f} "
                f"{'--' if share is None else f'{share:.3f}':>12} "
                f"{row['push_dispersion']:>11.3f} "
                f"{row['mean_direction_share_of_shift']:>12.3f} "
                f"{row['cosine_mean_direction_to_readout']:>11.3f}",
                flush=True,
            )

    if rows:
        out = os.path.join(args.results_dir, "logit_decomposition.json")
        with open(out, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
