"""GPU pass: per-layer TAC, backdoor-direction norm and CKA on a handful of checkpoints.

The CIFAR-10 backdoor_neurons records give the depth profile of the backdoor
direction on 1 dataset. This extends the same 3 per-layer statistics to the
GTSRB and CIFAR-100 checkpoints the mechanism chapter reads, with each dataset's
benign reference probed by the same trigger as the control, and writes 1 JSON of
every number under results/_experiments/tac_layers/. mech_tac_layers.py reads
it. 1 forward pass per split per checkpoint, class-token features at every
block, fp32 so the paired difference is not dominated by bfloat16 rounding.

    PYTHONPATH=. python scripts/paper/run_tac_layers.py \\
        --checkpoints-dir checkpoints --output results/_experiments/tac_layers/tac_layers.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from lightning import seed_everything  # noqa: E402

from analysis.cases import load_latent_case  # noqa: E402
from analysis.cka import debiased_linear_cka  # noqa: E402
from analysis.direction import backdoor_direction, trigger_activated_change  # noqa: E402
from data.splits import BENIGN_PROBE_ATTACK, BENIGN_PROBE_TARGET_LABEL  # noqa: E402
from utils.provenance import current_git_commit, utc_timestamp  # noqa: E402

DEFAULT_FOLDERS = (
    "vit_gtsrb_benign",
    "vit_gtsrb_badnet_a2o_0_1",
    "vit_gtsrb_blend_0_1",
    "vit_gtsrb_bpp_0_1",
    "vit_gtsrb_lf_0_1",
    "vit_gtsrb_tact_0_1",
    "vit_gtsrb_wanet_0_1",
    "vit_cifar100_benign",
    "vit_cifar100_badnet_a2o_0_1",
    "vit_cifar100_blend_0_1",
    "vit_cifar100_bpp_0_1",
    "vit_cifar100_lf_0_1",
    "vit_cifar100_tact_0_1",
    "vit_cifar100_badnet_a2o_0_01",
    "vit_cifar100_blend_0_01",
    "vit_tiny_benign",
    "vit_tiny_badnet_a2o_0_1",
    "vit_tiny_blend_0_1",
)
DEFAULT_OUTPUT = os.path.join("results", "_experiments", "tac_layers", "tac_layers.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="*", default=list(DEFAULT_FOLDERS))
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def layer_profile(clean: dict[int, torch.Tensor], triggered: dict[int, torch.Tensor]) -> list[dict]:
    """Per layer: mean and max TAC, relative direction norm and CKA, on (num_samples, dim) features."""
    rows = []
    for layer in sorted(clean):
        clean_features = clean[layer].float()  # (num_samples, dim)
        triggered_features = triggered[layer].float()  # (num_samples, dim)
        tac = trigger_activated_change(clean_features, triggered_features)  # (dim,)
        direction = backdoor_direction(clean_features, triggered_features)  # (dim,)
        # Divided by the mean clean norm so growth is not the residual stream growing.
        scale = clean_features.norm(dim=1).mean().clamp_min(1e-8)
        rows.append(
            {
                "layer": layer,
                "tac_mean": float(tac.mean()),
                "tac_max": float(tac.max()),
                "rel_direction_norm": float(direction.norm() / scale),
                "cka": float(debiased_linear_cka(clean_features, triggered_features)),
            }
        )
    return rows


def measure_folder(folder: str, args: argparse.Namespace, device: torch.device) -> dict:
    benign = "benign" in folder
    seed_everything(args.seed)
    case = load_latent_case(
        folder,
        samples=args.samples,
        batch_size=args.batch_size,
        checkpoint_root=args.checkpoints_dir,
        raw_data_dir=args.raw_data_dir,
        device=device,
        seed=args.seed,
        probe_attack=BENIGN_PROBE_ATTACK if benign else None,
        probe_target_label=BENIGN_PROBE_TARGET_LABEL if benign else None,
        use_bfloat16=False,
    )
    clean = {layer: tensor.cpu() for layer, tensor in case.clean_features.items()}
    triggered = {layer: tensor.cpu() for layer, tensor in case.backdoor_features.items()}
    record = {
        "folder": folder,
        "dataset": case.dataset,
        "attack": "benign" if benign else case.attack,
        "probe_attack": case.attack,
        "target_label": case.target_label,
        "samples": int(next(iter(clean.values())).shape[0]),
        "layers": layer_profile(clean, triggered),
    }
    return record


def write_output(path: str, records: list[dict], args: argparse.Namespace) -> None:
    payload = {
        "generator": "scripts/paper/run_tac_layers.py",
        "written_at": utc_timestamp(),
        "git_commit": current_git_commit(),
        "samples": args.samples,
        "seed": args.seed,
        "precision": "float32",
        "records": records,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("this pass needs a GPU, refusing a CPU-bound run")
    records = []
    for folder in args.folders:
        if not os.path.exists(os.path.join(args.checkpoints_dir, folder, "attack_result.pt")):
            print(f"skip {folder}: no checkpoint")
            continue
        started = time.time()
        records.append(measure_folder(folder, args, device))
        peak = max(records[-1]["layers"], key=lambda row: row["rel_direction_norm"])
        print(f"{folder}: peak layer {peak['layer']} rel norm {peak['rel_direction_norm']:.3f}, {time.time() - started:.0f}s")
        write_output(args.output, records, args)
    print(f"wrote {args.output} with {len(records)} records")


if __name__ == "__main__":
    main()
