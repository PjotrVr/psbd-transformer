"""Why does masking beat additive noise at some positions and lose at others?

A ViT block is a read-write cycle, `x -> ln_1 -> attn -> +x -> ln_2 -> mlp -> +x`, and a
placement is a point in it. Some points are immediately followed by a LayerNorm and some are
not, and LayerNorm is not a passive relabelling: it subtracts the per-token mean and divides
by the per-token standard deviation. Isotropic additive noise INFLATES that standard
deviation, so dividing by it puts part of the perturbation back. Zeroing a token or a channel
removes content that no rescaling can restore.

That predicts a split with no free parameters:

    at a position where a LayerNorm follows, additive noise is partly undone and masking is not
    at a position where the tensor is already normalised, both survive equally

which is the mechanism behind H23's secondary prediction, open since the operator study, and
behind the fact that `gaussian` scores 0.758 at `before_attention_norm` and 0.896 at
`before_mlp`: the SAME operator, 0.139 apart by position alone, with 13 of 67 cells inverted
against 4 of 67.

Measured here directly on real activations: inject a perturbation of a known relative size,
then compare the relative change immediately before the next normalisation to the relative
change immediately after it. A survival ratio of 1 means the normalisation passed the
perturbation through; below 1 means it renormalised part of it away.

    PYTHONPATH=. python experiments/residual_stream_mechanism/layernorm_absorption.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os
import statistics

import torch

from models.backbones import load_checkpoint, network_core
from data.registry import DATASET_REGISTRY

# Each entry: where the perturbation is injected, and the module that consumes it next.
# "normalised" says whether a LayerNorm stands between the injection and its use.
PROBES = {
    "before_attention_norm": {"tap": "ln_1", "next": "ln_1", "normalised": True},
    "before_attention": {"tap": "self_attention", "next": None, "normalised": False},
    "before_mlp_norm": {"tap": "ln_2", "next": "ln_2", "normalised": True},
    "before_mlp": {"tap": "mlp", "next": None, "normalised": False},
}
OPERATORS = ("gaussian", "token_mask", "channel_mask")
TARGETS = (0.1, 0.2, 0.4)


def capture_input(module, model, images):
    """The tensor entering `module` on a real forward pass."""
    captured = {}
    handle = module.register_forward_pre_hook(
        lambda _m, inputs: captured.__setitem__("x", inputs[0].detach())
    )
    with torch.inference_mode():
        model(images)
    handle.remove()
    return captured["x"]


def perturb(x: torch.Tensor, operator: str, target: float, generator) -> torch.Tensor:
    """A perturbation of `x` whose relative size is approximately `target`.

    Matched on INJECTED size rather than on shift ratio, because the question here is what
    the normalisation does to a perturbation of a given size, not how far it moves a
    prediction. Shift-ratio matching is what the detection sweep does, and it compensates for
    absorption by injecting more, which is the downstream consequence rather than the cause.
    """
    if operator == "gaussian":
        noise = torch.randn(x.shape, generator=generator)
        return x + noise * (target * x.norm() / noise.norm())
    # A masking rate of target**2 lands the relative change near `target`, since a removed
    # entry contributes its whole magnitude.
    if operator == "token_mask":
        keep = (torch.rand(x.shape[:2], generator=generator) >= target**2).float()
        return x * keep.unsqueeze(-1)
    keep = (
        torch.rand((x.shape[0], x.shape[2]), generator=generator) >= target**2
    ).float()
    return x * keep.unsqueeze(1)


def survival(x, perturbed, normalise) -> tuple[float, float]:
    """Relative change before and after the normalisation that follows the injection."""
    with torch.inference_mode():
        before = ((perturbed - x).norm() / x.norm()).item()
        if normalise is None:
            return before, before
        after = (
            (normalise(perturbed) - normalise(x)).norm() / normalise(x).norm()
        ).item()
    return before, after


def measure(model, core, images, args) -> list[dict]:
    rows = []
    for block_index in args.blocks:
        block = core.encoder.layers[block_index]
        for position, spec in PROBES.items():
            x = capture_input(getattr(block, spec["tap"]), model, images)
            normalise = getattr(block, spec["next"]) if spec["next"] else None
            for operator in OPERATORS:
                for target in TARGETS:
                    generator = torch.Generator().manual_seed(
                        args.seed + block_index * 97 + hash(operator) % 1000
                    )
                    perturbed = perturb(x, operator, target, generator)
                    before, after = survival(x, perturbed, normalise)
                    rows.append(
                        {
                            "block": block_index + 1,
                            "position": position,
                            "normalised": spec["normalised"],
                            "operator": operator,
                            "target": target,
                            "relative_before": before,
                            "relative_after": after,
                            "survival": after / before if before else float("nan"),
                        }
                    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--blocks", nargs="+", type=int, default=[0, 3, 5, 8, 11])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    for folder in args.checkpoint_folder:
        path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
        with open(os.path.join(args.checkpoints_dir, folder, "args.json")) as handle:
            dataset = json.load(handle)["dataset"]
        size = DATASET_REGISTRY[dataset].image_size
        model = load_checkpoint("vit", path, device)
        torch.manual_seed(args.seed)
        images = torch.rand(args.batch, 3, size, size)
        rows = measure(model, network_core(model), images, args)

        out_dir = os.path.join(args.results_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "layernorm_absorption.json"), "w") as handle:
            json.dump({"folder_name": folder, "rows": rows}, handle, indent=2)

        print(f"\n[ok] {folder}")
        print(
            f"{'position':24s} {'LN follows':>11s} "
            + "".join(f"{o:>14s}" for o in OPERATORS)
        )
        for position, spec in PROBES.items():
            values = []
            for operator in OPERATORS:
                subset = [
                    r["survival"]
                    for r in rows
                    if r["position"] == position and r["operator"] == operator
                ]
                values.append(f"{statistics.mean(subset):14.3f}")
            print(f"{position:24s} {str(spec['normalised']):>11s} " + "".join(values))


if __name__ == "__main__":
    main()
