"""Causal tracing: which (layer, token) carries the backdoor, measured by patching.

Everything measured so far is correlational. Patching is the causal version: run the model on
a triggered image, overwrite ONE (layer, token group) with the value it takes on the SAME
image without the trigger, and see how much of the clean answer comes back.

This setting has an advantage most causal-tracing papers do not. Zhang and Nanda (ICLR 2024)
recommend symmetric counterfactual replacement over Gaussian corruption, because noise pushes
activations off distribution and produced qualitatively wrong circuits in their tests. Paired
clean and triggered images ARE that counterfactual: same image, minimal edit. The recommended
baseline comes for free.

Metric is the normalised logit difference used by Wang et al. (Interpretability in the Wild):

    M = logit(target) - logit(true)
    IE = (M_patched - M_triggered) / (M_clean - M_triggered)

so 0 is no recovery and 1 is the clean answer fully restored. Values outside that range are
kept rather than clipped, because a component that pushes past clean or against it is a
finding, not an artifact.

Both directions are run, since they answer different questions and can disagree:

    denoise   clean value patched into a triggered run: is this site SUFFICIENT to restore
    noise     triggered value patched into a clean run: is it SUFFICIENT to induce

Token groups rather than all 197 positions, so the sweep is affordable and each row means
something: the CLS token, the trigger's own tokens, the highest-norm non-trigger token, and a
random group of the same size as the trigger's for the null.

    PYTHONPATH=. python experiments/residual_stream_mechanism/activation_patching.py \
        --checkpoint-folder vit_gtsrb_badnet_a2o_0_1
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from attacks import apply_config_overrides, build_attack, default_config
from defences.checkpoint_eval import (
    PSBD_SPLIT_SEED,
    build_psbd_loaders_from_checkpoint,
    read_checkpoint_metadata,
)
from models import load_checkpoint, network_core
from utils.config import DATASET_REGISTRY

MODEL_INPUT, PATCH = 224, 16
GRID = MODEL_INPUT // PATCH


def trigger_tokens(metadata) -> torch.Tensor:
    size = DATASET_REGISTRY[metadata["dataset"]].image_size
    config = apply_config_overrides(
        default_config(metadata["attack"]), metadata.get("attack_config_overrides")
    )
    attack = build_attack(metadata["attack"], config, size, metadata["target_label"])
    plant = attack.apply_trigger_eval or attack.apply_trigger
    base = torch.rand(3, size, size, generator=torch.Generator().manual_seed(0))
    delta = (plant(base.clone(), 0) - base).abs().sum(dim=0, keepdim=True)
    up = F.interpolate(
        delta.unsqueeze(0), size=(MODEL_INPUT, MODEL_INPUT), mode="bilinear"
    )[0, 0]
    per_token = (
        up.reshape(GRID, PATCH, GRID, PATCH)
        .permute(0, 2, 1, 3)
        .reshape(-1, PATCH * PATCH)
    )
    return (per_token.sum(dim=1) > 1e-6).nonzero(as_tuple=True)[0]


@torch.inference_mode()
def cache_stream(model, core, images):
    """Residual stream entering every block, plus both branch writes."""
    store = {}
    handles = []
    for index, block in enumerate(core.encoder.layers):
        handles.append(
            block.register_forward_pre_hook(
                lambda _m, inputs, index=index: store.__setitem__(
                    ("resid", index), inputs[0].detach()
                )
            )
        )
        handles.append(
            block.dropout.register_forward_hook(
                lambda _m, _i, out, index=index: store.__setitem__(
                    ("attn", index), out.detach()
                )
            )
        )
        handles.append(
            block.mlp.register_forward_hook(
                lambda _m, _i, out, index=index: store.__setitem__(
                    ("mlp", index), out.detach()
                )
            )
        )
    logits = model(images)
    for handle in handles:
        handle.remove()
    return store, logits


def patch_hook(block, site, donor, positions):
    """Overwrite `positions` of one tensor with the donor's values, for one forward pass."""
    if site == "resid":

        def hook(_m, inputs):
            x = inputs[0].clone()
            x[:, positions, :] = donor[:, positions, :]
            return (x,) + inputs[1:]

        return block.register_forward_pre_hook(hook)

    def hook(_m, _i, out):
        y = out.clone()
        y[:, positions, :] = donor[:, positions, :]
        return y

    module = block.dropout if site == "attn" else block.mlp
    return module.register_forward_hook(hook)


@torch.inference_mode()
def sweep(
    model,
    core,
    images_from,
    donor_cache,
    target,
    true_label,
    baseline,
    reference,
    groups,
    sites,
):
    """Normalised recovery for every (site, layer, token group)."""
    rows = []
    for site in sites:
        for layer in range(len(core.encoder.layers)):
            for name, positions in groups.items():
                handle = patch_hook(
                    core.encoder.layers[layer],
                    site,
                    donor_cache[(site, layer)],
                    positions,
                )
                logits = model(images_from)
                handle.remove()
                metric = (
                    logits.gather(1, target[:, None])
                    - logits.gather(1, true_label[:, None])
                ).squeeze(1)
                # Aggregate numerator and denominator separately. Per-sample division
                # explodes whenever one image's clean and triggered metrics happen to be
                # close, and the denominator is legitimately negative, so a positive clamp
                # turns those samples into enormous spurious values.
                rows.append(
                    {
                        "site": site,
                        "layer": layer + 1,
                        "group": name,
                        "numerator": float((metric - baseline).sum()),
                        "denominator": float((reference - baseline).sum()),
                        "n": int(len(metric)),
                    }
                )
    return rows


def analyse(folder, args) -> dict:
    path = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(path)
    if metadata["attack"] == "benign":
        raise ValueError("patching needs a real trigger; benign has none")
    tokens = trigger_tokens(metadata)
    loaders, manifest = build_psbd_loaders_from_checkpoint(
        path,
        seed=PSBD_SPLIT_SEED,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(metadata["architecture"], path, device).eval()
    core = network_core(model)
    target_label = metadata["target_label"]

    clean_indices = manifest["analysis_clean_indices"]
    row_of = {original: row for row, original in enumerate(clean_indices)}
    rows_needed = [
        row_of[o] for o in manifest["analysis_backdoor_indices"][: args.limit]
    ]
    clean_set = loaders["clean"].dataset
    bd_set = loaders["backdoor"].dataset

    generator = torch.Generator().manual_seed(args.seed)
    random_group = torch.randperm(196, generator=generator)[: len(tokens)] + 1
    groups = {
        "cls": torch.tensor([0]),
        "trigger": tokens + 1,
        "random_same_size": random_group,
        "all_patches": torch.arange(1, 197),
    }

    collected = []
    for start in range(0, len(rows_needed), args.batch_size):
        chunk = rows_needed[start : start + args.batch_size]
        clean_images = torch.stack([clean_set[i][0] for i in chunk]).to(device)
        bd_images = torch.stack(
            [
                bd_set[i][0]
                for i in range(start, min(start + args.batch_size, len(bd_set)))
            ]
        ).to(device)
        if len(bd_images) != len(clean_images):
            break
        clean_cache, clean_logits = cache_stream(model, core, clean_images)
        bd_cache, bd_logits = cache_stream(model, core, bd_images)
        true_label = clean_logits.argmax(dim=1)
        target = torch.full_like(true_label, target_label)
        m_clean = (
            clean_logits.gather(1, target[:, None])
            - clean_logits.gather(1, true_label[:, None])
        ).squeeze(1)
        m_bd = (
            bd_logits.gather(1, target[:, None])
            - bd_logits.gather(1, true_label[:, None])
        ).squeeze(1)
        collected += sweep(
            model,
            core,
            bd_images,
            clean_cache,
            target,
            true_label,
            m_bd,
            m_clean,
            groups,
            args.sites,
        )
        if args.noising:
            collected += [
                {**row, "site": row["site"] + "_noise"}
                for row in sweep(
                    model,
                    core,
                    clean_images,
                    bd_cache,
                    target,
                    true_label,
                    m_clean,
                    m_bd,
                    groups,
                    args.sites,
                )
            ]
        if start + args.batch_size >= args.limit:
            break

    merged = {}
    for row in collected:
        key = (row["site"], row["layer"], row["group"])
        entry = merged.setdefault(key, {"num": 0.0, "den": 0.0, "n": 0})
        entry["num"] += row["numerator"]
        entry["den"] += row["denominator"]
        entry["n"] += row["n"]
    return {
        "folder_name": folder,
        "attack": metadata["attack"],
        "dataset": metadata["dataset"],
        "n_trigger_tokens": int(len(tokens)),
        "rows": [
            {
                "site": site,
                "layer": layer,
                "group": group,
                "recovery": entry["num"] / entry["den"]
                if abs(entry["den"]) > 1e-6
                else float("nan"),
                "n": entry["n"],
            }
            for (site, layer, group), entry in sorted(merged.items())
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-folder", nargs="+", required=True)
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--sites", nargs="+", default=["resid", "attn", "mlp"])
    parser.add_argument("--noising", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for folder in args.checkpoint_folder:
        try:
            report = analyse(folder, args)
        except Exception as error:  # noqa: BLE001
            print(f"[FAILED] {folder}: {type(error).__name__}: {error}", flush=True)
            continue
        out_dir = os.path.join(args.results_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "activation_patching.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(
            f"\n[ok] {folder}  ({report['attack']}, {report['n_trigger_tokens']} trigger tokens)"
        )
        print(
            "     recovery of the clean answer when a site is patched from the clean run"
        )
        groups = sorted({row["group"] for row in report["rows"]})
        for site in sorted({row["site"] for row in report["rows"]}):
            print(f"  site={site}")
            print(f"{'layer':>7s} " + "".join(f"{g:>18s}" for g in groups))
            for layer in range(1, 13):
                cells = {
                    row["group"]: row["recovery"]
                    for row in report["rows"]
                    if row["site"] == site and row["layer"] == layer
                }
                if not cells:
                    continue
                print(
                    f"{layer:7d} "
                    + "".join(f"{cells.get(g, float('nan')):18.3f}" for g in groups)
                )


if __name__ == "__main__":
    main()
