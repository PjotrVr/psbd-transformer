"""Protocol constants as macros, read from the code and the declaration, never typed.

A chapter may not carry a bare digit, and the protocol is made of digits: the
shift-ratio targets, the headline quantile, the held-out split size, the pass
count, the attack success bar, the epoch budget, the architecture's block count.
Every 1 of them lives in exactly 1 place in the code or the declaration, and this
generator reads that place so a chapter that names the constant cannot drift from
the constant the sweep used.

    PYTHONPATH=. python scripts/paper/app_protocol.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from torchvision.models import swin_s, vit_b_16  # noqa: E402
from torchvision.models.swin_transformer import SwinTransformerBlock  # noqa: E402

from data.splits import (  # noqa: E402
    BENIGN_PROBE_ATTACK,
    BENIGN_PROBE_TARGET_LABEL,
    PSBD_HELDOUT_SIZE,
    PSBD_SPLIT_SEED,
)
from defences.decision import (  # noqa: E402
    ADAPTIVE_SHIFT_TARGET,
    HEADLINE_QUANTILE,
    PLACEMENT_MATCH_TARGET,
    PSBD_QUANTILES,
    PUBLISHED_PLACEMENT,
    RECOMMENDED_PLACEMENT,
    SHIFT_MATCH_TARGETS,
)
from data.registry import DATASET_REGISTRY  # noqa: E402
from defences.operators import OPERATORS  # noqa: E402
from detectors import DETECTOR_NAMES  # noqa: E402
from models.positions import (  # noqa: E402
    PORTED_POSITION_NAMES,
    SINGLE_POSITION_NAMES,
    STRUCTURED_POSITION_NAMES,
    VIT_POSITIONS,
)
from scripts.paper._common import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    build_parser_with_checkpoints,
    load_args_json,
    load_declaration,
    write_macros,
)

GENERATOR = "scripts/paper/app_protocol.py"


def architecture_sizes() -> dict[str, int]:
    """Block count, token count and width of the 2 backbones, read off torchvision.

    Both models are built without weights, so nothing is downloaded and the
    numbers come from the architecture definition rather than from a checkpoint.
    """
    vit = vit_b_16(weights=None)
    swin = swin_s(weights=None)
    swin_blocks = sum(
        1 for module in swin.modules() if isinstance(module, SwinTransformerBlock)
    )
    sizes = {
        "vit_blocks": len(vit.encoder.layers),
        "vit_tokens": vit.seq_length,
        "vit_width": vit.hidden_dim,
        "vit_heads": vit.encoder.layers[0].self_attention.num_heads,
        "swin_blocks": swin_blocks,
    }
    return sizes


def training_epochs(checkpoints_dir: str, benign_folders: dict[str, str]) -> int:
    """The epoch budget every benign reference recorded, which must be 1 number."""
    epochs = {
        dataset: (load_args_json(checkpoints_dir, folder) or {}).get("epochs")
        for dataset, folder in benign_folders.items()
    }
    distinct = {value for value in epochs.values() if value is not None}
    if len(distinct) != 1:
        raise SystemExit(f"benign references disagree on epochs: {epochs}")
    budget = distinct.pop()
    return budget


def percent(rate: float) -> str:
    text = f"{rate * 100:g}\\%"
    return text


def main() -> None:
    args = build_parser_with_checkpoints(__doc__).parse_args()
    declaration = load_declaration(args.declaration)
    panel = declaration["panel"]
    sizes = architecture_sizes()
    epochs = training_epochs(args.checkpoints_dir, declaration["benign_reference"])
    recommended = next(
        entry for entry in declaration["basis"] if entry["id"] == RECOMMENDED_PLACEMENT
    )
    rates = sorted(panel["poison_rates"])

    macros = {
        "adaptive_shift_target": (
            f"{ADAPTIVE_SHIFT_TARGET:g}",
            "the clean-validation shift ratio PSBD's adaptive rate rule reaches for",
        ),
        "placement_match_target": (
            f"{PLACEMENT_MATCH_TARGET:g}",
            "the clean-validation shift ratio the cross-placement comparison matches at",
        ),
        "shift_match_targets": (
            ", ".join(f"{target:g}" for target in SHIFT_MATCH_TARGETS),
            "the ladder of matched shift ratios every placement is read at",
        ),
        "headline_quantile": (
            f"{HEADLINE_QUANTILE:g}",
            "the clean-validation PSU quantile the headline threshold sits at",
        ),
        "headline_quantile_percent": (
            percent(HEADLINE_QUANTILE),
            "the headline quantile as a false-positive budget",
        ),
        "psbd_quantiles": (
            ", ".join(f"{quantile:g}" for quantile in PSBD_QUANTILES),
            "every quantile swept beside the headline quantile",
        ),
        "lowest_quantile_percent": (
            percent(min(PSBD_QUANTILES)),
            "the smallest false-positive budget swept",
        ),
        "heldout_size": (
            str(PSBD_HELDOUT_SIZE),
            "clean validation images held out for the threshold",
        ),
        "split_seed": (str(PSBD_SPLIT_SEED), "the seed of the held-out split"),
        "forward_passes": (
            str(panel["forward_passes"]),
            "perturbed forward passes per input, k, on the basis panel",
        ),
        "mask_seed": (str(panel["mask_seed"]), "the perturbation mask seed"),
        "asr_bar": (
            f"{declaration['asr_bar']:g}",
            "attack success rate a cell must reach to enter a detection table",
        ),
        "clean_drop_bar": (
            f"{declaration['clean_accuracy_drop_bar']:+g}",
            "the clean-accuracy drop a cell may show against its benign reference",
        ),
        "panel_rates": (
            ", ".join(percent(rate) for rate in rates),
            "the poison rates the panel declares",
        ),
        "panel_rate_low": (percent(rates[0]), "the lowest panel poison rate"),
        "panel_rate_mid": (percent(rates[1]), "the middle panel poison rate"),
        "panel_rate_high": (percent(rates[-1]), "the highest panel poison rate"),
        "basis_size": (
            str(len(declaration["basis"])),
            "placements the basis declares for every panel cell",
        ),
        "recommended_rate_ladder_low": (
            f"{min(recommended['rates']):g}",
            "the smallest rate on the recommended placement's ladder",
        ),
        "recommended_rate_ladder_high": (
            f"{max(recommended['rates']):g}",
            "the largest rate on the recommended placement's ladder",
        ),
        "recommended_rate_ladder_size": (
            str(len(recommended["rates"])),
            "rates on the recommended placement's ladder",
        ),
        "recommended_placement": (
            RECOMMENDED_PLACEMENT.replace("_", r"\_"),
            "the recommended placement id",
        ),
        "published_placement": (
            PUBLISHED_PLACEMENT.replace("_", r"\_"),
            "the published ConvNet placement id",
        ),
        "training_epochs": (str(epochs), "the uniform training budget in epochs"),
        "vit_blocks": (str(sizes["vit_blocks"]), "encoder blocks in ViT-B/16"),
        "vit_tokens": (
            str(sizes["vit_tokens"]),
            "tokens per image in ViT-B/16, patches plus the class token",
        ),
        "vit_patches": (
            str(sizes["vit_tokens"] - 1),
            "patch tokens per image in ViT-B/16",
        ),
        "vit_width": (str(sizes["vit_width"]), "residual width of ViT-B/16"),
        "vit_heads": (str(sizes["vit_heads"]), "attention heads per block in ViT-B/16"),
        "vit_total_heads": (
            str(sizes["vit_heads"] * sizes["vit_blocks"]),
            "attention heads in the whole ViT-B/16 stack",
        ),
        "swin_blocks": (str(sizes["swin_blocks"]), "transformer blocks in Swin-S"),
        "position_count": (
            str(len(VIT_POSITIONS)),
            "named probe positions in the ViT registry",
        ),
        "single_position_count": (
            str(len(SINGLE_POSITION_NAMES)),
            "atomic positions swept in isolation",
        ),
        "structured_position_count": (
            str(len(STRUCTURED_POSITION_NAMES)),
            "positions whose channel axis indexes a transformer unit",
        ),
        "ported_position_count": (
            str(len(PORTED_POSITION_NAMES)),
            "positions that exist to host ported detectors",
        ),
        "operator_count": (
            str(len(OPERATORS) + 1),
            "perturbation operators in the registry, scale_up included",
        ),
        "panel_datasets_declared": (
            str(len([key for key in declaration["benign_reference"] if not key.startswith("_")])),
            "datasets with a benign reference in the declaration",
        ),
        "detector_count": (
            str(len(DETECTOR_NAMES)),
            "competitor detectors in the registry",
        ),
        "bootstrap_resamples": (
            str(BOOTSTRAP_RESAMPLES),
            "resamples behind every bootstrap interval",
        ),
        "bootstrap_seed": (str(BOOTSTRAP_SEED), "the bootstrap seed"),
        "benign_probe_attack": (
            BENIGN_PROBE_ATTACK.replace("_", r"\_"),
            "the trigger a benign reference is probed with",
        ),
        "benign_probe_target": (
            str(BENIGN_PROBE_TARGET_LABEL),
            "the target class a benign reference is probed at",
        ),
        "panel_target_label": (
            str(BENIGN_PROBE_TARGET_LABEL),
            "the target class every panel cell attacks",
        ),
    }
    # The depth bands the basis declares, as "first to last" block ranges, and the
    # class count of every registered dataset, so a chapter names neither by hand.
    bands = sorted(
        {tuple(entry["block_range"]) for entry in declaration["basis"] if entry.get("block_range")}
    )
    for label, band in zip(("early", "middle", "late"), bands):
        macros[f"band_{label}"] = (f"{band[0]} to {band[1]}", f"the {label} depth band's block range")
        macros[f"band_{label}_first"] = (str(band[0]), f"first block of the {label} depth band")
        macros[f"band_{label}_last"] = (str(band[1]), f"last block of the {label} depth band")
    for dataset, spec in DATASET_REGISTRY.items():
        macros[f"classes_{dataset}"] = (str(spec.num_classes), f"classes in {dataset}")
    write_macros(
        os.path.join(args.paper_dir, "tables", "protocol.macros.json"),
        GENERATOR,
        [
            "defences/decision.py",
            "defences/operators.py",
            "models/positions.py",
            "data/splits.py",
            "detectors/__init__.py",
            args.declaration,
            f"{args.checkpoints_dir}/<benign reference>/args.json",
        ],
        macros,
    )
    print(f"protocol: {len(macros)} macros, epochs {epochs}, sizes {sizes}")


if __name__ == "__main__":
    main()
