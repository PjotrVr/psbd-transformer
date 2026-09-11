"""Why Swin-S reads higher AUROC than ViT-B/16 under PSBD-TM on global triggers.

Under token masking at the attention input (before_attention_norm, rate chosen by
the 0.8 clean-shift rule), Swin-S reads 0.982 mean AUROC against 0.939 for
ViT-B/16 on 42 matched (dataset, attack, rate) settings. The gap sits entirely in
the global triggers (WaNet +0.17, LC +0.14, SIG +0.54), while local triggers
(BadNet, LF, BPP, Blend) are within 0.04. 4 structural candidates:

    (a) no class token: Swin's readout is a mean over every token of the last
        stage, so masking a token removes its share of the readout directly,
        while ViT's head reads only the class token, which TokenMask always
        protects (defences.operators.TokenMask, models.positions SWIN_POSITIONS)
    (b) windowed attention keeps a trigger's evidence inside its own window for
        the early stages (torchvision's ShiftedWindowAttention, window_size 7),
        so a masked token only enters the attention pattern of its own window
        until the grid downsamples to 1 window at stage 4
    (c) 24 blocks against 12, so the same masking rate acts twice as often
    (d) Swin simply has a firmer shortcut (ASR 0.992 against 0.978 on the
        matched pairs), so there is more margin to erase before the prediction
        moves

Measured on 6 matched (architecture, dataset, attack, rate) pairs, 300 paired
clean and triggered images per checkpoint, split 150 to estimate the backdoor
direction and 150 held out to evaluate everything read against it, mirroring
experiments/whole_network_erasure/measure.py's own train/eval split:

    (1) logit margin of the predicted class over the runner-up, clean and
        triggered, no perturbation (candidate d)
    (2) the backdoor direction's relative norm per block on the pooled feature
        (class token for ViT, mean-pooled tokens for Swin, the same reductions
        analysis.features.default_reduction uses for the 2 architectures) and
        its onset block as a fraction of depth
    (3) the per-token share of the last stage's readout that carries the
        backdoor projection on Swin, against the ViT class token's
        attention-weighted source count at the last block (candidate a)
    (4) Swin only, stages 1 to 3: patching the window with the most trigger
        evidence with its clean values, against patching the same token count
        spread over other windows, read as recovery of the true-label
        prediction (candidate b)
    (5) from the psbd/ caches, the AUROC ladder against the achieved clean
        validation shift ratio and the rate the 0.8 rule selects, at
        before_attention_norm_token_mask and, where swept, the block-banded
        variants of it (candidate c)

    PYTHONPATH=. .venv/bin/python experiments/swin_mechanism/measure.py \
        --checkpoints-dir checkpoints --raw-data-dir raw_data
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from analysis.features import (  # noqa: E402
    as_token_sequence,
    captured_layers,
    default_reduction,
    transformer_blocks,
)
from data.splits import build_psbd_loaders_from_checkpoint, read_checkpoint_metadata  # noqa: E402
from defences.cache import (  # noqa: E402
    baseline_path,
    dropout_pass_path,
    load_baseline,
    load_dropout_pass_probs,
    read_split_manifest,
)
from defences.decision import (  # noqa: E402
    HEADLINE_QUANTILE,
    complete_rates,
    pair_clean_to_backdoor,
    select_rate_adaptively,
)
from defences.scores import psu_ratio_from_cache, shift_ratio  # noqa: E402
from experiments._paths import experiment_result_path  # noqa: E402
from models.backbones import load_checkpoint, network_core  # noqa: E402

SLUG = "swin_mechanism"
NUM_PAIRS = 300
ESTIMATE_PAIRS = 150
WINDOW_SIZE = 7
# The index into a Swin core's .features Sequential where each stage's own
# blocks begin: features[0] is the patch embedding, features[1] stage 1's
# blocks, features[2] the merge into stage 2. Every later stage follows the
# same pattern. Slicing here is both where a stage's input is read off and
# where the patched input re-enters the network for the rest of the forward
# pass.
SWIN_STAGE_ENTRY = {1: 1, 2: 3, 3: 5}
PLACEMENTS_FOR_LADDER = (
    "before_attention_norm_token_mask",
    "before_attention_norm_blocks_5_8_token_mask",
    "before_attention_norm_blocks_9_12_token_mask",
)
MATCHED_PAIRS = (
    ("cifar10_wanet_10", "vit_cifar10_wanet_0_1", "swin_cifar10_wanet_0_1"),
    ("cifar10_sig_10", "vit_cifar10_sig_0_1", "swin_cifar10_sig_0_1"),
    ("cifar100_wanet_10", "vit_cifar100_wanet_0_1", "swin_cifar100_wanet_0_1"),
    ("gtsrb_wanet_10", "vit_gtsrb_wanet_0_1", "swin_gtsrb_wanet_0_1"),
    (
        "cifar100_badnet_10",
        "vit_cifar100_badnet_a2o_0_1",
        "swin_cifar100_badnet_a2o_0_1",
    ),
    ("cifar100_blend_10", "vit_cifar100_blend_0_1", "swin_cifar100_blend_0_1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--raw-data-dir", default="raw_data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--max-samples", type=int, default=2000, help="analysis rows read per split"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="ViT batch size, halved for Swin"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--slugs",
        nargs="+",
        default=None,
        help="restrict to these pair slugs, for a smoke test",
    )
    return parser.parse_args()


def stack_loader(loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Every image and label a shuffle=False loader serves, on the CPU."""
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    stacked = torch.cat(images), torch.cat(labels).long()  # (n, C, H, W), (n,)
    return stacked


def paired_rows(loaders, manifest) -> dict[str, torch.Tensor]:
    """Triggered images with the clean copy and true label of the same source image.

    Same construction as experiments/whole_network_erasure/measure.py's
    paired_rows, minus the whole-clean-set fields that experiment needs and this
    script does not.
    """
    clean_images, clean_labels = stack_loader(loaders["clean"])
    backdoor_images, target_labels = stack_loader(loaders["backdoor"])
    row_of = {
        original: row for row, original in enumerate(manifest["analysis_clean_indices"])
    }
    rows = torch.tensor(
        [row_of[original] for original in manifest["analysis_backdoor_indices"]]
    )  # (n_backdoor,)
    pairs = {
        "clean": clean_images[rows],  # (n_backdoor, C, H, W)
        "backdoor": backdoor_images,  # (n_backdoor, C, H, W)
        "true_label": clean_labels[rows],  # (n_backdoor,)
        "target": target_labels,  # (n_backdoor,)
    }
    return pairs


@torch.inference_mode()
def logits_for(model, images: torch.Tensor, device, batch_size: int) -> torch.Tensor:
    """Every image's logits, (n, num_classes), float32 on the CPU. No autocast."""
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)  # (batch, C, H, W)
        chunks.append(model(batch).float().cpu())  # (batch, num_classes)
    stacked = torch.cat(chunks)  # (n, num_classes)
    return stacked


def margin_of(logits: torch.Tensor) -> torch.Tensor:
    """The predicted class's logit minus the runner-up's, (n,)."""
    top2 = logits.topk(2, dim=1).values  # (n, 2)
    margin = top2[:, 0] - top2[:, 1]  # (n,)
    return margin


@torch.inference_mode()
def pooled_features(
    model,
    architecture: str,
    images: torch.Tensor,
    layers: tuple[int, ...],
    device,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Every layer's pooled feature, {layer: (n, dim)} on the CPU.

    Pooling matches analysis.features.default_reduction: the class token for
    ViT, the mean over tokens for Swin, the same reduction each architecture's
    own head consumes.
    """
    reduction = default_reduction(architecture)
    storage: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)  # (batch, C, H, W)
        with captured_layers(model, layers, architecture) as captured:
            model(batch)
            for layer in layers:
                tokens = as_token_sequence(
                    captured[layer]
                ).float()  # (batch, tokens, dim)
                pooled = tokens[:, 0, :] if reduction == "cls" else tokens.mean(dim=1)
                storage[layer].append(pooled.cpu())  # (batch, dim)
    combined = {
        layer: torch.cat(chunks) for layer, chunks in storage.items()
    }  # (n, dim) each
    return combined


def backdoor_directions(
    model, architecture: str, estimate: dict, device, batch_size: int
) -> dict:
    """The mean triggered-minus-clean pooled feature at every block, and its onset.

    Mirrors experiments/whole_network_erasure/measure.py's directions() and Eq. 1
    reasoning, generalized to Swin's 24 blocks and mean-pooled readout. onset_layer
    is the first block whose direction norm reaches half of the largest norm on
    the ladder, read as a fraction of the architecture's own depth so ViT and
    Swin's onsets are comparable.
    """
    core = network_core(model)
    num_layers = len(transformer_blocks(core, architecture))
    layers = tuple(range(num_layers + 1))

    clean_feats = pooled_features(
        model, architecture, estimate["clean"], layers, device, batch_size
    )
    trig_feats = pooled_features(
        model, architecture, estimate["backdoor"], layers, device, batch_size
    )

    direction = {
        layer: (trig_feats[layer] - clean_feats[layer]).mean(dim=0) for layer in layers
    }  # (dim,) each
    norms = {layer: float(direction[layer].norm()) for layer in layers}
    max_norm = max(norms.values())
    relative = {
        layer: (norms[layer] / max_norm if max_norm > 0 else 0.0) for layer in layers
    }

    onset_layer = next((layer for layer in layers if relative[layer] >= 0.5), None)
    onset_fraction = onset_layer / num_layers if onset_layer is not None else None

    found = {
        "direction": direction,
        "relative_norm": relative,
        "onset_layer": onset_layer,
        "onset_fraction": onset_fraction,
        "num_layers": num_layers,
    }
    return found


@torch.inference_mode()
def swin_token_share(
    model,
    num_layers: int,
    unit_direction: torch.Tensor,
    images: torch.Tensor,
    device,
    batch_size: int,
) -> dict:
    """How many of the last stage's tokens carry half of the backdoor projection.

    unit_direction is the last block's backdoor direction, normalized. Per
    triggered image, each token's raw last-stage activation is dotted with the
    direction and the tokens are sorted by that contribution descending. The
    reported count is the smallest prefix whose cumulative contribution reaches
    half of the image's total. Candidate (a): Swin's readout means over every
    such token, so a small count here means the mask removes a large share of
    what the head reads.
    """
    unit = (unit_direction / unit_direction.norm()).cpu()  # (dim,)
    shares, total_tokens = [], None
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)  # (batch, C, H, W)
        with captured_layers(model, (num_layers,), "swin") as captured:
            model(batch)
            raw = captured[num_layers]  # (batch, height, width, dim)
        tokens = as_token_sequence(raw).float().cpu()  # (batch, tokens, dim)
        total_tokens = tokens.shape[1]

        contribution = tokens @ unit  # (batch, tokens)
        total = contribution.sum(dim=1, keepdim=True)  # (batch, 1)
        sorted_contribution, _ = contribution.sort(
            dim=1, descending=True
        )  # (batch, tokens)
        cumulative = sorted_contribution.cumsum(dim=1)  # (batch, tokens)

        reached = cumulative >= 0.5 * total  # (batch, tokens)
        any_reached = reached.any(dim=1)  # (batch,)
        first_reached = reached.float().argmax(dim=1) + 1  # (batch,), 1-indexed count
        count = torch.where(
            any_reached, first_reached, torch.full_like(first_reached, total_tokens)
        )
        shares.append(count.float() / total_tokens)  # (batch,)

    all_shares = torch.cat(shares)  # (n,)
    result = {
        "mean_share": float(all_shares.mean()),
        "mean_tokens": float(all_shares.mean() * total_tokens),
        "total_tokens": int(total_tokens),
    }
    return result


@torch.inference_mode()
def vit_token_share(model, core, images: torch.Tensor, device, batch_size: int) -> dict:
    """How many tokens the class token's last-block attention concentrates half its mass on.

    The ViT-side reading for candidate (a): the class token is the only token
    the head reads, so this asks how spread out the evidence it attends to is,
    the transformer-native analogue of swin_token_share's readout share.
    """
    last_block = transformer_blocks(core, "vit")[-1]
    shares, total_tokens = [], None
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)  # (batch, C, H, W)

        captured_input = {}

        def pre_hook(_module, args):
            captured_input["x"] = args[0]
            return args

        handle = last_block.self_attention.register_forward_pre_hook(pre_hook)
        model(batch)
        handle.remove()

        x = captured_input["x"]  # (batch, tokens, dim), ln_1's output feeding attention
        _, attn_weights = last_block.self_attention(
            x, x, x, need_weights=True, average_attn_weights=True
        )  # attn_weights: (batch, tokens, tokens)
        cls_row = attn_weights[:, 0, :].float().cpu()  # (batch, tokens)
        total_tokens = cls_row.shape[1]

        sorted_weights, _ = cls_row.sort(dim=1, descending=True)  # (batch, tokens)
        cumulative = sorted_weights.cumsum(dim=1)  # (batch, tokens)
        first_reached = (cumulative >= 0.5).float().argmax(dim=1) + 1  # (batch,)
        shares.append(first_reached.float() / total_tokens)  # (batch,)

    all_shares = torch.cat(shares)  # (n,)
    result = {
        "mean_share": float(all_shares.mean()),
        "mean_tokens": float(all_shares.mean() * total_tokens),
        "total_tokens": int(total_tokens),
    }
    return result


@torch.inference_mode()
def stage_entry_activations(
    core,
    resize,
    clean: torch.Tensor,
    backdoor: torch.Tensor,
    stage_index: int,
    device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (n, height, width, dim) activation entering a Swin stage, clean and triggered."""
    clean_chunks, trig_chunks = [], []
    for start in range(0, len(clean), batch_size):
        clean_batch = resize(
            clean[start : start + batch_size].to(device)
        )  # (batch, C, 224, 224)
        trig_batch = resize(backdoor[start : start + batch_size].to(device))
        clean_chunks.append(core.features[:stage_index](clean_batch).cpu())
        trig_chunks.append(core.features[:stage_index](trig_batch).cpu())
    clean_stage = torch.cat(clean_chunks)  # (n, height, width, dim)
    trig_stage = torch.cat(trig_chunks)
    return clean_stage, trig_stage


@torch.inference_mode()
def rest_of_network(
    core, patched: torch.Tensor, stage_index: int, device, batch_size: int
) -> torch.Tensor:
    """Logits from a (n, height, width, dim) activation carried through the rest of Swin."""
    chunks = []
    for start in range(0, len(patched), batch_size):
        batch = patched[start : start + batch_size].to(
            device
        )  # (batch, height, width, dim)
        x = core.features[stage_index:](batch)
        x = core.norm(x)
        x = core.permute(x)
        x = core.avgpool(x)
        x = core.flatten(x)
        x = core.head(x)  # (batch, num_classes)
        chunks.append(x.float().cpu())
    stacked = torch.cat(chunks)  # (n, num_classes)
    return stacked


def top_evidence_window(
    clean_stage: torch.Tensor, trig_stage: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Per sample, the (row, col) window index with the largest clean-vs-triggered difference."""
    n, height, width, dim = trig_stage.shape
    rows, cols = height // window, width // window
    diff = trig_stage - clean_stage  # (n, height, width, dim)
    evidence = (
        diff.reshape(n, rows, window, cols, window, dim).abs().sum(dim=(2, 4, 5))
    )  # (n, rows, cols)
    flat_index = evidence.reshape(n, -1).argmax(dim=1)  # (n,)
    top_row = flat_index // cols  # (n,)
    top_col = flat_index % cols  # (n,)
    return top_row, top_col, rows, cols


def patch_windows(
    clean_stage: torch.Tensor,
    trig_stage: torch.Tensor,
    top_row: torch.Tensor,
    top_col: torch.Tensor,
    rows: int,
    cols: int,
    window: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triggered activations with clean values patched in, at the top-evidence window or spread over an equal token count elsewhere."""
    patched_top = trig_stage.clone()
    patched_control = trig_stage.clone()
    for sample in range(trig_stage.shape[0]):
        row, col = int(top_row[sample]), int(top_col[sample])
        patched_top[
            sample,
            row * window : (row + 1) * window,
            col * window : (col + 1) * window,
            :,
        ] = clean_stage[
            sample,
            row * window : (row + 1) * window,
            col * window : (col + 1) * window,
            :,
        ]

        other_positions = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if not (r == row and c == col)
        ]
        picks = torch.randperm(len(other_positions), generator=generator)[
            : window * window
        ]
        for pick in picks:
            r, c = other_positions[int(pick)]
            patched_control[sample, r, c, :] = clean_stage[sample, r, c, :]

    return patched_top, patched_control


def window_patch_test(
    model, core, eval_pairs: dict, device, batch_size: int, seed: int
) -> dict:
    """Recovery of the true-label prediction from patching 1 window against a spread control.

    Candidate (b): if windowed attention keeps a trigger's evidence inside 1
    window at the early stages, patching that window alone should recover much
    more of the clean answer than patching the same number of tokens spread
    across the other windows.
    """
    resize = model[0]
    generator = torch.Generator().manual_seed(seed)
    true_label = eval_pairs["true_label"]

    results = {}
    for stage, stage_index in SWIN_STAGE_ENTRY.items():
        clean_stage, trig_stage = stage_entry_activations(
            core,
            resize,
            eval_pairs["clean"],
            eval_pairs["backdoor"],
            stage_index,
            device,
            batch_size,
        )
        top_row, top_col, rows, cols = top_evidence_window(
            clean_stage, trig_stage, WINDOW_SIZE
        )
        patched_top, patched_control = patch_windows(
            clean_stage,
            trig_stage,
            top_row,
            top_col,
            rows,
            cols,
            WINDOW_SIZE,
            generator,
        )

        logits_top = rest_of_network(core, patched_top, stage_index, device, batch_size)
        logits_control = rest_of_network(
            core, patched_control, stage_index, device, batch_size
        )

        results[f"stage_{stage}"] = {
            "grid": [trig_stage.shape[1], trig_stage.shape[2]],
            "windows": [rows, cols],
            "window_tokens": WINDOW_SIZE * WINDOW_SIZE,
            "ra_top_window_patch": float(
                (logits_top.argmax(dim=1) == true_label).float().mean()
            ),
            "ra_control_patch": float(
                (logits_control.argmax(dim=1) == true_label).float().mean()
            ),
        }

    return results


def auroc_ladder(psbd_dir: str, placement: str) -> dict | None:
    """The swept AUROC against clean-validation shift ratio for 1 placement, from disk.

    None when the placement was never swept for this checkpoint, which is
    candidate (c)'s answer for every Swin folder: the block-banded variants of
    before_attention_norm exist only on the ViT side of the panel.
    """
    if not os.path.isdir(os.path.join(psbd_dir, placement)):
        return None

    manifest = read_split_manifest(psbd_dir)
    val_probs, val_labels, _ = load_baseline(baseline_path(psbd_dir, "validation"))
    clean_probs, clean_labels, _ = load_baseline(baseline_path(psbd_dir, "clean"))
    backdoor_probs, backdoor_labels, _ = load_baseline(
        baseline_path(psbd_dir, "backdoor")
    )

    ladder, shift_by_rate = [], {}
    for rate in complete_rates(psbd_dir, placement):
        val_pass_probs, val_pass_argmax = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "validation")
        )
        clean_pass_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "clean")
        )
        backdoor_pass_probs, _ = load_dropout_pass_probs(
            dropout_pass_path(psbd_dir, placement, rate, "backdoor")
        )

        val_psu = psu_ratio_from_cache(val_probs, val_labels, val_pass_probs)
        clean_psu = psu_ratio_from_cache(clean_probs, clean_labels, clean_pass_probs)
        backdoor_psu = psu_ratio_from_cache(
            backdoor_probs, backdoor_labels, backdoor_pass_probs
        )
        clean_psu_paired = pair_clean_to_backdoor(clean_psu, manifest)

        auroc = detection_report_auroc(val_psu, clean_psu_paired, backdoor_psu)
        sigma = shift_ratio(val_labels, val_pass_argmax)
        shift_by_rate[rate] = sigma
        ladder.append({"rate": rate, "shift_ratio": sigma, "auroc": auroc})

    adaptive_rate = select_rate_adaptively(shift_by_rate)
    adaptive_row = next((row for row in ladder if row["rate"] == adaptive_rate), None)
    found = {
        "ladder": ladder,
        "adaptive_rate": adaptive_rate,
        "adaptive_auroc": adaptive_row["auroc"] if adaptive_row else None,
        "adaptive_shift_ratio": adaptive_row["shift_ratio"] if adaptive_row else None,
    }
    return found


def detection_report_auroc(
    val_psu: torch.Tensor, clean_psu: torch.Tensor, backdoor_psu: torch.Tensor
) -> float:
    """AUROC only, at fractional PSU, the canon headline statistic (defences.decision.detection_report)."""
    from defences.decision import detection_report

    report = detection_report(val_psu, clean_psu, backdoor_psu, HEADLINE_QUANTILE)
    auroc = report["auroc"]
    return auroc


def analyse_model(
    folder: str,
    expected_architecture: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    checkpoint = os.path.join(args.checkpoints_dir, folder, "attack_result.pt")
    metadata = read_checkpoint_metadata(checkpoint)
    if metadata["architecture"] != expected_architecture:
        raise ValueError(
            f"{folder} is {metadata['architecture']}, expected {expected_architecture}"
        )

    loaders, manifest = build_psbd_loaders_from_checkpoint(
        checkpoint,
        raw_data_dir=args.raw_data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        max_samples=args.max_samples,
    )
    model = load_checkpoint(metadata["architecture"], checkpoint, device)
    core = network_core(model)
    architecture = metadata["architecture"]
    batch_size = args.batch_size // 2 if architecture == "swin" else args.batch_size

    pairs = paired_rows(loaders, manifest)
    n_pairs = min(NUM_PAIRS, pairs["clean"].shape[0])
    n_estimate = min(ESTIMATE_PAIRS, n_pairs // 2)
    estimate = {key: value[:n_estimate] for key, value in pairs.items()}
    evaluate = {key: value[n_estimate:n_pairs] for key, value in pairs.items()}

    clean_logits = logits_for(model, evaluate["clean"], device, batch_size)
    trig_logits = logits_for(model, evaluate["backdoor"], device, batch_size)
    clean_margin = margin_of(clean_logits)
    trig_margin = margin_of(trig_logits)
    predicted_triggered = trig_logits.argmax(dim=1)

    directions = backdoor_directions(model, architecture, estimate, device, batch_size)
    last_layer = directions["num_layers"]

    if architecture == "swin":
        token_share = swin_token_share(
            model,
            last_layer,
            directions["direction"][last_layer],
            evaluate["backdoor"],
            device,
            batch_size,
        )
        window_results = window_patch_test(
            model, core, evaluate, device, batch_size, args.seed
        )
    else:
        token_share = vit_token_share(
            model, core, evaluate["backdoor"], device, batch_size
        )
        window_results = None

    psbd_dir = os.path.join(args.results_dir, folder, "psbd")
    ladders = {
        placement: auroc_ladder(psbd_dir, placement)
        for placement in PLACEMENTS_FOR_LADDER
    }

    report = {
        "folder": folder,
        "architecture": architecture,
        "dataset": metadata["dataset"],
        "attack": metadata["attack"],
        "poison_rate": metadata.get("poison_rate"),
        "n_pairs": n_pairs,
        "n_estimate": n_estimate,
        "n_eval": n_pairs - n_estimate,
        "margin_clean": float(clean_margin.mean()),
        "margin_triggered": float(trig_margin.mean()),
        "margin_gap": float(trig_margin.mean() - clean_margin.mean()),
        "eval_ra_triggered": float(
            (predicted_triggered == evaluate["true_label"]).float().mean()
        ),
        "eval_asr_triggered": float(
            (predicted_triggered == evaluate["target"]).float().mean()
        ),
        "num_layers": directions["num_layers"],
        "direction_relative_norm": [
            directions["relative_norm"][layer]
            for layer in range(directions["num_layers"] + 1)
        ],
        "direction_onset_layer": directions["onset_layer"],
        "direction_onset_fraction": directions["onset_fraction"],
        "last_layer_token_share": token_share,
        "window_patch": window_results,
        "auroc_ladders": ladders,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return report


def summary_line(pair_slug: str, vit_report: dict, swin_report: dict) -> str:
    """1 line comparing the pair's margin gap, direction onset and token share."""
    line = (
        f"{pair_slug:20s} margin gap vit {vit_report['margin_gap']:+.3f} swin {swin_report['margin_gap']:+.3f} | "
        f"onset frac vit {vit_report['direction_onset_fraction']} swin {swin_report['direction_onset_fraction']} | "
        f"token share vit {vit_report['last_layer_token_share']['mean_share']:.3f} "
        f"swin {swin_report['last_layer_token_share']['mean_share']:.3f} | "
        f"ladder auroc vit {vit_report['auroc_ladders']['before_attention_norm_token_mask']['adaptive_auroc']} "
        f"swin {swin_report['auroc_ladders']['before_attention_norm_token_mask']['adaptive_auroc']}"
    )
    return line


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = [
        pair for pair in MATCHED_PAIRS if args.slugs is None or pair[0] in args.slugs
    ]
    for slug, vit_folder, swin_folder in pairs:
        vit_report = analyse_model(vit_folder, "vit", args, device)
        swin_report = analyse_model(swin_folder, "swin", args, device)

        combined = {"slug": slug, "vit": vit_report, "swin": swin_report}
        path = experiment_result_path(SLUG, f"{slug}.json", args.results_dir)
        with open(path, "w") as handle:
            json.dump(combined, handle, indent=2)

        print(summary_line(slug, vit_report, swin_report), flush=True)


if __name__ == "__main__":
    main()
