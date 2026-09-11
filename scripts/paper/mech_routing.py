"""Routing and sinks: what attention does with the trigger, by layer and by attack.

4 per-checkpoint records, each consolidated per attack with the benign references
as the control. cls_routing.json gives the class token's attention mass on the
trigger's tokens by layer. logit_attribution.json splits each layer's write of
the backdoor direction between attention and the MLP. artifact_tokens.json gives
the trigger tokens' norm rank among patches and the max-over-median patch-norm
ratio. sink_anatomy.json gives the takeover fraction, how much of the class
token's attention mass is restored by other tokens when the trigger's are masked.
The routing records exist at the highest panel rate only, the artifact and sink
records on every clearing cell.

    PYTHONPATH=. python scripts/paper/mech_routing.py \\
        --results-dir /path/to/results --paper-dir paper
"""

import glob
import os
import sys

sys.path.insert(0, os.getcwd())

import scripts.paper._style  # noqa: E402,F401  the shared figure style
import matplotlib.pyplot as plt  # noqa: E402

from defences.decision import EASY_ATTACKS, HARD_ATTACKS  # noqa: E402
from scripts.paper._common import (  # noqa: E402
    OKABE_ITO,
    attack_label,
    build_parser,
    clearing_cells,
    figure_sidecar,
    fmt,
    is_panel_folder,
    load_coverage,
    load_declaration,
    load_json,
    mean_or_none,
    save_figure,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/mech_routing.py"
LOCAL_ATTACKS = ("badnet_a2o", "lc", "tact")
EARLY_LAYERS = (1, 2, 3, 4)
ROUTING_LAYERS = (5, 6, 7, 8)
LATE_LAYERS = (9, 10, 11, 12)
# The trigger's tokens are visible in raw pixels before the network has done
# anything with them, so the benign control reads high at the first 2 layers and
# the token-norm statistics are read from layer 3 on.
NORM_LAYERS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
# A trigger occupying at least this share of the patches has no token set to
# rank or to mask against the rest, so the token-level columns print as absent.
GLOBAL_TOKEN_SHARE = 0.5


def trigger_token_share(records: list[dict]) -> float | None:
    """The trigger's share of the patch tokens, read off the first record's layers."""
    for record in records:
        rows = record.get("layers", [])
        if rows and rows[0].get("n_trigger_tokens") is not None:
            share = rows[0]["n_trigger_tokens"] / 196
            return share
    return None


def ordered_attacks(names: set[str]) -> list[str]:
    ranked = list(HARD_ATTACKS) + list(EASY_ATTACKS)
    ordered = [name for name in ranked if name in names] + sorted(
        name for name in names if name not in ranked
    )
    return ordered


def records_by_attack(
    results_dir: str, name: str, folders: list[str]
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for folder in folders:
        record = load_json(os.path.join(results_dir, folder, name))
        if record is None:
            continue
        attack = (
            "benign"
            if "benign" in folder
            else record.get("attack", folder.split("_")[2])
        )
        grouped.setdefault(attack, []).append(record)
    return grouped


def layer_mean(records: list[dict], key: str, layers: tuple[int, ...]) -> float | None:
    values = [
        row[key]
        for record in records
        for row in record["layers"]
        if row["layer"] in layers and row.get(key) is not None
    ]
    mean = mean_or_none(values)
    return mean


def layer_curve(records: list[dict], key: str) -> list[float | None]:
    layers = sorted({row["layer"] for record in records for row in record["layers"]})
    curve = [layer_mean(records, key, (layer,)) for layer in layers]
    return curve


def all_folders(results_dir: str, name: str) -> list[str]:
    """Every ViT folder outside the SAM, evasion and seed sets carrying this record."""
    folders = sorted(
        os.path.basename(os.path.dirname(path))
        for path in glob.glob(os.path.join(results_dir, "vit_*", name))
        if is_panel_folder(os.path.basename(os.path.dirname(path)))
    )
    return folders


def main() -> None:
    args = build_parser(__doc__).parse_args()
    coverage_path = os.path.join(args.results_dir, "coverage", "coverage.json")
    declaration = load_declaration(args.declaration)
    clearing = [
        cell["folder_name"] for cell in clearing_cells(load_coverage(args.results_dir))
    ]
    benign = list(declaration["benign_reference"].values())

    routing = records_by_attack(
        args.results_dir,
        "cls_routing.json",
        all_folders(args.results_dir, "cls_routing.json"),
    )
    attribution = records_by_attack(
        args.results_dir,
        "logit_attribution.json",
        all_folders(args.results_dir, "logit_attribution.json"),
    )
    artifacts = records_by_attack(
        args.results_dir, "artifact_tokens.json", clearing + benign
    )
    sinks = records_by_attack(args.results_dir, "sink_anatomy.json", clearing + benign)
    inputs = [
        coverage_path,
        f"{args.results_dir}/<folder>/cls_routing.json ({sum(map(len, routing.values()))})",
        f"{args.results_dir}/<folder>/logit_attribution.json ({sum(map(len, attribution.values()))})",
        f"{args.results_dir}/<folder>/artifact_tokens.json ({sum(map(len, artifacts.values()))})",
        f"{args.results_dir}/<folder>/sink_anatomy.json ({sum(map(len, sinks.values()))})",
    ]

    # Figure: class-token attention on the trigger tokens by layer, triggered images,
    # 1 line per attack, the benign references as a grey band.
    attacks = ordered_attacks(set(routing) - {"benign"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    plotted = {"cls_attention_on_trigger": {}, "trigger_norm_rank": {}}
    for index, attack in enumerate(attacks):
        curve = layer_curve(routing[attack], "weight_backdoor")
        plotted["cls_attention_on_trigger"][attack] = curve
        left.plot(
            range(1, len(curve) + 1),
            curve,
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=f"{attack_label(attack)} (n={len(routing[attack])})",
        )
    if "benign" in routing:
        curve = layer_curve(routing["benign"], "weight_backdoor")
        plotted["cls_attention_on_trigger"]["benign"] = curve
        left.plot(
            range(1, len(curve) + 1),
            curve,
            color="0.5",
            linestyle="--",
            linewidth=1.2,
            label="benign, same trigger",
        )
    left.set_xlabel("layer")
    left.set_ylabel("CLS attention mass on the trigger tokens")
    left.legend(fontsize=6.5)
    artifact_attacks = ordered_attacks(set(artifacts) - {"benign"})
    for index, attack in enumerate(artifact_attacks):
        curve = layer_curve(artifacts[attack], "trigger_rank_triggered")
        plotted["trigger_norm_rank"][attack] = curve
        right.plot(
            range(1, len(curve) + 1),
            curve,
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=OKABE_ITO[index % len(OKABE_ITO)],
            label=f"{attack_label(attack)} (n={len(artifacts[attack])})",
        )
    if "benign" in artifacts:
        curve = layer_curve(artifacts["benign"], "trigger_rank_triggered")
        plotted["trigger_norm_rank"]["benign"] = curve
        right.plot(
            range(1, len(curve) + 1),
            curve,
            color="0.5",
            linestyle="--",
            linewidth=1.2,
            label="benign, same trigger",
        )
    right.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
    right.set_xlabel("layer")
    right.set_ylabel("norm rank of the trigger tokens among patches")
    right.legend(fontsize=6.5)
    figure_path = os.path.join(args.paper_dir, "figures", "mech_routing.pdf")
    save_figure(fig, figure_path)
    figure_sidecar(figure_path.replace(".pdf", ".json"), GENERATOR, inputs, plotted)

    # Table: per attack, the routing-phase numbers and the sink numbers.
    rows = []
    stats = {}
    for attack in ordered_attacks((set(routing) | set(artifacts)) - {"benign"}) + [
        "benign"
    ]:
        route = routing.get(attack, [])
        attr = attribution.get(attack, [])
        art = artifacts.get(attack, [])
        sink = sinks.get(attack, [])
        attention_write = (
            sum(
                value
                for value in (
                    layer_mean(attr, "attention_write_delta", (layer,))
                    for layer in range(1, 13)
                )
                if value is not None
            )
            if attr
            else None
        )
        mlp_write = (
            sum(
                value
                for value in (
                    layer_mean(attr, "mlp_write_delta", (layer,))
                    for layer in range(1, 13)
                )
                if value is not None
            )
            if attr
            else None
        )
        share = trigger_token_share(art)
        token_level = share is not None and share < GLOBAL_TOKEN_SHARE
        stats[attack] = {
            "cls_early": layer_mean(route, "weight_backdoor", EARLY_LAYERS),
            "cls_routing": layer_mean(route, "weight_backdoor", ROUTING_LAYERS),
            "cls_late": layer_mean(route, "weight_backdoor", LATE_LAYERS),
            "attention_share": (attention_write / (attention_write + mlp_write))
            if attention_write is not None
            and mlp_write is not None
            and (attention_write + mlp_write)
            else None,
            "rank_routing": layer_mean(art, "trigger_rank_triggered", ROUTING_LAYERS)
            if token_level
            else None,
            "rank_late": layer_mean(art, "trigger_rank_triggered", LATE_LAYERS)
            if token_level
            else None,
            "ratio_peak": max(
                (
                    row["ratio_triggered"]
                    for record in art
                    for row in record["layers"]
                    if row["layer"] in NORM_LAYERS
                ),
                default=None,
            ),
            "ratio_clean_peak": max(
                (
                    row["ratio_clean"]
                    for record in art
                    for row in record["layers"]
                    if row["layer"] in NORM_LAYERS
                ),
                default=None,
            ),
            "takeover_routing": layer_mean(sink, "takeover_fraction", ROUTING_LAYERS)
            if token_level
            else None,
            "token_share": share,
        }
        s = stats[attack]
        rows.append(
            [
                attack_label(attack),
                "control"
                if attack == "benign"
                else ("local" if attack in LOCAL_ATTACKS else "global"),
                f"{len(route)}/{len(art)}",
                fmt(s["cls_early"]),
                fmt(s["cls_routing"]),
                fmt(s["cls_late"]),
                fmt(s["attention_share"], places=2),
                fmt(s["rank_routing"], places=2),
                fmt(s["rank_late"], places=2),
                fmt(s["ratio_peak"], places=2),
                fmt(s["takeover_routing"], places=2),
            ]
        )
    write_table(
        path=os.path.join(args.paper_dir, "tables", "routing.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "Routing and sink anatomy per attack. CLS attention mass on the trigger "
            "tokens of triggered images, averaged over the early, routing and late "
            "layer bands (cls\\_routing.json, highest panel rate). Attention's share of "
            "the summed target-logit attribution of the 2 sublayers' writes "
            "(logit\\_attribution.json). The trigger tokens' norm rank among patches "
            "over the routing and late bands, the peak max-over-median patch-norm "
            "ratio from layer 3 on (artifact\\_tokens.json, every model whose attack "
            "succeeded), and the takeover fraction over the routing band, how much "
            "attention mass other tokens absorb when the trigger's are masked "
            "(sink\\_anatomy.json). A global trigger covers every patch, so its "
            "token-level columns are absent. n is routing/artifact models."
        ),
        label="tab:routing",
        header=[
            "attack",
            "trigger",
            "n",
            "CLS mass 1-4",
            "CLS mass 5-8",
            "CLS mass 9-12",
            "attn share",
            "rank 5-8",
            "rank 9-12",
            "peak ratio",
            "takeover 5-8",
        ],
        rows=rows,
        align="llrrrrrrrrr",
    )

    local = [stats[a] for a in stats if a in LOCAL_ATTACKS]
    global_ = [stats[a] for a in stats if a not in LOCAL_ATTACKS and a != "benign"]
    badnet = stats.get("badnet_a2o", {})
    macros = {
        "routing_cells": (
            str(sum(map(len, routing.values()))),
            "checkpoints with a routing record, benign included",
        ),
        "routing_badnet_cls_early": (
            fmt(badnet.get("cls_early")),
            "badnet_a2o CLS attention mass on the trigger tokens, layers 1 to 4",
        ),
        "routing_badnet_cls_routing": (
            fmt(badnet.get("cls_routing")),
            "badnet_a2o CLS attention mass on the trigger tokens, layers 5 to 8",
        ),
        "routing_badnet_cls_late": (
            fmt(badnet.get("cls_late")),
            "badnet_a2o CLS attention mass on the trigger tokens, layers 9 to 12",
        ),
        "routing_benign_cls_late": (
            fmt(stats.get("benign", {}).get("cls_late")),
            "benign CLS attention mass on the same trigger tokens, layers 9 to 12",
        ),
        "routing_badnet_attention_share": (
            fmt(badnet.get("attention_share"), places=2),
            "badnet_a2o attention share of the summed target-logit attribution",
        ),
        "routing_global_attention_share": (
            fmt(
                mean_or_none(
                    [
                        s["attention_share"]
                        for s in global_
                        if s["attention_share"] is not None
                    ]
                ),
                places=2,
            ),
            "mean attention share of the target-logit attribution over global-trigger attacks",
        ),
        "routing_local_rank_routing": (
            fmt(
                mean_or_none(
                    [s["rank_routing"] for s in local if s["rank_routing"] is not None]
                ),
                places=2,
            ),
            "mean trigger-token norm rank over the routing band, local-trigger attacks",
        ),
        "routing_benign_rank_routing": (
            fmt(stats.get("benign", {}).get("rank_routing"), places=2),
            "benign trigger-token norm rank over the routing band under the same trigger",
        ),
        "routing_benign_rank_late": (
            fmt(stats.get("benign", {}).get("rank_late"), places=2),
            "benign trigger-token norm rank over the late band under the same trigger",
        ),
        "routing_badnet_takeover": (
            fmt(badnet.get("takeover_routing"), places=2),
            "badnet_a2o takeover fraction over the routing band",
        ),
        "routing_benign_takeover": (
            fmt(stats.get("benign", {}).get("takeover_routing"), places=2),
            "benign takeover fraction over the routing band",
        ),
        "routing_benign_ratio_peak": (
            fmt(stats.get("benign", {}).get("ratio_clean_peak"), places=2),
            "benign peak max-over-median patch-norm ratio on clean images, layer 3 on",
        ),
        "routing_badnet_ratio_peak": (
            fmt(badnet.get("ratio_peak"), places=2),
            "badnet_a2o peak max-over-median patch-norm ratio on triggered images, layer 3 on",
        ),
        "routing_global_cls_routing": (
            fmt(
                mean_or_none(
                    [s["cls_routing"] for s in global_ if s["cls_routing"] is not None]
                )
            ),
            "mean CLS attention mass on the trigger tokens over the routing band, global-trigger attacks",
        ),
    }
    write_macros(
        os.path.join(args.paper_dir, "tables", "routing.macros.json"),
        GENERATOR,
        inputs,
        macros,
    )
    print(
        "routing: "
        + ", ".join(
            f"{a}: cls5-8 {fmt(s['cls_routing'])} rank5-8 {fmt(s['rank_routing'], 2)}"
            for a, s in stats.items()
        )
    )


if __name__ == "__main__":
    main()
