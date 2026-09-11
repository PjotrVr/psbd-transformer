"""T?: the H41 probe union, read on the ordinary (non-adaptive) backdoored ViT-B/16 models.

H41 built the min-rank probe union against an attacker trained to evade 1
probed operator. This table reads the same union rule where nobody trained
against a probe: the 65 models the paper's headline reads. 6 probe sets, each
a row: PSBD-TM alone, PSBD-TM plus PSBD-RD, PSBD-TM plus token masking on the
attention branch output, the 3-probe pool from the adaptive-attacker section
(PSBD-TM, attention-input dropout, MLP-norm-out gain scaling), that pool plus
PSBD-RD, and every basis placement present on all 65 models. Nothing here is
computed: every number comes from
results/_experiments/probe_union/probe_union.json, written by
experiments/probe_union/measure.py.

    PYTHONPATH=. .venv/bin/python scripts/paper/tab_probe_union.py \\
        --results-dir results --paper-dir paper
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from scripts.paper._common import (  # noqa: E402
    build_parser,
    ci_text,
    fmt,
    load_json,
    write_macros,
    write_table,
)

GENERATOR = "scripts/paper/tab_probe_union.py"
RECORD_PATH = os.path.join("results", "_experiments", "probe_union", "probe_union.json")

# Row order and the reader-facing name of each probe set's placements. The
# reference row carries no gain column entry, since it is what every other row
# is measured against.
ROW_LABELS = (
    ("psbd_tm", "PSBD-TM alone"),
    ("psbd_tm_rd", "PSBD-TM + PSBD-RD"),
    ("psbd_tm_attn_branch", "PSBD-TM + attention branch output token mask"),
    ("adaptive_3probe", "PSBD-TM + attention input dropout + MLP norm-out gain scale"),
    (
        "adaptive_4probe",
        "PSBD-TM + attention input dropout + MLP norm-out gain scale + PSBD-RD",
    ),
    ("all_65_basis", "every basis placement present on all 65 models"),
)


def load_record(results_dir: str) -> dict:
    path = os.path.join(results_dir, "_experiments", "probe_union", "probe_union.json")
    record = load_json(path)
    if record is None:
        raise SystemExit(
            f"{path} does not exist, run experiments/probe_union/measure.py first"
        )
    return record


def table_row(name: str, label: str, record: dict) -> list[str]:
    block = record["probe_sets"][name]
    summary = block["summary"]
    gain = block.get("gain_over_psbd_tm")
    if gain is None:
        gain_text = "--"
    else:
        gain_text = f"{fmt(gain['mean_gain'], signed=True)} {ci_text(gain['ci_low'], gain['ci_high'])}"
    row = [
        label,
        str(summary["n_models"]),
        fmt(summary["auroc_mean"]),
        fmt(summary["tpr_at_0.10_mean"]),
        fmt(summary["tpr_at_0.20_mean"]),
        gain_text,
    ]
    return row


def widen_table(path: str) -> None:
    """Rewrite the generated table as a full-width float, since its 6 columns overflow 1 column."""
    with open(path) as handle:
        text = handle.read()
    text = text.replace("\\begin{table}[htbp]", "\\begin{table*}[htbp]").replace(
        "\\end{table}", "\\end{table*}"
    )
    text = text.replace("\\begin{adjustbox}{max width=\\linewidth}\n", "").replace(
        "\\end{adjustbox}\n", ""
    )
    with open(path, "w") as handle:
        handle.write(text)


def main() -> None:
    args = build_parser(__doc__).parse_args()
    record = load_record(args.results_dir)

    inputs = [RECORD_PATH]

    rows = [table_row(name, label, record) for name, label in ROW_LABELS]
    write_table(
        path=os.path.join(args.paper_dir, "tables", "probe_union.tex"),
        generator=GENERATOR,
        inputs=inputs,
        caption=(
            "The min-rank union of probes on the 65 backdoored ViT-B/16 models, with the paired AUROC gain over PSBD-TM alone."
        ),
        label="tab:probe-union",
        header=[
            "Probes",
            "n",
            "AUROC",
            "TPR@FPR 10%",
            "TPR@FPR 20%",
            "Gain over PSBD-TM [95% CI]",
        ],
        rows=rows,
        align="lrrrrl",
    )
    widen_table(os.path.join(args.paper_dir, "tables", "probe_union.tex"))

    tm = record["probe_sets"]["psbd_tm"]["summary"]
    tm_rd = record["probe_sets"]["psbd_tm_rd"]
    wanet = record["wanet_cifar10"]
    macros = {
        "probe_union_n_models": (
            str(record["n_models_selected"]),
            "backdoored ViT-B/16 models the probe-union check reads, "
            "the models with both headline placements",
        ),
        "probe_union_psbd_tm_auroc": (
            fmt(tm["auroc_mean"]),
            "mean AUROC of PSBD-TM alone over the models the probe-union check reads",
        ),
        "probe_union_tm_rd_gain": (
            fmt(tm_rd["gain_over_psbd_tm"]["mean_gain"], signed=True),
            "paired AUROC gain of the PSBD-TM + PSBD-RD union over PSBD-TM alone",
        ),
        "probe_union_tm_rd_gain_ci": (
            ci_text(
                tm_rd["gain_over_psbd_tm"]["ci_low"],
                tm_rd["gain_over_psbd_tm"]["ci_high"],
            ),
            "95% bootstrap interval of that gain",
        ),
        "probe_union_wanet_cifar10_tm": (
            fmt(wanet["psbd_tm"]["auroc"]) if wanet["psbd_tm"] else "--",
            "PSBD-TM alone on WaNet at 10% on CIFAR-10, the inverted cell the "
            "headline names",
        ),
        "probe_union_wanet_cifar10_tm_rd": (
            fmt(wanet["psbd_tm_rd"]["auroc"]) if wanet["psbd_tm_rd"] else "--",
            "PSBD-TM + PSBD-RD union on the same cell",
        ),
    }
    write_macros(
        sidecar_path=os.path.join(args.paper_dir, "tables", "probe_union.macros.json"),
        generator=GENERATOR,
        inputs=inputs,
        macros=macros,
    )

    print(f"wrote {args.paper_dir}/tables/probe_union.tex")
    print(f"wrote {args.paper_dir}/tables/probe_union.macros.json")


if __name__ == "__main__":
    main()
