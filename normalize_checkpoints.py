"""One-time migration: normalize checkpoints/ folder names and write args.json.

Folder names today are inconsistent in two known ways: the SAM rho tag is
"sam_rho_0_1" in backdoor sweeps but "sam_rho0_1" in benign runs, and the
architecture token sits as a prefix ("swin_cifar100_...") for attacks but
mid-name ("cifar100_swin_benign_...") for benign. This script parses every
folder under checkpoints/ into (architecture, dataset, attack_or_benign,
poison_rate, optimizer, rho), renames to the canonical template

    {architecture}_{dataset}_{attack_or_benign}[_{poison_rate_tag}][_sam_rho_{rho_tag}]

SAM is always SAM-on-top-of-AdamW here, so adam is the unmarked default and
gets no tag at all, only a SAM run adds the rho suffix. Writes an args.json
into every folder (renamed or not), where "optimizer" is still explicit
("adam" or "sam") regardless of what the folder name omits.

Re-runnable: folders already in canonical form are left alone (no git mv) but
still get args.json written. backdoor_bench_checkpoints/ is out of scope, it's
external downloaded data in BackdoorBench's own convention.
"""

import argparse
import json
import os
import shutil

import torch

from attacks import ATTACK_NAMES, default_config

CHECKPOINTS_DIR = "checkpoints"

DATASET_TOKENS = ("cifar100", "cifar10", "gtsrb", "tiny")

# Longest token sequence first so "badnet_a2o" matches before bare "badnet".
ATTACK_TOKEN_SEQUENCES = (
    (("badnet", "a2o"), "badnet_a2o"),
    (("badnet", "a2a"), "badnet_a2a"),
    (("adaptive", "blend"), "adaptive_blend"),
    (("badnet",), "badnet_a2o"),
    (("blend",), "blend"),
    (("sig",), "sig"),
    (("wanet",), "wanet"),
    (("lf",), "lf"),
    (("lc",), "lc"),
    (("bpp",), "bpp"),
    (("tact",), "tact"),
)

# ViT's wrapped vit_b_16 and Swin's wrapped swin have structurally distinct
# state_dict key substrings, checked here rather than guessed from the name.
VIT_KEY_MARKERS = ("conv_proj", "class_token", "encoder.layers.encoder_layer_")
SWIN_KEY_MARKERS = ("features.",)


class UnparsedFolder(Exception):
    pass


def list_checkpoint_folders(checkpoints_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, name))
    )


def detect_architecture_from_state_dict(checkpoint_path: str) -> str:
    """Ground truth when a folder name has neither a "vit" nor "swin" token."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    keys = list(checkpoint["model"].keys())
    is_vit = any(marker in key for key in keys for marker in VIT_KEY_MARKERS)
    is_swin = any(marker in key for key in keys for marker in SWIN_KEY_MARKERS)
    if is_vit and not is_swin:
        return "vit"
    if is_swin and not is_vit:
        return "swin"
    raise UnparsedFolder(
        f"state_dict at {checkpoint_path} matched vit={is_vit} swin={is_swin}, "
        "expected exactly one"
    )


def extract_architecture(
    tokens: list[str], folder_path: str
) -> tuple[str, list[str], bool]:
    """Remove the architecture token if present, else infer from the checkpoint.

    Returns (architecture, remaining_tokens, was_inferred).
    """
    if "swin" in tokens:
        tokens = tokens.copy()
        tokens.remove("swin")
        return "swin", tokens, False
    if "vit" in tokens:
        tokens = tokens.copy()
        tokens.remove("vit")
        return "vit", tokens, False
    architecture = detect_architecture_from_state_dict(
        os.path.join(folder_path, "attack_result.pt")
    )
    return architecture, tokens, True


def extract_dataset(tokens: list[str]) -> tuple[str, list[str]]:
    for index, token in enumerate(tokens):
        if token in DATASET_TOKENS:
            remaining = tokens[:index] + tokens[index + 1 :]
            return token, remaining
    raise UnparsedFolder(f"no dataset token found in {tokens}")


def match_attack_tokens(tokens: list[str]) -> tuple[str, list[str]]:
    """Match the longest known attack token sequence at the start of tokens."""
    for sequence, canonical_name in ATTACK_TOKEN_SEQUENCES:
        length = len(sequence)
        if tuple(tokens[:length]) == sequence:
            return canonical_name, tokens[length:]
    raise UnparsedFolder(f"no attack token sequence matched the start of {tokens}")


def extract_sam_block(tokens: list[str]) -> tuple[float | None, list[str]]:
    """Find the sam/rho tag anywhere in tokens (it is not always trailing).

    Two known formats: "sam", "rho", "0", digits (backdoor sweeps) and "sam",
    "rho0", digits (benign runs). Both encode rho as float(f"0.{digits}").
    Returns (rho_or_None, leftover_tokens_with_sam_block_removed).
    """
    if "sam" not in tokens:
        return None, tokens
    index = tokens.index("sam")
    if index + 1 < len(tokens) and tokens[index + 1] == "rho":
        block_length = 4  # sam, rho, 0, digits
    elif index + 1 < len(tokens) and tokens[index + 1] == "rho0":
        block_length = 3  # sam, rho0, digits
    else:
        raise UnparsedFolder(f"unrecognized sam/rho tag shape in {tokens}")
    digits = tokens[index + block_length - 1]
    rho = float(f"0.{digits}")
    leftover = tokens[:index] + tokens[index + block_length :]
    return rho, leftover


def extract_poison_rate_tag(tokens: list[str]) -> str:
    if len(tokens) != 2 or tokens[0] != "0":
        raise UnparsedFolder(f"expected a single poison-rate tag, got {tokens}")
    return "_".join(tokens)


class ParsedFolder:
    def __init__(
        self,
        original_name: str,
        architecture: str,
        architecture_inferred: bool,
        dataset: str,
        attack_or_benign: str,
        poison_rate_tag: str | None,
        rho: float | None,
    ):
        self.original_name = original_name
        self.architecture = architecture
        self.architecture_inferred = architecture_inferred
        self.dataset = dataset
        self.attack_or_benign = attack_or_benign
        self.poison_rate_tag = poison_rate_tag
        self.rho = rho

    @property
    def optimizer(self) -> str:
        return "sam" if self.rho is not None else "adam"

    @property
    def optimizer_tag(self) -> str | None:
        # SAM is always SAM-on-top-of-AdamW here, so adam is the unmarked
        # default and gets no suffix at all, only the SAM+rho case is tagged.
        if self.rho is None:
            return None
        rho_tag = str(self.rho).replace(".", "_")
        return f"sam_rho_{rho_tag}"

    @property
    def poison_rate(self) -> float:
        if self.poison_rate_tag is None:
            return 0.0
        return float(self.poison_rate_tag.replace("_", ".", 1))

    def canonical_name(self) -> str:
        parts = [self.architecture, self.dataset, self.attack_or_benign]
        if self.poison_rate_tag is not None:
            parts.append(self.poison_rate_tag)
        if self.optimizer_tag is not None:
            parts.append(self.optimizer_tag)
        return "_".join(parts)


def parse_folder_name(folder_name: str, checkpoints_dir: str) -> ParsedFolder:
    tokens = folder_name.split("_")
    folder_path = os.path.join(checkpoints_dir, folder_name)

    # Re-running against folders already renamed by an earlier version of this
    # script (which used to tag adam explicitly) must still parse: adam carries
    # no information once rho is None, so drop a stray token for it.
    if "adam" in tokens:
        tokens = [token for token in tokens if token != "adam"]

    architecture, tokens, architecture_inferred = extract_architecture(
        tokens, folder_path
    )
    dataset, tokens = extract_dataset(tokens)

    if "benign" in tokens:
        tokens = tokens.copy()
        tokens.remove("benign")
        rho, leftover = extract_sam_block(tokens)
        if leftover:
            raise UnparsedFolder(f"unexpected leftover tokens for benign: {leftover}")
        return ParsedFolder(
            folder_name,
            architecture,
            architecture_inferred,
            dataset,
            "benign",
            None,
            rho,
        )

    attack_name, tokens = match_attack_tokens(tokens)
    rho, leftover = extract_sam_block(tokens)
    poison_rate_tag = extract_poison_rate_tag(leftover)
    return ParsedFolder(
        folder_name,
        architecture,
        architecture_inferred,
        dataset,
        attack_name,
        poison_rate_tag,
        rho,
    )


def build_args_json(parsed: ParsedFolder) -> dict:
    if parsed.attack_or_benign == "benign":
        label_mode = None
        cover_rate = 0.0
    else:
        config = default_config(parsed.attack_or_benign)
        label_mode = config.label_mode
        cover_rate = getattr(config, "cover_rate", 0.0)

    return {
        "dataset": parsed.dataset,
        "attack": parsed.attack_or_benign,
        "label_mode": label_mode,
        "target_label": 0,
        "poison_rate": parsed.poison_rate,
        "cover_rate": cover_rate,
        "architecture": parsed.architecture,
        "optimizer": parsed.optimizer,
        "rho": parsed.rho,
        "epochs": 15,
        "seed": None,
        "git_commit": None,
        "trained_started_at": None,
        "trained_ended_at": None,
    }


def rename_folder(checkpoints_dir: str, old_name: str, new_name: str) -> None:
    # checkpoints/ is entirely gitignored (large binary checkpoints), so `git mv`
    # refuses ("source directory is empty" from git's point of view). A plain
    # filesystem move is the correct operation here, git has nothing to track.
    shutil.move(
        os.path.join(checkpoints_dir, old_name), os.path.join(checkpoints_dir, new_name)
    )


def write_args_json(checkpoints_dir: str, folder_name: str, args_json: dict) -> None:
    path = os.path.join(checkpoints_dir, folder_name, "args.json")
    with open(path, "w") as file:
        json.dump(args_json, file, indent=2)
        file.write("\n")


def parse_all_folders(
    folder_names: list[str], checkpoints_dir: str
) -> tuple[list[ParsedFolder], list[tuple[str, str]]]:
    parsed_folders = []
    unparsed = []
    for folder_name in folder_names:
        try:
            parsed_folders.append(parse_folder_name(folder_name, checkpoints_dir))
        except UnparsedFolder as error:
            unparsed.append((folder_name, str(error)))
    return parsed_folders, unparsed


def print_dry_run_report(
    parsed_folders: list[ParsedFolder], unparsed: list[tuple[str, str]]
) -> None:
    renamed = [p for p in parsed_folders if p.canonical_name() != p.original_name]
    already_canonical = [
        p for p in parsed_folders if p.canonical_name() == p.original_name
    ]
    inferred = [p for p in parsed_folders if p.architecture_inferred]

    print("architecture inferred from attack_result.pt state_dict:")
    for parsed in inferred:
        print(f"  {parsed.original_name} -> architecture={parsed.architecture}")
    print(f"  ({len(inferred)} folders)")
    print()

    print("proposed renames:")
    for parsed in renamed:
        print(f"  {parsed.original_name} -> {parsed.canonical_name()}")
    print(
        f"  ({len(renamed)} folders to rename, {len(already_canonical)} already canonical)"
    )
    print()

    if unparsed:
        print("UNPARSED folders (flagged, not touched):")
        for folder_name, reason in unparsed:
            print(f"  {folder_name}: {reason}")
        print()

    print("summary:")
    print(f"  total folders scanned: {len(parsed_folders) + len(unparsed)}")
    print(f"  renamed: {len(renamed)}")
    print(f"  already canonical: {len(already_canonical)}")
    print(f"  unparsed/flagged: {len(unparsed)}")
    print(f"  architecture inferred: {len(inferred)}")


def apply_renames_and_write_metadata(
    parsed_folders: list[ParsedFolder], checkpoints_dir: str
) -> tuple[int, int]:
    renamed_count = 0
    args_written_count = 0
    for parsed in parsed_folders:
        canonical_name = parsed.canonical_name()
        if canonical_name != parsed.original_name:
            rename_folder(checkpoints_dir, parsed.original_name, canonical_name)
            renamed_count += 1
        write_args_json(checkpoints_dir, canonical_name, build_args_json(parsed))
        args_written_count += 1
    return renamed_count, args_written_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize checkpoints/ folder names and write args.json"
    )
    parser.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the proposed changes without touching the filesystem (default)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually execute the renames and write args.json (overrides --dry-run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_changes = args.apply

    folder_names = list_checkpoint_folders(args.checkpoints_dir)
    parsed_folders, unparsed = parse_all_folders(folder_names, args.checkpoints_dir)

    if not apply_changes:
        print_dry_run_report(parsed_folders, unparsed)
        return

    if unparsed:
        print("Refusing to apply: unparsed folders present:")
        for folder_name, reason in unparsed:
            print(f"  {folder_name}: {reason}")
        return

    renamed_count, args_written_count = apply_renames_and_write_metadata(
        parsed_folders, args.checkpoints_dir
    )
    already_canonical_count = len(parsed_folders) - renamed_count
    print(f"renamed: {renamed_count}")
    print(f"already canonical: {already_canonical_count}")
    print(f"args.json written: {args_written_count}")


if __name__ == "__main__":
    main()
