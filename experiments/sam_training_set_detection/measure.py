"""Does SAM training amplify training-set poison detection on ViT-B/16?

Zhang et al. (arXiv 2411.11525) claim SAM training amplifies the separation
between poisoned and clean features at a backdoored model's penultimate layer,
which makes training-set poison detectors (Spectral Signatures, Activation
Clustering, and others) more effective. Their setting is training-set poisoned
sample detection: a defender holds the released poisoned training set itself and
must flag which rows to drop before training, using features from a model already
trained on that set. This is a different question from PSBD, which probes a
finished model at test time with no access to the training set at all.

This script rebuilds the exact poisoned training set 6 matched checkpoint pairs
were trained on (Adam vs SAM rho=0.1, at 1% and 5% poisoning, CIFAR-100,
BadNets-A2O / Blend / WaNet), extracts class-token features after the final
LayerNorm for the target class, and runs Spectral Signatures (Tran et al.) and
Activation Clustering (Chen et al.) against the ground-truth poison indices. See
README.md for the full method and the result.

    PYTHONPATH=. .venv/bin/python experiments/sam_training_set_detection/measure.py
"""

import json
import os
from dataclasses import replace

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset

from attacks import build_attack, default_config
from attacks.poisoning import (
    PoisonedTrainingSet,
    choose_indices_with_cover,
    choose_poison_indices,
    poisoned_label,
)
from data.loading import (
    base_image_transform,
    extract_labels,
    limit_dataset,
    load_clean_datasets,
)
from data.registry import DATASET_REGISTRY
from data.splits import read_checkpoint_metadata
from models.backbones import load_checkpoint, network_core

CHECKPOINT_DIR = "checkpoints"
RAW_DATA_DIR = "raw_data"
RESULTS_DIR = "results/_experiments/sam_training_set_detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

# (dataset, attack folder stub, poison rate tag). The SAM checkpoint is the same
# stub with _sam_rho_0_1 appended, the recipe's own placement for --rho 0.1.
PAIRS = [
    ("cifar100", "badnet_a2o", "0_01"),
    ("cifar100", "badnet_a2o", "0_05"),
    ("cifar100", "blend", "0_01"),
    ("cifar100", "blend", "0_05"),
    ("cifar100", "wanet", "0_01"),
    ("cifar100", "wanet", "0_05"),
]

# Tran et al.'s own removal rule: flag the top multiplier times the expected
# poison count by squared projection onto the top singular vector.
SPECTRAL_SIGNATURE_MULTIPLIER = 1.5


def checkpoint_path(folder_name: str) -> str:
    path = os.path.join(CHECKPOINT_DIR, folder_name, "attack_result.pt")
    return path


def rebuild_poisoned_training_set(metadata: dict):
    """The exact (train_clean, attack, poison_indices) the checkpoint trained on.

    Mirrors cli.train_backdoor.build_training_loader: the same clean load, the
    same limit_dataset call, the same cover-rate resolution and the same rng draw
    from (labels, attack, poison_rate, seed), so the recovered index set is bit
    for bit what training saw. Asserted against the args.json poisoned count
    rather than trusted, since a mismatch here would silently mislabel every
    ground-truth flag downstream.
    """
    dataset_name = metadata["dataset"]
    spec = DATASET_REGISTRY[dataset_name]
    transform = base_image_transform(spec.image_size)
    train_clean, _ = load_clean_datasets(dataset_name, transform, RAW_DATA_DIR)

    # Older sidecars record seed and max_samples as null rather than omitting
    # them. Both fields fall back to train_backdoor's own argparse defaults,
    # which is what an unrecorded run actually used.
    seed = metadata.get("seed")
    seed = 0 if seed is None else seed
    max_samples = metadata.get("max_samples")
    train_clean = limit_dataset(train_clean, max_samples, seed)
    labels = extract_labels(train_clean)  # list[int], len n_train

    config = default_config(metadata["attack"])
    cover_rate = metadata.get("cover_rate") or 0.0
    if cover_rate > 0.0:
        config = replace(config, cover_rate=cover_rate)
    attack = build_attack(
        metadata["attack"], config, spec.image_size, metadata["target_label"]
    )

    poison_rate = metadata["poison_rate"]
    if cover_rate > 0.0:
        poison_indices, _cover_indices = choose_indices_with_cover(
            labels, attack, poison_rate, cover_rate, None, seed
        )
    else:
        poison_indices = choose_poison_indices(labels, attack, poison_rate, seed)

    expected_poisoned = metadata["n_poisoned"]
    assert len(poison_indices) == expected_poisoned, (
        f"rebuilt {len(poison_indices)} poisoned indices, args.json records "
        f"{expected_poisoned}"
    )

    return train_clean, labels, attack, poison_indices, spec


def target_class_subset(labels, attack, poison_indices, num_classes):
    """Indices whose label reads as the target class after poisoning, plus flags.

    Spectral Signatures and Activation Clustering both operate per suspected
    class: the target class is exactly where a poisoned sample lands under
    all_to_one, so this is the whole population either detector ever sees. It
    always undercuts 10000 images at these rates on CIFAR-100 (at most 2500
    poisoned plus roughly 500 clean target-class images), so no further random
    subsampling is applied.
    """
    kept_indices = []
    poisoned_flags = []
    for index, label in enumerate(labels):
        if index in poison_indices:
            final_label = poisoned_label(
                attack.label_mode,
                label,
                attack.target_label,
                num_classes,
                attack.num_targets,
            )
        else:
            final_label = label
        if final_label == attack.target_label:
            kept_indices.append(index)
            poisoned_flags.append(index in poison_indices)
    return kept_indices, np.array(poisoned_flags, dtype=bool)


def extract_cls_features(model, loader, device) -> np.ndarray:
    """Class-token features after the classifier's own final LayerNorm.

    torchvision's VisionTransformer.Encoder applies its LayerNorm internally
    before returning, so it never appears in models.positions' block outputs.
    Hooking it directly reads the exact vector network.heads.head consumes,
    matching the "feature extractor" phi the SAM paper's PSD stage-2 reads from.
    """
    ln = network_core(model).encoder.ln
    captured = {}

    def hook(_module, _inputs, output):
        captured["out"] = output.detach()

    handle = ln.register_forward_hook(hook)
    model.eval()

    features = []
    with torch.inference_mode():
        for images, _labels in loader:
            images = images.to(device)  # (batch, 3, height, width)
            model(images)
            cls_token = captured["out"][:, 0, :].float().cpu()  # (batch, 768)
            features.append(cls_token)
    handle.remove()

    stacked = torch.cat(features, dim=0).numpy()  # (n_kept, 768)
    return stacked


def spectral_signature_scores(features: np.ndarray):
    """Tran et al.: squared projection onto the top right-singular vector.

    original form
        M centered, M = U diag(s) V^T, score_i = <M_i, v_1>^2
    """
    mean = features.mean(axis=0, keepdims=True)  # (1, dim)
    centered = features - mean  # (n, dim)
    _u, singular_values, v_transpose = np.linalg.svd(centered, full_matrices=False)
    top_direction = v_transpose[0]  # (dim,)
    scores = (centered @ top_direction) ** 2  # (n,)
    singular_ratio = float(singular_values[0] / singular_values[1])
    return scores, singular_ratio


def spectral_signature_flags(
    scores: np.ndarray, expected_poison_count: int
) -> np.ndarray:
    flag_count = min(
        len(scores), int(round(SPECTRAL_SIGNATURE_MULTIPLIER * expected_poison_count))
    )
    order = np.argsort(-scores)  # descending
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[order[:flag_count]] = True
    return flagged


def activation_clustering(features: np.ndarray):
    """Chen et al.: PCA to 10 dims, 2-means, the smaller cluster is poisoned."""
    reduced = PCA(n_components=10, random_state=0).fit_transform(features)  # (n, 10)
    cluster_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
        reduced
    )
    counts = np.bincount(cluster_labels)
    minority_cluster = int(np.argmin(counts))
    flagged = cluster_labels == minority_cluster
    return flagged, reduced


def tpr_fpr(flagged: np.ndarray, poisoned_flags: np.ndarray) -> tuple[float, float]:
    n_positive = poisoned_flags.sum()
    n_negative = (~poisoned_flags).sum()
    true_positive = (flagged & poisoned_flags).sum()
    false_positive = (flagged & ~poisoned_flags).sum()
    tpr = float(true_positive / n_positive) if n_positive > 0 else float("nan")
    fpr = float(false_positive / n_negative) if n_negative > 0 else float("nan")
    return tpr, fpr


def evaluate_checkpoint(
    folder_name: str, expected_poison_count_override: int | None = None
):
    """1 checkpoint's full pass: rebuild its poisoned set, extract, detect, score."""
    metadata = read_checkpoint_metadata(checkpoint_path(folder_name))
    train_clean, labels, attack, poison_indices, spec = rebuild_poisoned_training_set(
        metadata
    )
    normalize = transforms_v2.Normalize(mean=spec.mean, std=spec.std)
    poisoned_full = PoisonedTrainingSet(
        train_clean, attack, poison_indices, normalize, spec.num_classes
    )

    kept_indices, poisoned_flags = target_class_subset(
        labels, attack, poison_indices, spec.num_classes
    )
    loader = DataLoader(
        Subset(poisoned_full, kept_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    model = load_checkpoint("vit", checkpoint_path(folder_name), DEVICE)
    features = extract_cls_features(model, loader, DEVICE)  # (n_kept, 768)
    del model
    torch.cuda.empty_cache()

    expected_poison_count = (
        expected_poison_count_override
        if expected_poison_count_override is not None
        else int(poisoned_flags.sum())
    )
    ss_scores, singular_ratio = spectral_signature_scores(features)
    ss_flags = spectral_signature_flags(ss_scores, expected_poison_count)
    ss_tpr, ss_fpr = tpr_fpr(ss_flags, poisoned_flags)

    ac_flags, reduced = activation_clustering(features)
    ac_tpr, ac_fpr = tpr_fpr(ac_flags, poisoned_flags)

    silhouette = float(silhouette_score(reduced, poisoned_flags.astype(int)))

    return {
        "folder_name": folder_name,
        "n_poisoned": int(poisoned_flags.sum()),
        "n_clean_target": int((~poisoned_flags).sum()),
        "ss_tpr": ss_tpr,
        "ss_fpr": ss_fpr,
        "ac_tpr": ac_tpr,
        "ac_fpr": ac_fpr,
        "silhouette": silhouette,
        "singular_ratio": singular_ratio,
    }


def sam_folder_name(dataset: str, attack: str, rate_tag: str) -> str:
    name = f"vit_{dataset}_{attack}_{rate_tag}_sam_rho_0_1"
    return name


def adam_folder_name(dataset: str, attack: str, rate_tag: str) -> str:
    name = f"vit_{dataset}_{attack}_{rate_tag}"
    return name


def print_summary_table(rows: list[dict]) -> None:
    header = (
        "pair | optimizer | poisoned | SS TPR | SS FPR | AC TPR | AC FPR | "
        "silhouette | singular ratio"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['pair']} | {row['optimizer']} | {row['n_poisoned']} | "
            f"{row['ss_tpr']:.3f} | {row['ss_fpr']:.3f} | "
            f"{row['ac_tpr']:.3f} | {row['ac_fpr']:.3f} | "
            f"{row['silhouette']:.3f} | {row['singular_ratio']:.2f}"
        )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows = []

    for dataset, attack, rate_tag in PAIRS:
        pair_name = f"{dataset}_{attack}_{rate_tag}"
        adam_result = evaluate_checkpoint(adam_folder_name(dataset, attack, rate_tag))
        sam_result = evaluate_checkpoint(sam_folder_name(dataset, attack, rate_tag))

        record = {
            "dataset": dataset,
            "attack": attack,
            "poison_rate_tag": rate_tag,
            "adam": adam_result,
            "sam": sam_result,
        }
        with open(os.path.join(RESULTS_DIR, f"{pair_name}.json"), "w") as handle:
            json.dump(record, handle, indent=2)

        summary_rows.append({"pair": pair_name, "optimizer": "adam", **adam_result})
        summary_rows.append({"pair": pair_name, "optimizer": "sam", **sam_result})
        print(f"done: {pair_name}")

    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
