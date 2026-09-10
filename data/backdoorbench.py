"""Loading BackdoorBench poisoned test images and building the evaluation splits.

BackdoorBench saves each poisoned test image as a PNG whose filename is the
original dataset index. That index lets us recover the exact clean counterpart for
a paired clean/backdoor comparison, which is the only way to read a poisoned test
set this project did not generate itself.

This is the PNG-backed twin of data.splits, which rebuilds a poisoned test set in
memory from the attack recorded in a checkpoint's args.json. BackdoorBench folders
carry no args.json, so they come through here instead. The eval-time eligibility
and labeling questions are the same in both paths, and both answer them with
attacks.poisoning's eval-time function pair, never the training-time pair.
"""

import glob
import os
from pathlib import Path

import numpy as np
from lightning import seed_everything
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from .loading import extract_labels
from attacks.poisoning import attack_success_label, is_eval_poisonable


class PngPathDataset(Dataset):
    """Serves poisoned images from PNG files, filtered and labeled like AttackSuccessSet.

    Eligibility and labeling follow the same eval-time question AttackSuccessSet
    asks in memory: is this sample allowed into the attack-success set, and what
    label counts as success. Ineligible paths (for example already-target-class
    images under all_to_one) are filtered out at construction time rather than
    served with a wrong label.
    """

    def __init__(
        self,
        paths: list[str],
        transform,
        true_labels: list[int],
        label_mode: str,
        target_label: int,
        num_classes: int,
    ):
        self.transform = transform
        self.true_labels = true_labels
        self.label_mode = label_mode
        self.target_label = target_label
        self.num_classes = num_classes
        self.paths = paths
        self.eligible_positions = [
            position
            for position, label in enumerate(true_labels)
            if is_eval_poisonable(label_mode, int(label), target_label)
        ]

    def __len__(self) -> int:
        return len(self.eligible_positions)

    def __getitem__(self, idx: int) -> tuple:
        position = self.eligible_positions[idx]
        image = Image.open(self.paths[position]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = attack_success_label(
            self.label_mode,
            int(self.true_labels[position]),
            self.target_label,
            self.num_classes,
        )
        return image, label


def load_backdoor_splits(
    folder_name: str,
    clean_test_dataset: Dataset,
    transform,
    weights_dir: str,
    label_mode: str,
    target_label: int,
    num_classes: int,
) -> tuple[Dataset, Dataset]:
    """(backdoor_test, clean_counterparts), both filtered to eligible samples.

    The 2 are index-aligned: position i in each is the same original test image,
    with and without the trigger. Samples is_eval_poisonable excludes are dropped
    from both, so backdoor_test can be shorter than the PNG folder itself.
    """
    backdoor_dir = os.path.join(weights_dir, folder_name, "bd_test_dataset")
    backdoor_paths = sorted(glob.glob(f"{backdoor_dir}/**/*.png", recursive=True))
    if not backdoor_paths:
        raise FileNotFoundError(f"No PNG files found in {backdoor_dir}")

    # The filename stem is the original test-set index, which is the only link
    # back to the clean image the trigger was stamped on.
    original_indices = [int(Path(p).stem) for p in backdoor_paths]
    clean_counterparts = Subset(clean_test_dataset, original_indices)
    true_labels = extract_labels(clean_counterparts)

    backdoor_test = PngPathDataset(
        backdoor_paths,
        transform=transform,
        true_labels=true_labels,
        label_mode=label_mode,
        target_label=target_label,
        num_classes=num_classes,
    )
    eligible_counterparts = Subset(clean_counterparts, backdoor_test.eligible_positions)
    return backdoor_test, eligible_counterparts


def split_validation_and_eval(
    clean_counterparts: Dataset,
    backdoor_test: Dataset,
    clean_val_size: int,
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    """(clean_val, clean_eval, backdoor_eval), with a validation set carved off the clean side.

    Stratified by label so the validation quantile threshold sees every class. The
    backdoor eval set reuses the clean eval indices, which keeps the pairing.
    """
    labels = np.array(extract_labels(clean_counterparts))
    indices = np.arange(len(clean_counterparts))

    val_idx, eval_idx = train_test_split(
        indices,
        test_size=len(indices) - clean_val_size,
        stratify=labels,
        random_state=seed,
    )

    clean_val = Subset(clean_counterparts, val_idx.tolist())
    clean_eval = Subset(clean_counterparts, eval_idx.tolist())
    backdoor_eval = Subset(backdoor_test, eval_idx.tolist())
    return clean_val, clean_eval, backdoor_eval


def balance_by_class(
    clean_eval: Dataset,
    backdoor_eval: Dataset,
    examples_per_class: int,
    seed: int,
) -> tuple[Subset, Subset]:
    """The 2 eval sets cut to a fixed number of examples per clean class.

    The backdoor side reuses the selected clean indices so the pairing survives.
    """
    clean_labels = extract_labels(clean_eval)

    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(clean_labels):
        class_to_indices.setdefault(int(label), []).append(idx)

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(class_to_indices):
        candidates = class_to_indices[class_id]
        rng.shuffle(candidates)
        selected.extend(candidates[:examples_per_class])

    balanced_clean = Subset(clean_eval, selected)
    balanced_backdoor = Subset(backdoor_eval, selected)
    return balanced_clean, balanced_backdoor
