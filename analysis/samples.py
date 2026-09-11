"""The dataset views a visualisation tool reads, built over the paired PSBD splits.

BackdoorBench's analysis scripts draw their images from 5 places: the clean test
set, the poisoned test set, a mixed test set where a poison ratio of the rows
carry the trigger, a class-balanced subset of any of those, and a handful of
2 clean plus 2 poisoned examples for the image panels (visual_utils.py,
generate_clean_dataset, generate_bd_dataset, generate_mix_dataset and
sub_sample_euqal_ratio_classes_index at commit f02e353). This module builds the
same 5 views over this project's loaders, so every tool sees the rows every
detector saw.

The clean loader serves the whole analysis pool and the backdoor loader only its
eligible rows (all-to-one drops the target class), so the 2 are paired through
the manifest rather than by position, the way defences.decision pairs scores.
Every view carries a poison mask beside its labels. The image panels read that
mask, never a row's position, because upstream titles its panels by position
("clean, clean, poison, poison") and mislabels them when fewer than 2 poisoned
rows exist and clean rows fill the gap.

Images stay at the dataset's native resolution in normalised space, which is
what the loaders serve and what the model's own Resize wrapper consumes.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from defences.decision import pair_clean_to_backdoor
from detectors.strip import normalization_buffers

# The label upstream gives poisoned rows when it subsets at equal ratio, so they
# count as a class of their own beside the real ones.
POISON_CLASS_MARK = -1


@dataclass(frozen=True)
class SampleView:
    """1 population of rows a tool reads: images, true labels and who carries the trigger.

    images is (num_rows, channels, height, width) in normalised space, labels
    (num_rows,) long holds the TRUE class of every row even when the row is
    triggered, poison_mask (num_rows,) bool marks the triggered rows and
    test_indices (num_rows,) long names the original test index of each row.
    """

    images: torch.Tensor
    labels: torch.Tensor
    poison_mask: torch.Tensor
    test_indices: torch.Tensor

    def __len__(self) -> int:
        count = int(self.images.shape[0])
        return count

    def subset(self, rows: torch.Tensor) -> "SampleView":
        """The same view restricted to rows, in the order rows gives them."""
        restricted = SampleView(
            images=self.images[rows],  # (len(rows), C, H, W)
            labels=self.labels[rows],  # (len(rows),)
            poison_mask=self.poison_mask[rows],  # (len(rows),)
            test_indices=self.test_indices[rows],  # (len(rows),)
        )
        return restricted


def collect_split(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Every image and label a loader serves, in served order, on the CPU."""
    image_batches = []
    label_batches = []
    for images, labels in loader:
        image_batches.append(images)  # (batch, C, H, W)
        label_batches.append(labels.long())  # (batch,)
    if not image_batches:
        raise ValueError("the loader served no rows, so no view can be built")

    images = torch.cat(image_batches)  # (N, C, H, W)
    labels = torch.cat(label_batches)  # (N,)
    return images, labels


def paired_views(
    loaders: dict[str, DataLoader], manifest: dict
) -> tuple[SampleView, SampleView]:
    """(clean, backdoor) over the same images in the same order, 1 row per eligible image.

    Row i of both views is the same test image, once clean and once triggered.
    Both carry the true label read from the clean split, since the backdoor
    loader's own label is the attack's intended one.
    """
    clean_images, clean_labels = collect_split(loaders["clean"])
    backdoor_images, _ = collect_split(loaders["backdoor"])

    clean_rows = pair_clean_to_backdoor(
        torch.arange(clean_images.shape[0]), manifest
    )  # (num_backdoor,)
    if clean_rows.shape[0] != backdoor_images.shape[0]:
        raise ValueError(
            f"the manifest pairs {clean_rows.shape[0]} rows but the backdoor loader "
            f"served {backdoor_images.shape[0]}, so the views cannot be aligned"
        )
    test_indices = torch.tensor(manifest["analysis_backdoor_indices"], dtype=torch.long)

    clean = SampleView(
        images=clean_images[clean_rows],  # (num_backdoor, C, H, W)
        labels=clean_labels[clean_rows],  # (num_backdoor,)
        poison_mask=torch.zeros(clean_rows.shape[0], dtype=torch.bool),
        test_indices=test_indices,
    )
    backdoor = SampleView(
        images=backdoor_images,  # (num_backdoor, C, H, W)
        labels=clean_labels[clean_rows],  # (num_backdoor,)
        poison_mask=torch.ones(clean_rows.shape[0], dtype=torch.bool),
        test_indices=test_indices,
    )
    return clean, backdoor


def clean_view(loaders: dict[str, DataLoader], manifest: dict) -> SampleView:
    """Upstream's clean_test: every row of the clean analysis split, none triggered."""
    images, labels = collect_split(loaders["clean"])
    view = SampleView(
        images=images,
        labels=labels,
        poison_mask=torch.zeros(images.shape[0], dtype=torch.bool),
        test_indices=torch.tensor(manifest["analysis_clean_indices"], dtype=torch.long),
    )
    return view


def mixed_view(
    clean: SampleView,
    backdoor: SampleView,
    whole_clean: SampleView,
    ratio: float,
    seed: int,
) -> SampleView:
    """Upstream's mixed test: the clean rows with a ratio of them swapped for their triggered self.

    generate_mix_dataset draws int(len(test set) * pratio) rows among the
    eligible ones and marks only those as poisoned, keeping every other row clean
    including the target-class rows the backdoor split never contains. The draw
    here is over the paired rows and the count is taken against the whole clean
    split, as upstream takes it against the whole test set.
    """
    generator = np.random.default_rng(seed)
    num_poisoned = min(int(len(whole_clean) * ratio), len(backdoor))
    chosen = generator.choice(len(backdoor), num_poisoned, replace=False)
    chosen_rows = torch.as_tensor(np.sort(chosen), dtype=torch.long)  # (num_poisoned,)

    swapped_index = {int(index): row for row, index in enumerate(backdoor.test_indices)}
    chosen_set = set(chosen_rows.tolist())
    images = whole_clean.images.clone()  # (N, C, H, W)
    poison_mask = torch.zeros(len(whole_clean), dtype=torch.bool)  # (N,)
    for row, test_index in enumerate(whole_clean.test_indices.tolist()):
        backdoor_row = swapped_index.get(test_index)
        if backdoor_row is None or backdoor_row not in chosen_set:
            continue
        images[row] = backdoor.images[backdoor_row]
        poison_mask[row] = True

    view = SampleView(
        images=images,
        labels=whole_clean.labels,
        poison_mask=poison_mask,
        test_indices=whole_clean.test_indices,
    )
    return view


def union_view(clean: SampleView, backdoor: SampleView) -> SampleView:
    """Both paired populations stacked, clean rows first, each image present twice."""
    view = SampleView(
        images=torch.cat([clean.images, backdoor.images]),  # (2 * N, C, H, W)
        labels=torch.cat([clean.labels, backdoor.labels]),  # (2 * N,)
        poison_mask=torch.cat([clean.poison_mask, backdoor.poison_mask]),  # (2 * N,)
        test_indices=torch.cat([clean.test_indices, backdoor.test_indices]),
    )
    return view


def select_classes(
    num_classes: int, count: int, target_label: int, seed: int
) -> np.ndarray:
    """The classes a categorical figure draws, the target always among them.

    Upstream (visual_tac.py lines 53 to 57) keeps every class when there are no
    more than c_sub of them, and otherwise draws c_sub minus 1 non-target
    classes at random and appends the target. A figure with 100 legend entries
    reads as noise, which is what the cap is for.
    """
    every_class = np.arange(num_classes)
    if num_classes <= count:
        return every_class

    generator = np.random.default_rng(seed)
    others = np.delete(every_class, target_label)
    drawn = generator.choice(others, count - 1, replace=False)

    selected = np.append(drawn, target_label)
    return selected


def equal_ratio_class_subset(
    labels: np.ndarray,
    max_num_samples: int | None,
    selected_classes: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Row indices sampled at 1 common ratio inside every selected class, then shuffled.

    A transcription of sub_sample_euqal_ratio_classes_index (visual_utils.py
    lines 901 to 926) onto an explicit generator. The ratio is the cap divided by
    the rows the selected classes hold, so the class proportions of the pool
    survive into the subset, and each class draws int(ratio * count) rows, which
    rounds down and can leave the subset a few rows under the cap.
    """
    generator = np.random.default_rng(seed)
    present = np.unique(labels)
    if selected_classes is not None:
        present = np.intersect1d(present, selected_classes, assume_unique=True)
    if present.size == 0:
        raise ValueError("none of the selected classes has any row")

    total_selected = int(sum(int((labels == c).sum()) for c in present))
    ratio = 1.0
    if max_num_samples is not None:
        ratio = min(total_selected, max_num_samples) / total_selected

    drawn = []
    for c in present:
        rows_of_class = np.flatnonzero(labels == c)
        drawn.append(
            generator.choice(
                rows_of_class, int(ratio * rows_of_class.shape[0]), replace=False
            )
        )
    concatenated = np.concatenate(drawn).reshape(-1)

    shuffled = concatenated[generator.permutation(concatenated.shape[0])]
    return shuffled


def class_subset(
    view: SampleView,
    max_num_samples: int | None,
    selected_classes: np.ndarray | None,
    seed: int,
) -> SampleView:
    """The view cut to the selected classes at equal ratio, poisoned rows as a class of their own.

    generate_bd_dataset and generate_mix_dataset relabel poisoned rows to -1
    before subsetting and add -1 to the selected classes, so the poisoned rows
    are sampled at the same ratio as every real class rather than filtered by
    their true class.
    """
    labels_with_poison = view.labels.numpy().copy()
    labels_with_poison[view.poison_mask.numpy()] = POISON_CLASS_MARK
    classes = (
        None
        if selected_classes is None
        else np.append(selected_classes, POISON_CLASS_MARK)
    )

    rows = equal_ratio_class_subset(labels_with_poison, max_num_samples, classes, seed)
    subset = view.subset(torch.as_tensor(rows, dtype=torch.long))
    return subset


def view_samples(
    loaders: dict[str, DataLoader],
    manifest: dict,
    view: str,
    max_num_samples: int | None,
    selected_classes: np.ndarray | None,
    mixed_ratio: float,
    seed: int,
) -> SampleView:
    """The named view, subset to the selected classes at equal ratio.

    clean_test is the clean split, bd_test the triggered rows, mixed the clean
    split with mixed_ratio of its eligible rows triggered and paired both
    populations stacked. Every view carries true labels and a poison mask.
    """
    clean, backdoor = paired_views(loaders, manifest)
    whole_clean = clean_view(loaders, manifest)

    if view == "clean_test":
        population = whole_clean
    elif view == "bd_test":
        population = backdoor
    elif view == "mixed":
        population = mixed_view(clean, backdoor, whole_clean, mixed_ratio, seed)
    elif view == "paired":
        population = union_view(clean, backdoor)
    else:
        raise ValueError(f"unknown view {view!r}")

    subset = class_subset(population, max_num_samples, selected_classes, seed)
    return subset


def paired_examples(
    clean: SampleView,
    backdoor: SampleView,
    num_clean: int,
    num_poisoned: int,
    seed: int,
) -> SampleView:
    """num_clean clean rows then num_poisoned triggered rows of the SAME images, for an image panel.

    Upstream draws its 2 clean and 2 poisoned examples independently and notes
    that drawing the same indices "builds the correspondence" between them.
    That correspondence is what a backdoor panel is for, so the same rows are
    used here: panel k of the clean half and panel k of the poisoned half show 1
    image with and without its trigger. clean and backdoor are the paired
    views, row for row. When the pool holds fewer rows than the panels ask
    for, fewer panels come back, and the returned poison mask says which of
    them carry the trigger, so a caller reads the mask and never the position.
    """
    if len(clean) != len(backdoor):
        raise ValueError(
            f"paired examples need the paired views, got {len(clean)} clean and "
            f"{len(backdoor)} triggered rows"
        )
    generator = np.random.default_rng(seed)
    wanted = max(num_clean, num_poisoned)
    drawn = generator.choice(len(clean), min(wanted, len(clean)), replace=False)
    rows = torch.as_tensor(drawn, dtype=torch.long)

    clean_part = clean.subset(rows[:num_clean])
    poisoned_part = backdoor.subset(rows[:num_poisoned])
    examples = SampleView(
        images=torch.cat([clean_part.images, poisoned_part.images]),
        labels=torch.cat([clean_part.labels, poisoned_part.labels]),
        poison_mask=torch.cat([clean_part.poison_mask, poisoned_part.poison_mask]),
        test_indices=torch.cat([clean_part.test_indices, poisoned_part.test_indices]),
    )
    return examples


def to_pixels(
    images: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]
) -> torch.Tensor:
    """Normalised images back in [0, 1] pixel space, (N, C, H, W), for drawing."""
    mean_tensor, std_tensor = normalization_buffers(
        mean, std, images.device, images.dtype
    )  # (1, C, 1, 1) each

    pixels = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)  # (N, C, H, W)
    return pixels


def to_display(pixels: torch.Tensor) -> np.ndarray:
    """(N, C, H, W) pixels as the (N, H, W, C) float array imshow draws."""
    display = pixels.detach().cpu().permute(0, 2, 3, 1).numpy()  # (N, H, W, C)
    return display
