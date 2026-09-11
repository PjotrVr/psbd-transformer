"""Line-for-line transcriptions of the BackdoorBench formula lines the tests compare against.

BackdoorBench (Wu et al., NeurIPS 2022 Datasets and Benchmarks track,
https://github.com/SCLBD/BackdoorBench, commit f02e353) is licensed under the
Creative Commons Attribution-NonCommercial 4.0 International licence. The
lines below are quoted under that licence as test-only reference material,
attributed to its authors, and nothing here is imported by analysis/,
visualization/ or evaluation/, which re-implement the statistics from the
papers and are compared against these quotations in tests/ alone. Each
function names the upstream file and the lines it transcribes. Variable names,
operation order and constants are kept as upstream wrote them, so a test that
passes against a function here passes against BackdoorBench itself.
"""

import numpy as np
import torch
from sklearn.manifold import TSNE


def reduce_flatten(output_: torch.Tensor) -> torch.Tensor:
    """analysis/visual_utils.py line 415, get_features with reduction='flatten'."""
    feature_vector = output_
    feature_vector = torch.flatten(feature_vector, 1)
    return feature_vector


def reduce_sum(output_: torch.Tensor) -> torch.Tensor:
    """analysis/visual_utils.py lines 443 to 449, get_features with reduction='sum'."""
    feature_vector = output_
    if feature_vector.dim() > 2:
        feature_vector = torch.sum(torch.flatten(feature_vector, 2), 2)
    else:
        feature_vector = feature_vector
    return feature_vector


def tac(features_bd: np.ndarray, features_bd_clean: np.ndarray) -> np.ndarray:
    """analysis/visual_tac.py lines 141 to 150, the per-neuron TAC of 1 module."""
    total_neuron = features_bd.shape[1]
    feature_diff = features_bd - features_bd_clean
    np.abs(feature_diff, out=feature_diff)  # inplace abs for faster computation
    feature_abs_diff = feature_diff
    # average over batch
    feature_abs_diff_mean = np.mean(feature_abs_diff, axis=0)
    # average for each channel
    if feature_abs_diff_mean.ndim > 1:
        feature_abs_diff_mean = np.sum(
            feature_abs_diff_mean.reshape(total_neuron, -1), axis=1
        )
    return feature_abs_diff_mean


def neuron_activation_average(features: np.ndarray) -> np.ndarray:
    """analysis/visual_na.py line 130 and 138, the bar heights."""
    features_avg = np.mean(features, axis=0)
    return features_avg


def neuron_activation_order(features_clean_avg: np.ndarray) -> np.ndarray:
    """analysis/visual_na.py line 140, the bar order."""
    sort_bar = np.argsort(features_clean_avg)[::-1]
    return sort_bar


def top_indx(features: np.ndarray) -> np.ndarray:
    """analysis/visual_act.py line 150, the top activating image per neuron."""
    top_indx = np.argsort(-features, axis=0)
    return top_indx


def num_image_rule(
    len_visual_dataset: int, len_selected_classes: int, poi_indicator: np.ndarray
) -> int:
    """analysis/visual_actdist.py lines 147 to 149, how many top images are counted."""
    num_image = int(len_visual_dataset / len_selected_classes)
    if poi_indicator.sum() > 0:
        num_image = poi_indicator.sum()
    return int(num_image)


def activation_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    poi_indicator: np.ndarray,
    num_classes: int,
    num_image: int,
) -> tuple[np.ndarray, np.ndarray]:
    """analysis/visual_actdist.py lines 151 to 181, the per-neuron class shares.

    Returns (label_set, percent) with percent shaped (total_neuron, len(label_set)).
    """
    labels = np.array(labels).copy()
    labels[poi_indicator == 1] = num_classes
    label_set = np.unique(labels)
    label_set.sort()

    total_neuron = features.shape[1]
    top_indx = np.argsort(-features, axis=0)[:num_image, :]
    top_pred = np.array(labels)[top_indx]

    percent = np.zeros((total_neuron, len(label_set)))
    for neuron_i in range(total_neuron):
        for i in range(len(label_set)):
            percent[neuron_i, i] = (
                np.sum(top_pred[:, neuron_i] == label_set[i]) / num_image
            )
    return label_set, percent


def saliency(input: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """analysis/visual_utils.py lines 1000 to 1018, the frequency saliency map.

    Mutates its arguments exactly as upstream does: the model's parameters lose
    requires_grad and input is unsqueezed in place.
    """
    for param in model.parameters():
        param.requires_grad = False
    input.unsqueeze_(0)
    input.requires_grad = True
    preds = model(input)
    score, indices = torch.max(preds, 1)
    score.backward()
    gradients = input.grad.data.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    gradients_fre = np.fft.ifft2(gradients, axes=(0, 1))

    gradients_fre_shift = np.fft.fftshift(gradients_fre, axes=(0, 1))
    gradients_fre_shift = np.log(np.abs(gradients_fre_shift))

    gradient_norm = (gradients_fre_shift - gradients_fre_shift.min()) / (
        gradients_fre_shift.max() - gradients_fre_shift.min()
    )
    gradient_norm = np.mean(gradient_norm, axis=2)
    gradient_norm = np.uint8(255 * gradient_norm)
    return gradient_norm


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list, normalize: bool
) -> np.ndarray:
    """analysis/visual_utils.py lines 627 to 635, plot_confusion_matrix's matrix."""
    cm = np.zeros((len(classes), len(classes)))
    for i in range(y_true.shape[0]):
        cm[y_true[i], y_pred[i]] += 1

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-24)
    else:
        cm = cm.astype("int")
    return cm


def plot_embedding_scaling(tsne_result: np.ndarray) -> np.ndarray:
    """analysis/visual_utils.py lines 496 and 497, the unit square scaling."""
    x_min, x_max = np.min(tsne_result, 0), np.max(tsne_result, 0)
    tsne_result = (tsne_result - x_min) / (x_max - x_min)
    return tsne_result


def get_embedding_tsne(data: np.ndarray) -> np.ndarray:
    """analysis/visual_utils.py lines 550 and 551, the t-SNE call with its fixed seed."""
    tsne = TSNE(n_components=2, init="random", random_state=0)
    result = tsne.fit_transform(data)
    return result


def sub_sample_euqal_ratio_classes_index(
    y: np.ndarray,
    ratio: float | None = None,
    selected_classes: np.ndarray | None = None,
    max_num_samples: int | None = None,
) -> np.ndarray:
    """analysis/visual_utils.py lines 901 to 926, the equal-ratio class subset.

    Draws from numpy's global generator as upstream does, so a test seeds
    np.random before calling it and checks per-class counts rather than rows.
    """
    class_unique = np.unique(y)
    if selected_classes is not None:
        class_unique = np.intersect1d(
            class_unique, selected_classes, assume_unique=True, return_indices=False
        )
    select_idx = []
    if max_num_samples is not None:
        total_selected_samples = np.sum(
            [np.where(y == c_idx)[0].shape[0] for c_idx in class_unique]
        )
        ratio = (
            np.min([total_selected_samples, max_num_samples]) / total_selected_samples
        )

    for c_idx in class_unique:
        sub_idx = np.where(y == c_idx)
        sub_idx = np.random.choice(
            sub_idx[0], int(ratio * sub_idx[0].shape[0]), replace=False
        )
        select_idx.append(sub_idx)
    sub_idx = np.concatenate(select_idx, -1).reshape(-1)
    sub_idx = sub_idx[np.random.permutation(sub_idx.shape[0])]
    return sub_idx


def defense_effectiveness_rate_simplied(acc_bd, acc_defnese, asr_bd, asr_defense):
    """utils/metric.py lines 92 to 94, the DER visual_metric.py calls."""
    return (max(0, asr_bd - asr_defense) - max(0, acc_bd - acc_defnese) + 1) / 2


def robust_improvement_rate_simplied(acc_bd, acc_defnese, ra_bd, ra_defense):
    """utils/metric.py lines 96 to 98, the RIR visual_metric.py calls."""
    return (max(0, -ra_bd + ra_defense) - max(0, acc_bd - acc_defnese) + 1) / 2
