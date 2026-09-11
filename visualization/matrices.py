"""The confusion matrix figure."""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from .style import CONFUSION_CMAP, SINGLE_COLUMN, paper_style

# Cell text is drawn only below this many classes, past which the numbers
# overlap and hide the colour they annotate.
ANNOTATION_CLASS_LIMIT = 20


def confusion_matrix_figure(
    matrix: np.ndarray, normalised: bool, title: str
) -> matplotlib.figure.Figure:
    """Blues imshow of a (K, K) confusion matrix, true class down, predicted across."""
    counts = np.asarray(matrix, dtype=np.float64)  # (K, K)
    num_classes = counts.shape[0]
    side = (
        SINGLE_COLUMN if num_classes <= ANNOTATION_CLASS_LIMIT else SINGLE_COLUMN * 1.6
    )

    with paper_style():
        figure, axis = plt.subplots(figsize=(side, side))
        image = axis.imshow(counts, interpolation="nearest", cmap=CONFUSION_CMAP)
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
        ticks = np.arange(num_classes)
        step = max(1, num_classes // 20)
        axis.set_xticks(
            ticks[::step],
            [str(t) for t in ticks[::step]],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(ticks[::step], [str(t) for t in ticks[::step]])
        axis.set_xlabel("predicted label")
        axis.set_ylabel("true label")
        axis.set_title(title)
        if num_classes <= ANNOTATION_CLASS_LIMIT:
            text_format = ".2f" if normalised else "d"
            threshold = counts.max() / 2.0
            for row in range(num_classes):
                for column in range(num_classes):
                    value = (
                        counts[row, column] if normalised else int(counts[row, column])
                    )
                    axis.text(
                        column,
                        row,
                        format(value, text_format),
                        ha="center",
                        va="center",
                        color="white" if counts[row, column] > threshold else "black",
                    )
        axis.set_xlim(-0.5, num_classes - 0.5)
        axis.set_ylim(num_classes - 0.5, -0.5)
    return figure
