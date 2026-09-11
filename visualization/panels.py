"""Image panels: Grad-CAM overlays, frequency maps, attribution rows and top activating images.

Every builder takes pixels as (num_images, height, width, channels) float
arrays in [0, 1], the layout imshow draws, and returns a Figure the caller
saves. The overlay follows pytorch_grad_cam's show_cam_on_image, a jet colour
map blended half and half with the image and rescaled by its maximum, which is
what upstream draws.
"""

import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    ATTRIBUTION_CMAP,
    DOUBLE_COLUMN,
    FONT_SIZE,
    FREQUENCY_CMAP,
    OVERLAY_CMAP,
    paper_style,
)

# show_cam_on_image's image_weight: the overlay is half image and half heat map.
OVERLAY_IMAGE_WEIGHT = 0.5

# shap.image_plot clips the colour scale at this percentile of |attribution|,
# so a single extreme pixel does not wash out every other one.
ATTRIBUTION_CLIP_PERCENTILE = 99.9


def overlay_heat_map(pixels: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """Jet-coloured heat blended onto the image, (N, H, W, 3) in [0, 1].

    pixels is (N, H, W, 3) and heat (N, H, W) in [0, 1]. Blended half and half
    then divided by the per-image maximum, as show_cam_on_image does.
    """
    jet = plt.get_cmap(OVERLAY_CMAP)
    coloured = jet(heat)[..., :3]  # (N, H, W, 3)
    blended = (1 - OVERLAY_IMAGE_WEIGHT) * coloured + OVERLAY_IMAGE_WEIGHT * pixels
    peak = blended.reshape(blended.shape[0], -1).max(axis=1)[:, None, None, None]

    overlay = blended / np.where(peak > 0, peak, 1.0)  # (N, H, W, 3)
    return overlay


def paired_image_panels(
    pixels: np.ndarray,
    companions: np.ndarray,
    image_titles: list[str],
    companion_titles: list[str],
    companion_kind: str,
) -> matplotlib.figure.Figure:
    """Upstream's 2 by 4 layout: image, companion, image, companion on each of 2 rows.

    companion_kind "rgb" draws companions as (N, H, W, 3) images, the Grad-CAM
    overlay. companion_kind "frequency" draws them as (N, H, W) maps on the
    coolwarm scale from 0 to 255 with a colour bar per panel, the frequency
    saliency layout.
    """
    num_images = pixels.shape[0]
    if num_images != 4:
        raise ValueError(f"the panel is laid out for 4 images, got {num_images}")

    with paper_style():
        figure, axes = plt.subplots(2, 4, figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.55))
        for index in range(num_images):
            image_axis = axes[index // 2, index % 2 * 2]
            companion_axis = axes[index // 2, index % 2 * 2 + 1]
            image_axis.imshow(pixels[index])
            image_axis.axis("off")
            image_axis.set_title(image_titles[index])
            if companion_kind == "rgb":
                companion_axis.imshow(companions[index])
            elif companion_kind == "frequency":
                drawn = companion_axis.imshow(
                    companions[index],
                    cmap=FREQUENCY_CMAP,
                    norm=matplotlib.colors.Normalize(vmin=0, vmax=255),
                )
                figure.colorbar(
                    drawn,
                    ax=companion_axis,
                    orientation="vertical",
                    fraction=0.046,
                    pad=0.04,
                )
            else:
                raise ValueError(f"unknown companion kind {companion_kind!r}")
            companion_axis.axis("off")
            companion_axis.set_title(companion_titles[index])
    return figure


def attribution_rows(
    pixels: np.ndarray,
    attributions: np.ndarray,
    class_titles: list[list[str]],
    image_titles: list[str],
) -> matplotlib.figure.Figure:
    """1 row per image: the image, then its attribution map per explained class.

    attributions is (N, ranked, H, W), already summed over channels as
    shap.image_plot sums them, drawn on 1 symmetric red-blue scale clipped at
    the ATTRIBUTION_CLIP_PERCENTILE of the absolute values over the figure.
    """
    num_images, ranked = attributions.shape[:2]
    limit = (
        float(np.nanpercentile(np.abs(attributions), ATTRIBUTION_CLIP_PERCENTILE))
        or 1.0
    )

    with paper_style():
        figure, axes = plt.subplots(
            num_images,
            ranked + 1,
            figsize=(1.6 * (ranked + 1), 1.7 * num_images),
            squeeze=False,
        )
        for row in range(num_images):
            axes[row, 0].imshow(pixels[row])
            axes[row, 0].set_title(image_titles[row])
            axes[row, 0].axis("off")
            for rank in range(ranked):
                axis = axes[row, rank + 1]
                axis.imshow(pixels[row].mean(axis=2), cmap="gray", alpha=0.15)
                drawn = axis.imshow(
                    attributions[row, rank],
                    cmap=ATTRIBUTION_CMAP,
                    vmin=-limit,
                    vmax=limit,
                    alpha=0.9,
                )
                axis.set_title(class_titles[row][rank])
                axis.axis("off")
        figure.colorbar(
            drawn,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.03,
            pad=0.04,
            label="attribution, summed over channels",
        )
    return figure


def activating_image_grid(
    pixels: np.ndarray,
    dimensions: np.ndarray,
    values: np.ndarray,
    poisoned: np.ndarray,
    title: str,
) -> matplotlib.figure.Figure:
    """The top images of each dimension, 1 row per dimension, red titles on poisoned rows.

    pixels is (num_dims, k, H, W, 3), values (num_dims, k) the activation that
    ranked each image and poisoned (num_dims, k) bool, read from the view's
    poison mask and never from position.
    """
    num_dims, k = values.shape

    with paper_style():
        figure, axes = plt.subplots(
            num_dims,
            k,
            figsize=(0.9 * k, 1.0 * num_dims + 0.4),
            squeeze=False,
            layout="constrained",
        )
        figure.suptitle(title)
        for row in range(num_dims):
            for column in range(k):
                axis = axes[row, column]
                axis.imshow(pixels[row, column])
                axis.set_xticks([])
                axis.set_yticks([])
                # The column is the rank, so the title carries only the
                # dimension and the activation that ranked the image.
                axis.set_title(
                    f"d{int(dimensions[row])}: {values[row, column]:.0f}",
                    color="red" if poisoned[row, column] else "black",
                    pad=2,
                    fontsize=FONT_SIZE - 2,
                )
    return figure
