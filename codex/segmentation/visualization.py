from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from segmentation.segmentation import scale_marker_sum


def plot_cropped_subregion(
    im: np.ndarray,
    x_mid: int,
    y_mid: int,
    length: int,
    figsize: tuple[int, int] = (10, 10),
    cmap: Optional[str] = None,
) -> None:
    """
    Plots an image with a highlighted subregion and the cropped subregion side-by-side.

    Args:
        im (np.ndarray): 
            The input image as a NumPy array.
        x_mid (int): 
            The x-coordinate of the center of the subregion to be cropped.
        y_mid (int): 
            The y-coordinate of the center of the subregion to be cropped.
        length (int): 
            The length of the square subregion to be cropped.
        figsize (tuple[int, int], optional): 
            The size of the figure to be displayed in inches. Default is (10, 10).
        cmap (Optional[str], optional): 
            The colormap for displaying the image. If None, the default colormap is used. 
            Default is None.

    Returns:
        None: Displays the plot with the original image and the cropped subregion.
    """
    # Extract the subregion of the image centered at (x_mid, y_mid) with the specified length
    im_sm = im[
        int(y_mid - length / 2) : int(y_mid + length / 2),
        int(x_mid - length / 2) : int(x_mid + length / 2),
    ]

    fig = plt.figure(figsize=figsize)

    # Plot 1: Original Image with a rectangle highlighting the subregion
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im, cmap=cmap)
    rect = patches.Rectangle(
        (x_mid - length / 2, y_mid - length / 2),
        length,
        length,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title("Original Image with Subregion Highlighted")

    # Plot 2: Cropped subregion of the original image
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(im_sm, cmap=cmap)
    ax.set_title("Cropped Subregion Image")

    plt.tight_layout()
    plt.show()


def plot_scale_marker_sum(
    marker_lists: list[list[str]],
    title_list: list[str],
    marker_dict: dict[str : np.ndarray],
    scale: bool = True,
    ncol: int = 2,
    vmax: Optional[float] = None,
    figsize: tuple[int, int] = (5, 5),
) -> None:
    """
    Plots the sum of scaled markers across multiple marker lists.

    This function generates a grid of subplots where each subplot visualizes the sum
    of markers provided in the input lists. The markers can be optionally scaled,
    and the colormap range can be customized with `vmax`.

    Args:
        marker_lists (list[list[str]]):
            A list of marker lists, where each inner list contains marker names to be summed.
            Each marker list corresponds to a single subplot.

        title_list (list[str]):
            A list of titles for each subplot, corresponding to the marker lists.

        marker_dict (dict[str, np.ndarray]):
            A dictionary mapping marker names to their corresponding image data (np.ndarray).

        scale (bool, optional):
            If True, scales the marker values to a common range. Defaults to True.

        ncol (int, optional):
            Number of columns in the subplot grid. The number of rows is calculated based on
            the total number of marker lists. Defaults to 2.

        vmax (float, optional):
            The maximum value for scaling the colormap in `imshow`. If None, the maximum value
            from the data is used. Defaults to None.

        figsize (tuple[int, int], optional):
            Size of each subplot in inches (width, height). The overall figure size is determined
            by multiplying the number of rows and columns by these values. Defaults to (5, 5).

    Returns:
        None: This function only displays the plot and does not return any value.
    """
    im_list = [scale_marker_sum(marker_list, marker_dict, scale=scale) for marker_list in marker_lists]
    nrow = int(np.ceil(len(im_list) / ncol))
    fig, axs = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(ncol * figsize[0], nrow * figsize[1]),
    )
    axs = axs.flatten()
    for i in range(len(im_list)):
        axs[i].imshow(im_list[i], vmax=vmax)
        axs[i].set_title(title_list[i])
        axs[i].axis("off")
    for j in range(len(im_list), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_scale_marker_sum_segmentation(
    marker_lists: list[list[str]],
    title_list: list[str],
    marker_dict: dict[str, np.ndarray],
    mask_dict: dict[str, np.ndarray],
    scale: bool = True,
    vmax: Optional[float] = None,
    alpha: float = 0.75,
    figsize: tuple[int, int] = (5, 5),
) -> None:
    """
    Plots scaled marker sums with segmentation masks and overlays.

    Args:
        marker_lists (list[list[str]]):
            A list of marker lists, where each inner list contains marker names to be summed.
        title_list (list[str]):
            A list of titles for each subplot, corresponding to the marker lists.
        marker_dict (dict[str, np.ndarray]):
            A dictionary mapping marker names to their corresponding image data.
        mask_dict (dict[str, np.ndarray]):
            A dictionary containing segmentation masks and overlays.
            Should contain keys `"segmentation_mask"` and `"overlay"`.
        scale (bool, optional):
            If True, scales marker values. Defaults to True.
        vmax (Optional[float], optional):
            The maximum value for colormap scaling. If None, the colormap will scale automatically.
            Defaults to None.
        alpha (float, optional):
            Transparency of the segmentation contour overlay. Defaults to 0.75.
        figsize (tuple[int, int], optional):
            The size of each subplot in inches (width, height). Defaults to (5, 5).

    Returns:
        None: Displays the plot with the original image, segmentation overlays, and cropped regions.

    """

    im_list = [scale_marker_sum(marker_list, marker_dict, scale=scale) for marker_list in marker_lists]
    ncol = len(im_list)

    def _im_in_out_segmentation(
        marker_im: np.ndarray, segmentation_mask: np.ndarray, overlay: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates images inside and outside the segmentation mask and an overlay with contours.

        Args:
            marker_im (np.ndarray):
                The input marker image.
            segmentation_mask (np.ndarray):
                Segmentation mask for the single cells.
            overlay (np.ndarray):
                Overlay for drawing segmentation contours.

        Returns:
            tuple: A tuple containing:
                - marker_im_in (np.ndarray): Marker image inside the segmentation mask.
                - marker_im_out (np.ndarray): Marker image outside the segmentation mask.
                - argb_overlay (np.ndarray): ARGB image with white contours and transparent background.
        """
        marker_im_in = marker_im.copy()
        marker_im_in[segmentation_mask == 0] = 0
        marker_im_out = marker_im.copy()
        marker_im_out[segmentation_mask != 0] = 0

        argb_overlay = np.zeros((overlay.shape[0], overlay.shape[1], 4), dtype=np.uint8)
        argb_overlay[:, :, 0:3] = 255  # All white in RGB channels
        argb_overlay[:, :, 3] = np.where(overlay[:, :, 0] == 1, 255, 0)  # Set alpha channel
        return marker_im_in, marker_im_out, argb_overlay

    nrow = 4
    fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol * figsize[0], nrow * figsize[1]))
    # Add Y-axis labels for each row
    y_labels = [
        "Raw",
        "Inside Segmentation Mask",
        "Segmentation Contour",
        "Outside Segmentation Mask",
    ]
    for row, label in enumerate(y_labels):
        axs[row, 0].set_ylabel(label, rotation=90, fontsize=12, ha="center")

    for i, marker_im in enumerate(im_list):
        marker_im_in, marker_im_out, argb_overlay = _im_in_out_segmentation(
            marker_im, mask_dict["segmentation_mask"], mask_dict["overlay"]
        )
        axs[0, i].imshow(marker_im)
        axs[0, i].set_title(title_list[i])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

        axs[1, i].imshow(marker_im_in)
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

        axs[2, i].imshow(marker_im)
        axs[2, i].imshow(argb_overlay, alpha=alpha)
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])

        axs[3, i].imshow(marker_im_out)
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])

    plt.tight_layout()
    plt.show()
