import matplotlib.pyplot as plt
import numpy as np
from deepcell.utils.plot_utils import create_rgb_image
from matplotlib.axes import Axes
from skimage.segmentation import find_boundaries

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def find_label_boundaries(
    segmentation_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert a segmentation mask to a mask for boundaries.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        An array in which different regions are labeled with different integers.

    Returns
    -------
    np.ndarray
        An array with same shape as `segmentation_mask` and same dtype, where
        values of 0 represent background pixels and other values represent
        the boundaries of labels.
    """
    boundaries = find_boundaries(
        segmentation_mask, mode="inner", connectivity=1, background=0
    )
    segmentation_boundary = segmentation_mask * boundaries

    return segmentation_boundary


def plot_labels(
    segmentation_mask: np.ndarray,
    ax: Axes = None,
    color: str = "white",
    fontsize: int = 8,
    ha: str = "center",
    va: str = "center",
    bbox: dict = dict(facecolor="black", alpha=0.5, edgecolor="none"),
) -> Axes:
    """
    Plot labels from a segmentation mask.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        An array in which different regions are labeled with different integers.
    ax : matplotlib.axes.Axes
        The axes on which to plot the labels. If None, uses the current axes.
    color : str, optional
        Color of the text labels. Default is "white".
    fontsize : int, optional
        Font size of the text labels. Default is 8.
    ha : str, optional
        Horizontal alignment of the text labels. Default is "center".
    va : str, optional
        Vertical alignment of the text labels. Default is "center".
    bbox : dict, optional
        A dictionary defining the bounding box properties for the text labels.
        Default is a semi-transparent black box with no edge color.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the labels plotted.
    """
    if ax is None:
        ax = plt.gca()

    labels = np.unique(segmentation_mask)
    labels = labels[labels != 0]
    for label in labels:
        y_indices, x_indices = np.where(segmentation_mask == label)
        x_mid = (x_indices.min() + x_indices.max()) // 2
        y_mid = (y_indices.min() + y_indices.max()) // 2
        ax.text(
            x_mid,
            y_mid,
            str(label),
            color=color,
            fontsize=fontsize,
            ha=ha,
            va=va,
            bbox=bbox,
        )

    return ax


def create_rgb_segmentation_mask(
    internal_channel: np.ndarray,
    boundary_channel: np.ndarray,
    internal_color: str = "blue",
    boundary_color: str = "green",
    outline: bool = False,
    segmentation_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Create an RGB image from internal and boundary channels.

    Parameters
    ----------
    internal_channel : np.ndarray
        2D array representing the internal channel.
    boundary_channel : np.ndarray
        2D array representing the boundary channel.
    internal_color : str, optional
        Color for the internal channel, by default "blue".
    boundary_color : str, optional
        Color for the boundary channel, by default "green".
    outline : bool, optional
        If True, outlines the segmentation mask in white, by default False.
    segmentation_mask : np.ndarray, optional
        2D array representing the segmentation mask, required if outline is True.

    Returns
    -------
    np.ndarray
        RGB image of internal, boundary channels and outline.
    """
    image_stack = np.stack((internal_channel, boundary_channel), axis=-1)
    image_stack = np.expand_dims(image_stack, 0)
    rgb_image = create_rgb_image(
        image_stack, channel_colors=[internal_color, boundary_color]
    )
    rgb_image = rgb_image[0]

    if outline:
        if segmentation_mask is None:
            raise ValueError("Segmentation mask is None, cannot plot outline.")
        segmentation_boundary = find_label_boundaries(segmentation_mask)
        rgb_image[0, segmentation_boundary > 0] = 1

    return rgb_image
