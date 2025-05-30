import numpy as np
from matplotlib.axes import Axes
from skimage.segmentation import find_boundaries

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def find_label_boundaries(
    segmentation_mask: np.ndarray,
):
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
    ax: Axes,
    color: str = "white",
    fontsize: int = 8,
    ha: str = "center",
    va: str = "center",
    bbox: dict = dict(facecolor="black", alpha=0.5, edgecolor="none"),
):
    """
    Plot labels from a segmentation mask.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        An array in which different regions are labeled with different integers.
    ax : matplotlib.axes.Axes
        The axes on which to plot the labels.
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
    """

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
