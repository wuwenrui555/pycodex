import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from tqdm import tqdm

from .segmentation_mask import find_label_boundaries

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def create_rgb_annotation(
    segmentation_mask: np.ndarray,
    annotation_dict: dict[int, str],
    color_dict: dict[str, str],
    outline: bool = False,
    color_bg: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Create an RGB image for segmentation annotations.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        2D array representing the segmentation mask.
    annotation_dict : dict[int, str]
        A dictionary mapping label integers to annotation names.
    color_dict : dict[str, str]
        A dictionary mapping annotation names to colors.
    outline : bool, optional
        If True, outlines the segmentation mask in white, by default False.
    color_bg : tuple[int, int, int], optional
        Background color for the RGB image, by default (255, 255, 255).

    Returns
    -------
    np.ndarray
        RGB image of annotations and outline (if enabled).
    """
    # Map label integers to RGB colors based on annotations
    unique_labels = np.unique(segmentation_mask[segmentation_mask != 0])
    color_mapping = np.zeros((np.max(unique_labels) + 1, 3), dtype=np.uint8)
    for label in tqdm(
        unique_labels, desc="Mapping labels to colors", bar_format=TQDM_FORMAT
    ):
        annotation = annotation_dict.get(label, None)
        color_hex = color_dict.get(annotation, None)
        if color_hex is not None:
            color_rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))
            color_mapping[label] = color_rgb

    # Create and fill RGB image with colors
    rgb_image = np.zeros(
        (segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8
    )
    rgb_image = color_mapping[segmentation_mask]
    rgb_image[segmentation_mask == 0] = color_bg

    # Add cell boundaries if outline option is enabled
    if outline:
        segmentation_boundary = find_label_boundaries(segmentation_mask)
        rgb_image[segmentation_boundary.astype(bool)] = color_bg

    return rgb_image


def add_legend(
    color_dict: dict[str, str],
    ax: Axes = None,
    loc: str = "center left",
    bbox_to_anchor: tuple[float, float] = (1, 0.5),
    fontsize: int = 8,
) -> Axes | None:
    """
    Add a legend to the given axes with colors from the color dictionary.

    Parameters
    ----------
    color_dict : dict[str, str]
        A dictionary mapping annotation names to colors.
    ax : matplotlib.axes.Axes, optional
        The axes to add the legend to. If None, uses the current axes.
    loc : str, optional
        Location of the legend. Default is "center left".
    bbox_to_anchor : tuple[float, float], optional
        Bounding box to anchor the legend. Default is (1, 0.5).
    fontsize : int, optional
        Font size of the legend text. Default is 8.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the legend added.
    """
    if ax is None:
        ax = plt.gca()

    legend_handles = []
    legend_labels = []
    for label, color in color_dict.items():
        patch = Patch(color=color, label=label)
        legend_handles.append(patch)
        legend_labels.append(label)

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
    )

    return ax
