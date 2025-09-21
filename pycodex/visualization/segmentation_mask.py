# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepcell.utils.plot_utils import create_rgb_image
from matplotlib.axes import Axes
from skimage.segmentation import find_boundaries

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


################################################################################
# segmentation mask
################################################################################
def find_segmentation_boundaries(
    segmentation_mask: np.ndarray,
    mode: str = "inner",
    connectivity: int = 1,
    background: int = 0,
) -> np.ndarray:
    """
    Convert a segmentation mask to a mask for boundaries.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        An array in which different regions are labeled with different integers.
    mode : str, optional
        The mode of boundary detection. Options are 'inner', 'outer', 'thick',
        and 'subpixel'. Default is 'inner'.
    connectivity : int, optional
        The connectivity defining the neighborhood of a pixel. Default is 1.
    background : int, optional
        The value representing the background in the segmentation mask. Default
        is 0.

    Returns
    -------
    np.ndarray
        An array with same shape as `segmentation_mask` and same dtype, where
        values of 0 represent background pixels and other values represent
        the boundaries of labels.
    """
    boundaries = find_boundaries(
        segmentation_mask,
        mode=mode,
        connectivity=connectivity,
        background=background,
    )
    segmentation_boundary = segmentation_mask * boundaries

    return segmentation_boundary


def ax_plot_segmentation_labels(
    segmentation_mask: np.ndarray,
    ax: Axes = None,
    color: str = "white",
    fontsize: int = 12,
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
        Font size of the text labels. Default is 12.
    ha : str, optional
        Horizontal alignment of the text labels. Default is "center".
    va : str, optional
        Vertical alignment of the text labels. Default is "center".
    bbox : dict, optional
        A dictionary defining the bounding box properties for the text labels.
        Default is a semi-transparent black box with no edge color.
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
        segmentation_boundary = find_segmentation_boundaries(segmentation_mask)
        rgb_image[segmentation_boundary > 0] = 1

    return rgb_image


def create_rgb_with_segmentation_boundaries(
    img_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    boundary_color_rgb: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    Create an RGB image with segmentation boundaries overlaid.

    Parameters
    ----------
    image : np.ndarray
        2D array representing the grayscale image.
    segmentation_mask : np.ndarray
        2D array representing the segmentation mask.
    outline_color : str, optional
        Color for the segmentation boundaries, by default "white".
    outline : bool, optional
        If True, outlines the segmentation mask in the specified color, by default True.

    Returns
    -------
    np.ndarray
        RGB image with segmentation boundaries overlaid.
    """
    segmentation_boundaries = find_segmentation_boundaries(segmentation_mask) > 0

    img_rgb_boundary = img_rgb.copy()
    img_rgb_boundary[segmentation_boundaries] = boundary_color_rgb

    return img_rgb_boundary


################################################################################
# annotation
################################################################################


def create_rgb_annotation(
    segmentation_mask: np.ndarray,
    annotation_dict: dict[int, str],
    color_dict: dict[str, str],
    boundary: bool = False,
    boundary_color_rgb: tuple[int, int, int] = (0, 0, 0),
    background_color_rgb: tuple[int, int, int] = (0, 0, 0),
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
    # Map annotations to colors
    unique_annotations = list(set(annotation_dict.values()))
    color_mapping = np.zeros((len(unique_annotations) + 1, 3), dtype=np.uint8)
    annotation_mask = np.zeros_like(segmentation_mask, dtype=np.int32)

    # Create Annotation mask
    annotation_df = pd.DataFrame(
        {"cell_label": annotation_dict.keys(), "annotation": annotation_dict.values()}
    )
    for i, annotation in enumerate(unique_annotations):
        color_hex = color_dict.get(annotation, None)
        if color_hex is not None:
            color_rgb = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))
            color_mapping[i + 1] = color_rgb

            cell_labels = annotation_df.loc[
                annotation_df["annotation"] == annotation, "cell_label"
            ].values
            if len(cell_labels) > 0:
                annotation_mask[np.isin(segmentation_mask, cell_labels)] = i + 1

    # Create and fill RGB image with colors
    rgb_image = np.zeros(
        (segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8
    )
    rgb_image = color_mapping[annotation_mask]
    rgb_image[annotation_mask == 0] = background_color_rgb

    # Add cell boundaries if outline option is enabled
    if boundary:
        segmentation_boundaries = find_segmentation_boundaries(segmentation_mask)
        rgb_image[segmentation_boundaries.astype(bool)] = boundary_color_rgb

    return rgb_image


# %%
if __name__ == "__main__":
    import random

    import matplotlib.pyplot as plt
    import tifffile
    from pyqupath.tiff import TiffZarrReader

    from pycodex.visualization.multiplex import create_rgb_multiplex
    from pycodex.visualization.utils import ax_plot_legend, ax_plot_rgb_with_scalebar

    reader = TiffZarrReader.from_ometiff(
        "/mnt/nfs/home/wenruiwu/projects/pycodex/demo/data/segmentation/reg001/reg001.ome.tiff"
    )
    img_dict = reader.zimg_dict
    {marker: img_dict[marker] for i, marker in enumerate(img_dict.keys()) if i < 6}

    segmentation_mask = tifffile.imread(
        "/mnt/nfs/home/wenruiwu/projects/pycodex/demo/data/segmentation_mask/segmentation_mask.tiff"
    )
    print(segmentation_mask.shape)

    xmin, ymin = 300, 300
    step = 100
    xmax, ymax = xmin + step, ymin + step

    # %%
    segmentation_boundaries = find_segmentation_boundaries(segmentation_mask)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(segmentation_boundaries[ymin:ymax, xmin:xmax])
    ax_plot_segmentation_labels(segmentation_mask[ymin:ymax, xmin:xmax])

    # %%
    rgb_dapi = create_rgb_multiplex(
        img_dict,
        marker_color_dict={"DAPI": "#999999"},
        marker_cutoff_dict={
            "DAPI": [
                np.percentile(img_dict["DAPI"], 1),
                np.percentile(img_dict["DAPI"], 99),
            ]
        },
        markers_to_plot=["DAPI"],
    )
    rgb_dapi_boundary = create_rgb_with_segmentation_boundaries(
        rgb_dapi, segmentation_mask
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax_plot_rgb_with_scalebar(
        rgb_dapi_boundary[ymin:ymax, xmin:xmax],
        mpp=0.37,
        bar_width_um=10,
        bar_height_px=3,
        text_to_bar_px=2,
        text_size=20,
    )
    ax_plot_segmentation_labels(segmentation_mask[ymin:ymax, xmin:xmax])

    # %%
    color_dict = {
        "Fibroblast": "#0781be",
        "Immune": "#f5eb00",
        "Others": "#808080",
        "Endothelial": "#9f4d9d",
        "Epithelial": "#f47621",
    }
    unique_labels = np.unique(segmentation_mask)
    unique_labels = unique_labels[unique_labels != 0]

    random.seed(123)
    annotations = random.choices(
        list(color_dict.keys()),
        k=len(unique_labels),
    )
    annotation_dict = dict(zip(unique_labels, annotations))
    annotation_dict

    rgb_annotation = create_rgb_annotation(
        segmentation_mask,
        annotation_dict,
        color_dict,
        boundary=True,
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax_plot_rgb_with_scalebar(
        rgb_annotation[ymin:ymax, xmin:xmax],
        mpp=0.37,
        bar_width_um=10,
        bar_height_px=3,
        text_to_bar_px=2,
        text_size=20,
    )
    ax_plot_legend(color_dict)


# %%
