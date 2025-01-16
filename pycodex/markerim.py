import numpy as np
import pandas as pd

import skimage.measure
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay


########################################################################################################################
# subset
########################################################################################################################


def subset_subregion(
    marker_dict: dict[str, np.ndarray],
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> dict[str, np.ndarray]:
    """
    Crop a specified subregion from each image in the marker dictionary.

    Args:
        marker_dict (dict[str, np.ndarray]):
            Dictionary containing marker names as keys and corresponding images as values.
        x_min (int): Minimum x-coordinate of the subregion.
        x_max (int): Maximum x-coordinate of the subregion.
        y_min (int): Minimum y-coordinate of the subregion.
        y_max (int): Maximum y-coordinate of the subregion.

    Returns:
        dict[str, np.ndarray]: A new dictionary with cropped images.
    """
    cropped_marker_dict = {}
    for marker, im in marker_dict.items():
        cropped_im = im[y_min:y_max, x_min:x_max]
        cropped_marker_dict[marker] = cropped_im
    return cropped_marker_dict


########################################################################################################################
# segmentation
########################################################################################################################


def scale_marker_sum(
    marker_list: list[str],
    marker_dict: dict[str : np.ndarray],
    scale: bool = True,
) -> np.ndarray:
    """
    Sum scaled images of specified markers.

    Args:
        marker_list (list): List of marker name to be scaled.
        marker_dict (dict): Dictionary containing marker names as keys and corresponding images as values.
        scale (bool, optional): Whether to scale the images or not. Defaults to True.

    Returns:
        np.ndarray: Summed and scaled image of the specified markers.
    """
    scaled_marker_list = [
        (marker_dict[marker] - marker_dict[marker].min())
        / (marker_dict[marker].max() - marker_dict[marker].min())
        if scale
        else marker_dict[marker]
        for marker in marker_list
    ]
    scaled_marker_sum = np.sum(scaled_marker_list, axis=0)
    # scaled_marker_sum = (
    #     255 * (scaled_marker_sum - scaled_marker_sum.min()) / (scaled_marker_sum.max() - scaled_marker_sum.min())
    # ).astype("uint8")
    scaled_marker_sum = (scaled_marker_sum - scaled_marker_sum.min()) / (
        scaled_marker_sum.max() - scaled_marker_sum.min()
    )
    return scaled_marker_sum


def segmentation_mesmer(
    marker_dict: dict[str : np.ndarray],
    boundary_markers: list[str],
    internal_markers: list[str],
    pixel_size_um: float,
    scale: bool = True,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
    compartment="whole-cell",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform segmentation (Mesmer) on a given image.

    Args:
        marker_dict (dict): Dictionary containing marker names as keys and corresponding images as values.
        boundary_markers (list): List of boundary marker names.
        internal_markers (list): List of internal marker names.
        pixel_size_um (float): Pixel size in micrometers.
        scale (bool, optional): Whether to scale the images or not. Defaults to True.
        maxima_threshold (float, optional): Maxima threshold, larger for fewer cells. Defaults to 0.075.
        interior_threshold (float, optional): Interior threshold, larger for larger cells. Defaults to 0.20.
        compartment (str, optional): Specify type of segmentation to predict. Must be one of "whole-cell", "nuclear", "both". Defaults to "whole-cell".

    Returns:
        Tuple: Segmentation mask, RGB image, and overlay.
    """
    # Data for markers
    boundary_sum = scale_marker_sum(boundary_markers, marker_dict, scale=scale)
    internal_sum = scale_marker_sum(internal_markers, marker_dict, scale=scale)

    # Data for Mesmer
    seg_stack = np.stack((internal_sum, boundary_sum), axis=-1)
    seg_stack = np.expand_dims(seg_stack, 0)

    # Do segmentation
    mesmer = Mesmer()
    segmentation_mask = mesmer.predict(
        seg_stack,
        image_mpp=pixel_size_um,
        postprocess_kwargs_whole_cell={
            "maxima_threshold": maxima_threshold,
            "interior_threshold": interior_threshold,
        },
        compartment=compartment,
    )
    rgb_image = create_rgb_image(seg_stack, channel_colors=["blue", "green"])
    overlay = make_outline_overlay(
        rgb_data=rgb_image, predictions=segmentation_mask
    )
    segmentation_mask = segmentation_mask[0, ..., 0]
    rgb_image = rgb_image[0, ...]
    overlay = overlay[0, ...]
    return segmentation_mask, rgb_image, overlay


def extract_cell_features(
    marker_dict: dict[str, np.ndarray], segmentation_mask: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract single cell features from segmeantaion mask.

    Args:
        marker_dict (dict): Dictionary containing marker names as keys and corresponding images as values.
        segmentation_mask (np.ndarray): Segmentation mask to extract single cell information.

    Returns:
        Tuple: DataFrames containing single cell data and size-scaled data.
    """
    marker_name = [marker for marker in marker_dict.keys()]
    marker_array = np.stack(
        [marker_dict[marker] for marker in marker_name], axis=2
    )

    # extract properties
    props = skimage.measure.regionprops_table(
        segmentation_mask,
        properties=["label", "area", "centroid"],
    )
    props_df = pd.DataFrame(props)
    props_df.columns = ["cellLabel", "cellSize", "Y_cent", "X_cent"]

    # extract marker intensity
    stats = skimage.measure.regionprops(segmentation_mask)
    n_cell = len(stats)
    n_marker = len(marker_name)
    sums = np.zeros((n_cell, n_marker))
    avgs = np.zeros((n_cell, n_marker))
    for i, region in enumerate(stats):
        # Extract the pixel values for the current region from the marker_array
        label_counts = [
            marker_array[coord[0], coord[1], :] for coord in region.coords
        ]
        sums[i] = np.sum(label_counts, axis=0)  # Sum of marker intensities
        avgs[i] = sums[i] / region.area  # Average intensity per unit area

    sums_df = pd.DataFrame(sums, columns=marker_name)
    avgs_df = pd.DataFrame(avgs, columns=marker_name)
    data = pd.concat([props_df, sums_df], axis=1)
    data_scale_size = pd.concat([props_df, avgs_df], axis=1)
    return data, data_scale_size
