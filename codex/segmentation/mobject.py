import numpy as np
import pandas as pd
from segmentation.segmentation import segmentation_mesmer
from tqdm import tqdm


def metadata_dict_subset_region(
    metadata_dict: dict[str, pd.DataFrame], region_list=list[str]
) -> dict[str, pd.DataFrame]:
    """
    Subset specified region(s) from the metadata dictionary.

    Parameters:
        metadata_dict (dict): Dictionary containing regions as keys and DataFrames as values.
        region_list (list): List of region name(s).

    Returns:
        dict: A new metadata dictionary with specified region(s).
    """
    subset_metadata_dict = {}
    for region in region_list:
        subset_metadata_dict[region] = metadata_dict[region]
    return subset_metadata_dict


def marker_object_subset_region(
    marker_object: dict[str, dict[str, np.ndarray]], region_list=list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """
    Subset specified region(s) from the marker object.

    Parameters:
        marker_object (dict): Dictionary containing regions as keys and marker dictionaries as values.
        region_list (list): List of region name(s).

    Returns:
        dict: A new marker object with specified region(s).
    """
    subset_marker_object = {}
    for region in region_list:
        subset_marker_object[region] = marker_object[region]
    return subset_marker_object


def marker_object_crop_subregion(
    marker_object: dict[str, dict[str, np.ndarray]], x_mid: int, y_mid: int, length: int
) -> dict[str, dict[str, np.ndarray]]:
    """
    Crops a specified subregion from each image in the marker object.

    Parameters:
        marker_object (dict): Dictionary containing regions as keys and marker dictionaries as values.
        x_mid (int): The x-coordinate of the center of the subregion to be cropped.
        y_mid (int): The y-coordinate of the center of the subregion to be cropped.
        length (int): The length of the square subregion to be cropped.

    Returns:
        dict: A new marker object with cropped images.
    """
    cropped_marker_object = {}
    for region, marker_dict in marker_object.items():
        cropped_marker_dict = {}
        for marker, im in marker_dict.items():
            cropped_im = im[
                int(y_mid - length / 2) : int(y_mid + length / 2),
                int(x_mid - length / 2) : int(x_mid + length / 2),
            ]
            cropped_marker_dict[marker] = cropped_im
        cropped_marker_object[region] = cropped_marker_dict
    return cropped_marker_object


def marker_object_segmentation_mesmer(
    marker_object: dict[str, dict[str, np.ndarray]],
    boundary_markers: list[str],
    internal_markers: list[str],
    pixel_size_um: float,
    scale: bool = True,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Perform segmentation (Mesmer) on each image in the marker object.

    Args:
        marker_object (dict): Dictionary containing regions as keys and marker dictionaries as values.
        boundary_markers (list): List of boundary marker names.
        internal_markers (list): List of internal marker names.
        pixel_size_um (float): Pixel size in micrometers.
        scale (bool, optional): Whether to scale the images or not. Defaults to True.
        maxima_threshold (float, optional): Maxima threshold, larger for fewer cells. Defaults to 0.075.
        interior_threshold (float, optional): Interior threshold, larger for larger cells. Defaults to 0.20.

    Returns:
        dict: A mask object with segmentation mask, RGB image, and overlay for each region.
    """
    mask_object = {}
    for region, marker_dict in tqdm(marker_object.items()):
        mask_dict = {}
        mask_dict["segmentation_mask"], mask_dict["rgb_image"], mask_dict["overlay"] = (
            segmentation_mesmer(
                boundary_markers=boundary_markers,
                internal_markers=internal_markers,
                marker_dict=marker_dict,
                pixel_size_um=pixel_size_um,
                scale=scale,
                maxima_threshold=maxima_threshold,
                interior_threshold=interior_threshold,
            )
        )
        mask_object[region] = mask_dict
    return mask_object
