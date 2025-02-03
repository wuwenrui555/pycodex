import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.measure
import tifffile
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay
from pyqupath.geojson import mask_to_geojson_joblib
from pyqupath.ometiff import export_ometiff_pyramid_from_dict, load_tiff_to_dict

from pycodex.io import setup_logging

###############################################################################
# segmentation
###############################################################################


def cut_quantile(x: np.ndarray, q_min: float = 0.01, q_max: float = 0.99):
    """
    Cut values in an array at the specified quantiles.
    Values below q_min are set to 0, and values above q_max are set to q_max.

    Parameters
    ----------
    x : np.ndarray
        Values to be cut.
    q_min : float, optional, default=0.01
        Lower quantile to cut at.
    q_max : float, optional, default=0.99
        Upper quantile to cut at.

    Returns
    -------
    np.ndarray
        Array with values modified based on the specified quantiles.
    """
    min_val = np.quantile(x, q_min)
    max_val = np.quantile(x, q_max)
    x = np.where(x < min_val, 0, x)  # Set values below q_min to 0
    x = np.where(x > max_val, max_val, x)  # Set values above q_max to q_max
    return x


def scale_to_0_1(
    x: np.ndarray,
    constant_value: float = 0,
) -> np.ndarray:
    """
    Scale values in an array to the range [0, 1].

    Parameters
    ----------
    x : np.ndarray
        Values to be scaled.
    constant_value : float, optional, default=0.0
        Value to return when all elements in `x` are equal (min == max).

    Returns
    -------
    np.ndarray
        Scaled values in the range [0, 1], or an array with the specified
        `constant_value` if all elements in `x` are the same.
    """
    min_val = np.min(x)
    max_val = np.max(x)

    # Handle the edge case where all values are the same
    if min_val == max_val:
        return np.full_like(x, fill_value=constant_value, dtype=float)

    return (x - min_val) / (max_val - min_val)


def scale_marker_sum(
    marker_list: list[str],
    marker_dict: dict[str : np.ndarray],
    scale: bool = True,
) -> np.ndarray:
    """
    Sum scaled images of specified markers.

    Parameters
    ----------
    marker_list : list
        List of marker name to be scaled.
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as values.
    scale : bool, optional
        Whether to scale the images or not. Defaults to True.

    Returns
    -------
    np.ndarray
        Summed and scaled image of the specified markers.
    """
    scaled = [
        scale_to_0_1(marker_dict[marker]) if scale else marker_dict[marker]
        for marker in marker_list
    ]
    constant = [
        np.min(marker_dict[marker]) == np.max(marker_dict[marker])
        for marker in marker_list
    ]
    if any(constant):
        logging.warning(
            f"Marker with constant value: {np.array(marker_list)[constant].tolist()}"
        )
    scaled_sum = scale_to_0_1(np.sum(scaled, axis=0))
    return scaled_sum


def segment_mesmer(
    marker_dict: dict[str : np.ndarray],
    boundary_markers: list[str],
    internal_markers: list[str],
    pixel_size_um: float,
    q_min: float = 0,
    q_max: float = 1,
    scale: bool = True,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
    compartment="whole-cell",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform segmentation (Mesmer) on a given image.

    Parameters
    ----------
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as values.
    boundary_markers : list
        List of boundary marker names.
    internal_markers : list
        List of internal marker names.
    pixel_size_um : float
        Pixel size in micrometers. Note:
        - Fusion: 0.5068164319979996
        - Keyence: 0.3775202
    q_min : float, optional
        Lower quantile to cut at. Defaults to 0.
    q_max : float, optional
        Upper quantile to cut at. Defaults to 1.
    scale : bool, optional
        Whether to scale the images or not. Defaults to True.
    maxima_threshold : float, optional
        Maxima threshold, larger for fewer cells. Defaults to 0.075.
    interior_threshold : float, optional
        Interior threshold, larger for larger cells. Defaults to 0.20.
    compartment : str, optional
        Specify type of segmentation to predict. Must be one of "whole-cell",
        "nuclear", "both". Defaults to "whole-cell".

    Returns
    -------
    Tuple
        Segmentation mask, RGB image, overlay, stacked signals of boundary markers
        and stacked signals of internal markers used for segmentation.
    """
    # Cut quantile
    if q_min != 0 or q_max != 1:
        marker_dict = {
            key: cut_quantile(value, q_min=q_min, q_max=q_max)
            for key, value in marker_dict.items()
        }

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
    return segmentation_mask, rgb_image, overlay, boundary_sum, internal_sum


def extract_cell_features(
    marker_dict: dict[str, np.ndarray],
    segmentation_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract single cell features from segmentation mask.

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


def run_segmentation_mesmer(
    output_dir: str,
    boundary_markers: list[str],
    internal_markers: list[str],
    pixel_size_um: float,
    q_min: float = 0,
    q_max: float = 1,
    scale: bool = True,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
    compartment="whole-cell",
    tag: str = None,
    ometiff_path: str = None,
    mask_to_geojson: bool = False,
):
    """
    Run segmentation using Mesmer.

    Parameters
    ----------
    output_dir : str
        Output directory to save the results.
    boundary_markers : list
        List of boundary marker names.
    internal_markers : list
        List of internal marker names.
    pixel_size_um : float
        Pixel size in micrometers. Note:
        - Fusion: 0.5068164319979996
        - Keyence: 0.3775202
    q_min : float, optional
        Lower quantile to cut at. Defaults to 0.
    q_max : float, optional
        Upper quantile to cut at. Defaults to 1.
    scale : bool, optional
        Whether to scale the images or not. Defaults to True.
    maxima_threshold : float, optional
        Maxima threshold, larger for fewer cells. Defaults to 0.075.
    interior_threshold : float, optional
        Interior threshold, larger for larger cells. Defaults to 0.20.
    compartment : str, optional
        Specify type of segmentation to predict. Must be one of "whole-cell",
        "nuclear", "both". Defaults to "whole-cell".
    tag : str, optional
        Tag to append to the output directory. Defaults to None, using time as
        tag (YYYYMMDD_HHMMSS).
    ometiff_path : str, optional
        Path to the input OME-TIFF file. Defaults to None, using the only one
        OME-TIFF file under `output_dir`.
    mask_to_geojson : bool, optional
        Whether to generate a GeoJSON file for segmentation mask. Defaults to
        False.
    """
    # Set up logging
    dir_root = Path(output_dir)
    if tag is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
    dir_output = dir_root / tag
    dir_output.mkdir(parents=True, exist_ok=True)
    setup_logging(dir_output / "segmentation.log")

    # Load OME-TIFF file
    if ometiff_path is None:
        ometiff_paths = list(dir_root.glob("*.ome.tiff"))
        if len(ometiff_paths) == 0:
            logging.error("No OME-TIFF file found in the directory.")
            raise FileNotFoundError("No OME-TIFF file found in the directory.")
        elif len(ometiff_paths) > 1:
            logging.error("Multiple OME-TIFF files found in the directory.")
            raise ValueError("Multiple OME-TIFF files found in the directory.")
        else:
            ometiff_path = ometiff_paths[0]
            marker_dict = load_tiff_to_dict(ometiff_path, filetype="ome.tiff")
            logging.info(f"OME-TIFF file loaded: {ometiff_path}.")

    # Check whether selected markers are present in the OME-TIFF file
    all_markers = list(marker_dict.keys())
    missing_markers = [
        marker
        for marker in boundary_markers + internal_markers
        if marker not in all_markers
    ]
    if len(missing_markers) > 0:
        logging.error(f"Missing markers: {missing_markers}")
        raise ValueError(f"Missing markers: {missing_markers}")

    # Write parameters
    config = {
        "boundary_markers": boundary_markers,
        "internal_markers": internal_markers,
        "pixel_size_um": pixel_size_um,
        "q_min": q_min,
        "q_max": q_max,
        "scale": scale,
        "maxima_threshold": maxima_threshold,
        "interior_threshold": interior_threshold,
        "compartment": compartment,
    }
    with open(
        f"{dir_output}/parameter_segmentation.json", "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    # Perform segmentation
    try:
        # Segmentation
        segmentation_mask, _, _, boundary_sum, internal_sum = segment_mesmer(
            boundary_markers=boundary_markers,
            internal_markers=internal_markers,
            marker_dict=marker_dict,
            pixel_size_um=pixel_size_um,
            q_min=q_min,
            q_max=q_max,
            scale=scale,
            maxima_threshold=maxima_threshold,
            interior_threshold=interior_threshold,
            compartment=compartment,
        )
        path_segmentation_mask = dir_output / "segmentation_mask.tiff"
        tifffile.imwrite(
            path_segmentation_mask, segmentation_mask.astype(np.uint32)
        )
        logging.info("Segmentation completed.")

        # Save stacked signals of boundary and internal markers used for segmentation
        if not scale:
            boundary_sum = scale_to_0_1(boundary_sum)
            internal_sum = scale_to_0_1(internal_sum)
        dtype = marker_dict[internal_markers[0]].dtype
        boundary_sum = (boundary_sum * np.iinfo(dtype).max).astype(dtype)
        internal_sum = (internal_sum * np.iinfo(dtype).max).astype(dtype)
        marker_seg_dict = {
            marker: marker_dict[marker]
            for marker in boundary_markers + internal_markers
        }
        marker_seg_dict["boundary_sum"] = boundary_sum
        marker_seg_dict["internal_sum"] = internal_sum
        path_marker_seg = dir_output / "segmentation_markers.ome.tiff"
        if path_marker_seg.exists():
            os.remove(path_marker_seg)
        export_ometiff_pyramid_from_dict(marker_seg_dict, path_marker_seg)
        logging.info("Markers used for segmentation saved as OME-TIFF.")

        # Extract single-cell features
        data, data_scale_size = extract_cell_features(
            marker_dict, segmentation_mask
        )
        data.to_csv(dir_output / "data.csv")
        data_scale_size.to_csv(dir_output / "dataScaleSize.csv")
        logging.info("Single-cell features extracted.")

        # Generate GeoJSON file for segmentation mask
        if mask_to_geojson:
            mask_to_geojson_joblib(
                segmentation_mask, dir_output / "segmentation_mask.geojson"
            )
            logging.info("Segmentation GeoJSON generated.")

    except Exception as e:
        logging.error(f"Segmentation failed: {e}")
