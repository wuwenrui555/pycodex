# %%
import json
import logging
import re
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import skimage.measure
import tifffile
from deepcell.applications import Mesmer
from pyqupath.tiff import PyramidWriter, TiffZarrReader
from sklearn.preprocessing import minmax_scale

from pycodex.io import setup_logging

###############################################################################
# segmentation
###############################################################################


def preprocess_marker(
    image: np.ndarray,
    thresh_q_min: float = 0,
    thresh_q_max: float = 1,
    thresh_otsu: bool = False,
    scale: bool = True,
) -> np.ndarray:
    """
    Helper function to preprocess a single marker image.

    Parameters
    ----------
    image : np.ndarray
        Image to be preprocessed.
    thresh_q_min : float, optional
        Lower quantile to cut at. Values below this quantile will be set to 0.
        Defaults to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at. Values above this quantile will be set to
        the quantile value. Defaults to 1.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding to the image or not. Defaults
        to False.
    scale : bool, optional
        Whether to scale the image or not. Defaults to True.

    Returns
    -------
    np.ndarray
        Preprocessed image.
    """

    if thresh_q_min != 0 or thresh_q_max != 1:
        value_q_min = np.quantile(image, thresh_q_min)
        value_q_max = np.quantile(image, thresh_q_max)
        image = np.where(image < value_q_min, 0, image)
        image = np.where(image > value_q_max, value_q_max, image)
    if thresh_otsu:
        min_type = np.min_scalar_type(np.max(image).astype(np.int_))
        _, mask_otsu = cv.threshold(
            image.astype(min_type), 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
    if scale:
        image = minmax_scale(image, feature_range=(0, 1))
    return image if not thresh_otsu else image * mask_otsu


def construct_channel(
    marker_list: list[str],
    marker_dict: dict[str : np.ndarray],
    thresh_q_min: float = 0,
    thresh_q_max: float = 1,
    thresh_otsu: bool = False,
    scale: bool = True,
) -> np.ndarray:
    """
    Construct a channel by images of specified markers.

    Parameters
    ----------
    marker_list : list[str]
        List of marker names to be scaled and summed.
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as
        values.
    thresh_q_min : float, optional
        Lower quantile to cut at. Values below this quantile will be set to 0.
        Defaults to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at. Values above this quantile will be set to
        the quantile value. Defaults to 1.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding to the image or not. Defaults
        to False.
    scale : bool, optional
        Whether to scale the image or not. Defaults to True.

    Returns
    -------
    np.ndarray
        Mean of the images of the specified markers after preprocessing.
    """

    # Check if any marker has constant value
    constant = [
        np.min(marker_dict[marker]) == np.max(marker_dict[marker])
        for marker in marker_list
    ]
    if any(constant):
        logging.warning(
            f"Marker with constant value: {np.array(marker_list)[constant].tolist()}"
        )

    # Construct channel of the specified markers
    image_dict = {
        marker: preprocess_marker(
            marker_dict[marker],
            thresh_q_min=thresh_q_min,
            thresh_q_max=thresh_q_max,
            thresh_otsu=thresh_otsu,
            scale=scale,
        )
        for marker in marker_list
    }
    image_channel = np.mean([image for image in image_dict.values()], axis=0)
    return image_channel, image_dict


def segmentation_mesmer(
    marker_dict: dict[str : np.ndarray],
    internal_markers: list[str],
    boundary_markers: list[str],
    thresh_q_min: float,
    thresh_q_max: float,
    thresh_otsu: bool,
    scale: bool,
    pixel_size_um: float,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
    compartment="whole-cell",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform segmentation (Mesmer) on a given image.

    Parameters
    ----------
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as
        values.
    boundary_markers : list
        List of boundary marker names.
    internal_markers : list
        List of internal marker names.
    thresh_q_min : float, optional
        Lower quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values below this quantile will be set to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values above this quantile will be set to the
        quantile value.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding for each marker in `internal_markers`
        and `boundary_markers`. Values below the OTSU threshold will be set to 0.
    scale : bool, optional
        Whether to scale each marker in `internal_markers` and `boundary_markers`
        before summing.
    pixel_size_um : float
        Pixel size in micrometers for marker images.
        Note:
        - Fusion: 0.5068164319979996
        - Keyence: 0.3775202
    maxima_threshold : float, optional
        Maxima threshold for Mesmer. Lower values will result in more separate
        cells being predicted, whereas higher values will result in fewer cells.
        Defaults to 0.075.
    interior_threshold : float, optional
        Interior threshold for Mesmer. Lower values will result in larger cells,
        whereas higher values will result in smaller cells. Defaults to 0.20.
    compartment : str, optional
        Specify type of segmentation to predict. Must be one of "whole-cell",
        "nuclear", "both". Defaults to "whole-cell".

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Segmentation mask, boundary channel, and internal channel.
    """
    # Data for Mesmer
    internal_channel, internal_dict = construct_channel(
        marker_list=internal_markers,
        marker_dict=marker_dict,
        thresh_q_min=thresh_q_min,
        thresh_q_max=thresh_q_max,
        thresh_otsu=thresh_otsu,
        scale=scale,
    )
    internal_channel = minmax_scale(internal_channel, feature_range=(0, 1))
    boundary_channel, boundary_dict = construct_channel(
        marker_list=boundary_markers,
        marker_dict=marker_dict,
        thresh_q_min=thresh_q_min,
        thresh_q_max=thresh_q_max,
        thresh_otsu=thresh_otsu,
        scale=scale,
    )
    boundary_channel = minmax_scale(boundary_channel, feature_range=(0, 1))
    image_stack = np.stack((internal_channel, boundary_channel), axis=-1)
    image_stack = np.expand_dims(image_stack, 0)

    # Do segmentation
    mesmer = Mesmer()
    segmentation_mask = mesmer.predict(
        image_stack,
        image_mpp=pixel_size_um,
        postprocess_kwargs_whole_cell={
            "maxima_threshold": maxima_threshold,
            "interior_threshold": interior_threshold,
        },
        compartment=compartment,
    )
    return (
        segmentation_mask,
        internal_channel,
        boundary_channel,
        internal_dict,
        boundary_dict,
    )


def extract_cell_features(
    marker_dict: dict[str, np.ndarray],
    segmentation_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract single cell features from segmentation mask.

    Parameters
    ----------
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as
        values.
    segmentation_mask : np.ndarray
        A 2D segmentation mask with the same shape as the marker images, in
        which each cell is labeled with a unique integer.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Dataframe containing single-cell features.
        - Dataframe containing single-cell features with marker intensities
          scaled by cell size.
    """
    marker_name = [marker for marker in marker_dict.keys()]
    marker_array = np.stack([marker_dict[marker] for marker in marker_name], axis=2)

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
        label_counts = [marker_array[coord[0], coord[1], :] for coord in region.coords]
        sums[i] = np.sum(label_counts, axis=0)  # Sum of marker intensities
        avgs[i] = sums[i] / region.area  # Average intensity per unit area

    sums_df = pd.DataFrame(sums, columns=marker_name)
    avgs_df = pd.DataFrame(avgs, columns=marker_name)
    data = pd.concat([props_df, sums_df], axis=1)
    data_scale_size = pd.concat([props_df, avgs_df], axis=1)
    return data, data_scale_size


def run_segmentation_mesmer_cell(
    unit_dir: str,
    internal_markers: list[str],
    boundary_markers: list[str],
    thresh_q_min: float,
    thresh_q_max: float,
    thresh_otsu: bool,
    scale: bool,
    pixel_size_um: float,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
    tag: str = None,
    ometiff_path: str = None,
    num_threads: int = 8,
):
    """
    Run whole-cell segmentation using Mesmer.

    Parameters
    ----------
    unit_dir : str
        Directory to load and save data for segmentation.
    internal_markers : list
        List of internal marker names.
    boundary_markers : list
        List of boundary marker names.
    thresh_q_min : float, optional
        Lower quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values below this quantile will be set to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values above this quantile will be set to the
        quantile value.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding for each marker in `internal_markers`
        and `boundary_markers`. Values below the OTSU threshold will be set to 0.
    scale : bool, optional
        Whether to scale each marker in `internal_markers` and `boundary_markers`
        before summing.
    pixel_size_um : float
        Pixel size in micrometers for marker images.
        Note:
        - Fusion: 0.5068164319979996
        - Keyence: 0.3775202
    maxima_threshold : float, optional
        Maxima threshold for Mesmer. Lower values will result in more separate
        cells being predicted, whereas higher values will result in fewer cells.
        Defaults to 0.075.
    interior_threshold : float, optional
        Interior threshold for Mesmer. Lower values will result in larger cells,
        whereas higher values will result in smaller cells. Defaults to 0.20.
    tag : str, optional
        Tag for the segmentation directory. Defaults to None, using time as tag
        (YYYYMMDD_HHMMSS).
    ometiff_path : str, optional
        Path to the OME-TIFF file containing the marker images for segmentation.
        Defaults to None, using the only OME-TIFF file in `unit_dir`.
    num_threads : int, optional
        The number of threads to use for writing the OME-TIFF file. Defaults to 8.
    """
    # Set up directories
    unit_dir = Path(unit_dir)
    if tag is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
    segmentation_dir = unit_dir / tag
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(segmentation_dir / "segmentation.log")

    # Load OME-TIFF file
    if ometiff_path is None:
        pattern = re.compile(r".*\.ome\.tif[f]?", re.IGNORECASE)
        ometiff_paths = [f for f in unit_dir.glob("*") if pattern.match(f.name)]
        if len(ometiff_paths) == 0:
            logging.error("No OME-TIFF file found in the directory.")
            raise FileNotFoundError("No OME-TIFF file found in the directory.")
        elif len(ometiff_paths) > 1:
            logging.error("Multiple OME-TIFF files found in the directory.")
            raise ValueError("Multiple OME-TIFF files found in the directory.")
        else:
            ometiff_path = ometiff_paths[0]
            tiff_reader = TiffZarrReader.from_ometiff(ometiff_path)
            marker_dict = tiff_reader.zimg_dict
            logging.info(f"OME-TIFF file loaded: {ometiff_path}.")

    # Check whether selected markers are present in the OME-TIFF file
    markers = tiff_reader.channel_names
    missing_markers = [
        marker
        for marker in boundary_markers + internal_markers
        if marker not in markers
    ]
    if len(missing_markers) > 0:
        logging.error(f"Missing markers: {missing_markers}")
        raise ValueError(f"Missing markers: {missing_markers}")

    # Write parameters
    params = {
        "internal_markers": internal_markers,
        "boundary_markers": boundary_markers,
        "thresh_q_min": thresh_q_min,
        "thresh_q_max": thresh_q_max,
        "thresh_otsu": thresh_otsu,
        "scale": scale,
        "pixel_size_um": pixel_size_um,
        "maxima_threshold": maxima_threshold,
        "interior_threshold": interior_threshold,
        "compartment": "whole-cell",
    }
    with open(
        f"{segmentation_dir}/parameter_segmentation.json", "w", encoding="utf-8"
    ) as file:
        json.dump(params, file, indent=4, ensure_ascii=False)

    # Segmentation
    (
        segmentation_mask,
        internal_channel,
        boundary_channel,
        internal_dict,
        boundary_dict,
    ) = segmentation_mesmer(marker_dict=marker_dict, **params)
    segmentation_mask = segmentation_mask[0, :, :, 0]
    segmentation_mask_f = segmentation_dir / "segmentation_mask.tiff"
    tifffile.imwrite(str(segmentation_mask_f), segmentation_mask)
    logging.info("Segmentation completed.")

    # Save markers for segmentation, boundary and internal channels
    segmentation_markers_dict = {}
    segmentation_markers_dict.update(internal_dict)
    segmentation_markers_dict.update(boundary_dict)
    min_type = np.max([img.dtype for img in marker_dict.values()])
    if scale:
        segmentation_markers_dict["internal_sum"] = internal_channel
        segmentation_markers_dict["boundary_sum"] = boundary_channel
        segmentation_markers_dict = {
            marker: (image * np.iinfo(min_type).max).astype(min_type)
            for marker, image in segmentation_markers_dict.items()
        }
    else:
        segmentation_markers_dict["internal_sum"] = (
            internal_channel * np.iinfo(min_type).max
        )
        segmentation_markers_dict["boundary_sum"] = (
            boundary_channel * np.iinfo(min_type).max
        )
        segmentation_markers_dict = {
            marker: image.astype(min_type) for marker, image in marker_dict.items()
        }
    segmentation_markers_f = segmentation_dir / "segmentation_markers.ome.tiff"
    tiff_writer = PyramidWriter.from_dict(segmentation_markers_dict)
    tiff_writer.export_ometiff_pyramid(
        segmentation_markers_f, overwrite=True, num_threads=num_threads
    )
    logging.info("Markers used for segmentation saved as OME-TIFF.")

    # Extract single-cell features
    data, data_scale = extract_cell_features(marker_dict, segmentation_mask)
    data.to_csv(segmentation_dir / "data.csv")
    data_scale.to_csv(segmentation_dir / "dataScaleSize.csv")
    logging.info("Single-cell features extracted.")
