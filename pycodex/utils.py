import json
import logging
import os

import numpy as np
import pandas as pd
import tifffile
from IPython.display import display
from tifffile import tifffile
from tqdm import tqdm

from pycodex import markerim, metadata

########################################################################################################################
# tiff
########################################################################################################################


def get_tiff_size(tiff_path: str) -> dict[str, float]:
    """
    Retrieves the physical dimensions of a TIFF image in micrometers.

    Args:
        tiff_path (str): Path to the TIFF image file.

    Returns:
        dict: A dictionary containing the width and height of the image in pixels and micrometers,
        as well as pixel size in micrometers.
    """
    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]  # Assuming single-page TIFF

        # Get resolution unit
        res_unit = page.tags.get("ResolutionUnit", None)
        if res_unit is None or res_unit.value == 2:  # 2 means inch
            unit_scale = 25400  # Convert inches to micrometers
        elif res_unit.value == 3:  # 3 means centimeter
            unit_scale = 10000  # Convert centimeters to micrometers
        else:
            raise ValueError("Unsupported resolution unit")

        # Get resolution values
        x_res = page.tags["XResolution"].value[0] / page.tags["XResolution"].value[1]
        y_res = page.tags["YResolution"].value[0] / page.tags["YResolution"].value[1]

        # Calculate size
        width_px = page.imagewidth
        height_px = page.imagelength
        pixel_width_um = unit_scale / x_res
        pixel_height_um = unit_scale / y_res
        width_um = width_px * pixel_width_um
        height_um = height_px * pixel_height_um
        size_dict = {
            "width_px": width_px,
            "height_px": height_px,
            "pixel_width_um": pixel_width_um,
            "pixel_height_um": pixel_height_um,
            "width_um": width_um,
            "height_um": height_um,
        }
        return size_dict


########################################################################################################################
# display
########################################################################################################################


def display_items(items: list[str], ncol: int = 10) -> None:
    """
    Display a list in tabular format.

    Args:
        marker_list (dict): Dictionary or list of markers to display in tabular form.
        ncol (int): Number of columns to display in the output table.

    Returns:
        None: This function displays the DataFrame of markers.
    """
    ncol = min(ncol, len(items))
    markers_df = pd.DataFrame(
        [items[i : i + ncol] for i in range(0, len(items), ncol)],
        columns=[i + 1 for i in range(ncol)],
    ).fillna("")
    display(markers_df)


################################################################################
# segmentation
################################################################################


def segmentation_mesmer(
    output_dir: str,
    metadata_dict: dict[str, pd.DataFrame],
    regions: list[str], 
    boundary_markers: list[str],
    internal_markers: list[str],
    pixel_size_um: float,
    scale: bool = True,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
) -> None:
    """
    Perform segmentation (Mesmer) on each image in the marker object.

    Args:
        output_dir (str): Directory for segmentation output files.
        metadata_dict (dict): Dictionary containing region names as keys and metadata DataFrames as values.
        regions (list): List of regions to perform segmentation.
        boundary_markers (list): List of boundary marker names.
        internal_markers (list): List of internal marker names.
        pixel_size_um (float): Pixel size in micrometers.
        scale (bool, optional): Whether to scale the images or not. Defaults to True.
        maxima_threshold (float, optional): Maxima threshold, larger for fewer cells. Defaults to 0.075.
        interior_threshold (float, optional): Interior threshold, larger for larger cells. Defaults to 0.20.

    Returns:
        None: Save segmentation_mask, rgb_image, and overlay in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # write parameters
    config = {
        "boundary_markers": boundary_markers,
        "internal_markers": internal_markers,
        "pixel_size_um": pixel_size_um,
        "scale": scale,
        "maxima_threshold": maxima_threshold,
        "interior_threshold": interior_threshold,
    }
    with open(f"{output_dir}/parameter_segmentation.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    for region in tqdm(regions): 
        metadata_df = metadata_dict[region]
        all_markers = list(metadata_df["marker"])
        marker_dict = metadata.organize_marker_dict(metadata_dict, region, all_markers)
        try:
            # segmentation
            segmentation_mask, rgb_image, overlay = markerim.segmentation_mesmer(
                boundary_markers=boundary_markers,
                internal_markers=internal_markers,
                marker_dict=marker_dict,
                pixel_size_um=pixel_size_um,
                scale=scale,
                maxima_threshold=maxima_threshold,
                interior_threshold=interior_threshold,
            )

            # save segmentation mask
            output_subdir = os.path.join(output_dir, region)
            os.makedirs(output_subdir, exist_ok=True)
            tifffile.imwrite(os.path.join(output_subdir, "segmentation_mask.tiff"), segmentation_mask.astype(np.uint32))
            tifffile.imwrite(os.path.join(output_subdir, "rgb_image.tiff"), rgb_image)
            tifffile.imwrite(os.path.join(output_subdir, "overlay.tiff"), overlay)
            logging.info(f"{region}: Segmentation completed")

            # save single-cell features
            data, data_scale_size = markerim.extract_cell_features(marker_dict, segmentation_mask)
            data.to_csv(os.path.join(output_subdir, "data.csv"))
            data_scale_size.to_csv(os.path.join(output_subdir, "dataScaleSize.csv"))
            logging.info(f"{region}: Single-cell features extraction completed")

        except Exception as e:
            logging.info(f"[ERROR] '{region}': Failed to process: {e}")
            continue
