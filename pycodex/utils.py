import json
import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
import tifffile
from IPython.display import display
from tqdm import tqdm

from pycodex import crop, markerim, metadata

###############################################################################
# tiff
###############################################################################


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


###############################################################################
# display
###############################################################################


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
    compartment="whole-cell",
) -> None:
    """
    Perform segmentation (Mesmer) on each image in the marker object.

    Parameters
    ----------
    output_dir : str
        Directory for segmentation output files.
    metadata_dict : dict
        Dictionary containing region names as keys and metadata DataFrames as values.
    regions : list
        List of regions to perform segmentation.
    boundary_markers : list
        List of boundary marker names.
    internal_markers : list
        List of internal marker names.
    pixel_size_um : float
        Pixel size in micrometers. (Referenced: 0.5068164319979996 for Fusion,
        0.3775202 for Keyence)
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
    None
        Save segmentation_mask, rgb_image, and overlay in the output directory.
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
        "compartment": compartment,
    }
    with open(
        f"{output_dir}/parameter_segmentation.json", "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    unique_markers, _, _, _ = metadata.summary_markers(metadata_dict)
    for region in tqdm(regions):
        marker_dict = metadata.organize_marker_dict(
            metadata_dict, region, unique_markers
        )
        logging.info(f"{region}: Markers loaded")

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
                compartment=compartment,
            )

            # save segmentation mask
            output_subdir = os.path.join(output_dir, region)
            os.makedirs(output_subdir, exist_ok=True)
            tifffile.imwrite(
                os.path.join(output_subdir, "segmentation_mask.tiff"),
                segmentation_mask.astype(np.uint32),
            )
            tifffile.imwrite(os.path.join(output_subdir, "rgb_image.tiff"), rgb_image)
            tifffile.imwrite(os.path.join(output_subdir, "overlay.tiff"), overlay)
            logging.info(f"{region}: Segmentation completed")

            # save single-cell features
            data, data_scale_size = markerim.extract_cell_features(
                marker_dict, segmentation_mask
            )
            data.to_csv(os.path.join(output_subdir, "data.csv"))
            data_scale_size.to_csv(os.path.join(output_subdir, "dataScaleSize.csv"))
            logging.info(f"{region}: Single-cell features extraction completed")

        except Exception as e:
            logging.info(f"[ERROR] '{region}': Failed to process: {e}")
            continue


################################################################################
# cropping for mantis
################################################################################


def crop_image_into_blocks(
    marker_dir: str,
    segmentation_dir: str,
    output_dir: str,
    regions: list[str],
    max_block_size=3000,
):
    """
    Crop large marker and segmentation images into smaller blocks for each region.

    This function processes marker images and segmentation masks for multiple regions.
    It identifies markers with inconsistent shapes, filters them out, and crops the
    remaining images into smaller blocks. Each block is saved as a separate file.

    Args:
        marker_dir (str):
            Directory containing marker image subdirectories for each region.
        segmentation_dir (str):
            Directory containing segmentation masks for each region.
        output_dir (str):
            Directory where cropped blocks will be saved.
        regions (List[str]):
            List of region names to crop.
        max_block_size (int):
            The maximum allowed size for any block along both dimensions (height and width).
            Defaults to 3000.

    Returns:
        None: The function saves the cropped blocks as TIFF files in the output directory.
    """
    for region in regions:
        marker_subdir = os.path.join(marker_dir, region)
        marker_files = os.listdir(marker_subdir)
        marker_files = [
            file
            for file in marker_files
            if os.path.splitext(file)[1] in [".tiff", ".tif"]
        ]
        marker_paths = [os.path.join(marker_subdir, file) for file in marker_files]

        segmentation_subdir = os.path.join(segmentation_dir, region)
        segmentation_path = os.path.join(segmentation_subdir, "segmentation_mask.tiff")

        all_paths = marker_paths + [segmentation_path]

        marker_dict = {}
        for path in tqdm(all_paths, desc=f"Loading {region}"):
            marker_name = os.path.splitext(os.path.basename(path))[0]
            marker_image = tifffile.imread(path)
            marker_dict[marker_name] = marker_image

        # Get all shapes from marker_dict and track their indices
        shapes = [image.shape for image in marker_dict.values()]

        # Count the occurrences of each shape
        shape_counter = Counter(shapes)
        most_common_shape = shape_counter.most_common(1)[0][0]
        outlier_markers = [
            list(marker_dict.keys())[i]
            for i, shape in enumerate(shapes)
            if shape != most_common_shape
        ]
        print(f"Outlier markers: {outlier_markers}")

        filtered_marker_dict = {
            marker: image
            for marker, image in marker_dict.items()
            if marker not in outlier_markers
        }
        xy_limits = crop.crop_image_into_blocks(
            most_common_shape, max_block_size=max_block_size
        )

        output_subdir = os.path.join(output_dir, region)
        os.makedirs(output_subdir, exist_ok=True)

        fig = crop.plot_block_labels(
            filtered_marker_dict["segmentation_mask"], xy_limits
        )
        fig.savefig(os.path.join(output_subdir, f"{region}_subregions.tiff"))

        for label, (x_beg, x_end, y_beg, y_end) in tqdm(
            xy_limits.items(), desc=f"Cropping {region}: "
        ):
            for marker, im in filtered_marker_dict.items():
                output_subdir = os.path.join(output_dir, region, f"{region}_{label}")
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, f"{marker}.tiff")
                dtype = im.dtype.type
                im_sm = (im[y_beg:y_end, x_beg:x_end]).astype(dtype)
                tifffile.imwrite(output_path, im_sm)
        logging.info(f"{region}: Cropping completed")
