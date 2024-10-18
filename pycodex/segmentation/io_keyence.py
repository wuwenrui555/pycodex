import os
import re

import numpy as np
import pandas as pd
from IPython.display import display
from codex.segmentation.utils import get_tiff_size, rename_invalid_marker
from tifffile import tifffile
from tqdm import tqdm


def parse_marker(marker_path: str) -> tuple[str, str, str, str, str]:
    """
    Parse marker information from the file name.

    Args:
        marker_path (str): Path to the marker image file (*.tif).

    Returns:
        tuple: Parsed marker information including path, region, cycle, channel, and renamed marker.
    """
    marker_basename = os.path.basename(marker_path)
    pattern = r"(reg\d+)_(cyc\d+)_(ch\d+)_(.+)\.tif"
    match = re.match(pattern, marker_basename)
    region, cycle, channel, marker = match.groups()
    return marker_path, region, cycle, channel, rename_invalid_marker(marker)


def get_marker_metadata(region_dir: str) -> tuple[str, pd.DataFrame]:
    """
    Get metadata for all markers in a given region directory.

    Args:
        region_dir (str): Directory containing marker files for a specific region.

    Returns:
        tuple: Region name and metadata DataFrame containing paths, region, cycle, channel, and marker.
    """
    marker_paths = []
    for root, dirs, files in os.walk(region_dir):
        for file in files:
            marker_paths.append(os.path.join(root, file))
    metadata_df = pd.DataFrame(
        [parse_marker(path) for path in marker_paths],
        columns=["path", "region", "cycle", "channel", "marker"],
    )
    region = metadata_df["region"].unique().item()
    return region, metadata_df


def organize_metadata(final_dir: str) -> dict[str, pd.DataFrame]:
    """
    Organize metadata for all regions in the final directory.

    Args:
        final_dir (str): Directory containing subdirectories for each region.

    Returns:
        dict: Dictionary containing region names as keys and metadata DataFrames as values.
    """
    region_dirs = [os.path.join(final_dir, subdir) for subdir in os.listdir(final_dir)]
    metadata_dict = {}
    for region_dir in region_dirs:
        region, metadata_df = get_marker_metadata(region_dir)
        metadata_dict[region] = metadata_df
    return metadata_dict


def summary_markers(
    metadata_dict: dict[str, pd.DataFrame],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Summarize marker information across regions.

    Args:
        metadata_dict (dict): Dictionary containing region names as keys and metadata DataFrames as values.

    Returns:
        tuple: Lists of unique markers, blank markers, duplicated markers, and markers missing in some regions.
    """
    # Combine all metadata DataFrames into one
    combined_metadata_df = pd.concat(metadata_dict.values(), ignore_index=True)

    # Get all unique markers
    all_markers = combined_metadata_df["marker"].unique()

    # Identify blank markers
    blank_markers = [
        marker for marker in all_markers if re.match(r"blank", marker, re.IGNORECASE)
    ]

    # Filter out blank markers
    metadata_df = combined_metadata_df.loc[
        ~combined_metadata_df["marker"].isin(blank_markers)
    ]

    # Create a pivot table to count occurrences of each marker in each region
    count_pivot = metadata_df.pivot_table(
        index="marker", columns="region", aggfunc="size", fill_value=0
    )

    # Identify markers that are missing in some regions
    missing_markers_n = (count_pivot == 0).sum(axis=1)
    missing_markers_df = count_pivot.loc[missing_markers_n > 0]
    missing_markers = list(missing_markers_df.index)

    # Identify markers that are duplicated in some regions
    duplicated_markers_n = (count_pivot > 1).sum(axis=1)
    duplicated_markers_df = count_pivot.loc[duplicated_markers_n > 0]
    duplicated_markers = list(duplicated_markers_df.index)

    # Identify unique markers (not blank, not duplicated, and not missing in any region)
    unique_markers = [
        marker
        for marker in all_markers
        if marker not in (blank_markers + duplicated_markers + missing_markers)
    ]
    unique_markers = sorted(unique_markers)

    # Display summary information
    print(
        f"Summary of Markers:\n"
        f"- Total unique markers: {len(all_markers)}\n"
        f"- Unique markers: {len(unique_markers)} {unique_markers}\n"
        f"- Blank markers: {len(blank_markers)} {blank_markers}\n"
        f"- Markers duplicated in some regions: {len(duplicated_markers)} {duplicated_markers}\n"
        f"- Markers missing in some regions: {len(missing_markers)} {missing_markers}"
    )
    return unique_markers, blank_markers, duplicated_markers, missing_markers


def display_markers(marker_list: list[str], ncol: int = 10) -> None:
    """
    Display markers in tabular format.

    Args:
        marker_list (dict): Dictionary or list of markers to display in tabular form.
        ncol (int): Number of columns to display in the output table.

    Returns:
        None: This function displays the DataFrame of markers.
    """
    markers_df = pd.DataFrame(
        [marker_list[i : i + ncol] for i in range(0, len(marker_list), ncol)],
        columns=[i + 1 for i in range(ncol)],
    ).fillna("")
    display(markers_df)


def display_pixel_size(metadata_dict: dict[str, pd.DataFrame], n: int = 1) -> None:
    """
    Display the unique pixel sizes from TIFF metadata.

    Parameters:
    metadata_dict (dict): Dictionary containing region names as keys and metadata DataFrames as values.
    n (int, optional): The number of rows to extract from each DataFrame. Default is 1.

    Returns:
    None: Displays a DataFrame of unique pixel sizes (width or height in micrometers) found in the TIFF files.
    """
    path_list = [
        path
        for metadata_df in metadata_dict.values()
        for path in metadata_df.iloc[:n]["path"]
    ]
    size_df = []
    for path in tqdm(path_list): 
        size_df.append(get_tiff_size(path))
    size_df = pd.DataFrame(size_df)
    size_df = size_df[["pixel_width_um", "pixel_height_um"]].drop_duplicates()
    display(size_df)


def organize_marker_object(
    metadata_dict: dict[str, pd.DataFrame], marker_list: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """
    Organize marker images by region.

    Args:
        metadata_dict (dict): Dictionary containing region names as keys and metadata DataFrames as values.
        marker_list (list): List of markers to be organized.

    Returns:
        dict: Dictionary containing region names as keys and dictionaries of marker images as values.
    """
    marker_object = {}
    for region, metadata_df in tqdm(metadata_dict.items()):
        marker_dict = {}
        for marker in marker_list:
            marker_path = metadata_df["path"][metadata_df["marker"] == marker].item()
            marker_dict[marker] = tifffile.imread(marker_path)
        marker_object[region] = marker_dict
    return marker_object
