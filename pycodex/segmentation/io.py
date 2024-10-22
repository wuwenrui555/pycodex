import os
import re

import numpy as np
import pandas as pd
from tifffile import tifffile

from pycodex.segmentation.utils import rename_invalid_marker

################################################################################
# fusion
################################################################################


def parse_marker_fusion(marker_path: str) -> tuple[str, str, str, str, str]:
    """
    Fusion: Parse marker information from the file name.

    Args:
        marker_path (str): Path to the marker image file (*.tif).

    Returns:
        tuple: Parsed marker information including path, region, cycle, channel, and renamed marker.
    """
    marker_basename = os.path.basename(marker_path)
    marker_dirname = os.path.dirname(marker_path)
    region = os.path.basename(marker_dirname)
    pattern = r"(.+)\.tif"
    match = re.match(pattern, marker_basename)
    (marker,) = match.groups()
    return marker_path, region, rename_invalid_marker(marker)


def get_marker_metadata_fusion(region_dir: str) -> tuple[str, pd.DataFrame]:
    """
    Fusion: Get metadata for all markers in a given region directory.

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
        [parse_marker_fusion(path) for path in marker_paths],
        columns=["path", "region", "marker"],
    )
    region = metadata_df["region"].unique().item()
    return region, metadata_df


def organize_metadata_fusion(marker_dir: str) -> dict[str, pd.DataFrame]:
    """
    Organize metadata for all regions in the final directory.

    Args:
        marker_dir (str): Directory containing subdirectories which contain marker files for a specific region.

    Returns:
        dict: Dictionary containing region names as keys and metadata DataFrames as values.
    """
    region_dirs = [os.path.join(marker_dir, subdir) for subdir in os.listdir(marker_dir)]
    metadata_dict = {}
    for region_dir in region_dirs:
        region, metadata_df = get_marker_metadata_fusion(region_dir)
        metadata_dict[region] = metadata_df
    return metadata_dict


################################################################################
# keyence
################################################################################


def parse_marker_keyence(marker_path: str) -> tuple[str, str, str, str, str]:
    """
    Keyence: Parse marker information from the file name.

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


def get_marker_metadata_keyence(region_dir: str) -> tuple[str, pd.DataFrame]:
    """
    Keyence: Get metadata for all markers in a given region directory.

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
        [parse_marker_keyence(path) for path in marker_paths],
        columns=["path", "region", "cycle", "channel", "marker"],
    )
    region = metadata_df["region"].unique().item()
    return region, metadata_df


def organize_metadata_keyence(marker_dir: str) -> dict[str, pd.DataFrame]:
    """
    Keyence: Organize metadata for all regions in the final directory.

    Args:
        marker_dir (str): Directory containing subdirectories which contain marker files for a specific region.

    Returns:
        dict: Dictionary containing region names as keys and metadata DataFrames as values.
    """
    region_dirs = [os.path.join(marker_dir, subdir) for subdir in os.listdir(marker_dir)]
    metadata_dict = {}
    for region_dir in region_dirs:
        region, metadata_df = get_marker_metadata_keyence(region_dir)
        metadata_dict[region] = metadata_df
    return metadata_dict


################################################################################
# metadata_dict
################################################################################


def summary_metadata(
    metadata_dict: dict[str, pd.DataFrame],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Summarize marker information across regions.

    Args:
        metadata_dict (dict): Dictionary containing region names as keys and metadata DataFrames as values.

    Returns:
        tuple: Lists of unique markers, blank markers, duplicated markers, and markers missing in some regions.
    """
    # Get All unique markers
    combined_metadata_df = pd.concat(metadata_dict.values(), ignore_index=True)
    all_markers = combined_metadata_df["marker"].unique()

    # Identify and filter out blank markers
    blank_markers = [marker for marker in all_markers if re.match(r"blank", marker, re.IGNORECASE)]
    metadata_df = combined_metadata_df.loc[~combined_metadata_df["marker"].isin(blank_markers)]
    count_pivot = metadata_df.pivot_table(index="marker", columns="region", aggfunc="size", fill_value=0)

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
        marker for marker in all_markers if marker not in (blank_markers + duplicated_markers + missing_markers)
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


def organize_marker_dict(
    metadata_dict: dict[str, pd.DataFrame], region: str, marker_list: list[str]
) :
    """
    Organize marker dictionary for a specific region.

     Args:
        metadata_dict (dict): Dictionary containing region names as keys and metadata DataFrames as values.
        region (str): Name of the region to extract markers for.
        marker_list (list): List of marker names to organize.

    Returns:
        dict: Dictionary containing marker names as keys and marker images as values for a specific region.
    """
    marker_dict = {}
    metadata_df = metadata_dict[region]
    for marker in marker_list:
        marker_path = metadata_df["path"][metadata_df["marker"] == marker].item()
        marker_dict[marker] = tifffile.imread(marker_path)
    return marker_dict


################################################################################
# marker_dict
################################################################################