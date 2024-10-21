import os
import re

import pandas as pd
from pycodex.segmentation.utils import rename_invalid_marker


def parse_marker(marker_path: str) -> tuple[str, str, str, str, str]:
    """
    Parse marker information from the file name.

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
        columns=["path", "region", "marker"],
    )
    region = metadata_df["region"].unique().item()
    return region, metadata_df


def organize_metadata(marker_dir: str) -> dict[str, pd.DataFrame]:
    """
    Organize metadata for all regions in the final directory.

    Args:
        marker_dir (str): Directory containing subdirectories for each region.

    Returns:
        dict: Dictionary containing region names as keys and metadata DataFrames as values.
    """
    region_dirs = [os.path.join(marker_dir, subdir) for subdir in os.listdir(marker_dir)]
    metadata_dict = {}
    for region_dir in region_dirs:
        region, metadata_df = get_marker_metadata(region_dir)
        metadata_dict[region] = metadata_df
    return metadata_dict
