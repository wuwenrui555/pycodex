import os
import re

import pandas as pd

########################################################################################################################
# rename marker
########################################################################################################################


def rename_invalid_marker(marker_name: str) -> str:
    """
    Rename the invalid marker name (containing "/" and ":").

    Args:
        marker (str): Original marker name.

    Returns:
        str: Modified marker name with invalid characters replaced by "_".
    """
    marker_name = re.sub(r"[/:]", "_", marker_name)
    return marker_name


def rename_duplicate_markers(marker_list: list[str]) -> list[str]:
    """
    Renames duplicate markers by appending a numeric suffix to ensure uniqueness.

    Parameters:
    marker_list (list): A list of strings representing marker names.

    Returns:
    list: A list of marker names where duplicates are renamed with a suffix (e.g., '_2', '_3').
    """
    seen = {}
    renamed_list = []
    for marker in marker_list:
        if marker in seen:
            seen[marker] += 1
            renamed_list.append(f"{marker}_{seen[marker]}")
        else:
            seen[marker] = 1
            renamed_list.append(marker)
    print(f"Duplicated markers: {[marker for marker, n in seen.items() if n > 1]}")

    return renamed_list


########################################################################################################################
# organize metadata
########################################################################################################################


def parse_marker_fusion(marker_path: str) -> pd.DataFrame:
    """
    Fusion: Parse marker information from the file name.

    Args:
        marker_path (str): Path to the marker image file (*.tif).

    Returns:
        pd.DataFrame: DataFrame of metadata.
    """
    marker_basename = os.path.basename(marker_path)
    pattern = r"(.+)\.tiff?$"
    match = re.match(pattern, marker_basename)
    (marker,) = match.groups()
    metadata = pd.DataFrame([[marker_path, rename_invalid_marker(marker)]], columns=["path", "marker"])
    return metadata


def parse_marker_keyence(marker_path: str) -> pd.DataFrame:
    """
    Keyence: Parse marker information from the file name.

    Args:
        marker_path (str): Path to the marker image file (*.tif).

    Returns:
        pd.DataFrame: DataFrame of metadata.
    """
    marker_basename = os.path.basename(marker_path)
    pattern = r"(reg\d+)_(cyc\d+)_(ch\d+)_(.+)\.tiff?$"
    match = re.match(pattern, marker_basename)
    region, cycle, channel, marker = match.groups()
    metadata = pd.DataFrame(
        [[marker_path, region, cycle, channel, rename_invalid_marker(marker)]],
        columns=["path", "_region", "cycle", "channel", "marker"],
    )
    return metadata


def _organize_metadata(
    marker_dir: str, parse_marker_func, subfolders: bool = True, extensions: list[str] = [".tiff", ".tif"]
):
    """
    Internal function to organize metadata from marker files.

    Args:
        marker_dir (str): Directory containing marker images or subdirectories of marker images.
        parse_marker_func (function): Function to parse individual marker image files into DataFrames.
        subfolders (bool): If True, marker files are in subfolders of marker_dir (e.g., one subfolder per region).
            If False, marker files are directly in marker_dir. Default is True.
        extensions (list): List of allowed file extensions. Default is [".tiff", ".tif"].

    Returns:
        dict: Dictionary with region names as keys and metadata DataFrames as values.
    """
    metadata_dict = {}
    if subfolders:
        for region in os.listdir(marker_dir):
            region_dir = os.path.join(marker_dir, region)
            metadata_dfs = [
                parse_marker_func(os.path.join(region_dir, marker))
                for marker in os.listdir(region_dir)
                if os.path.splitext(marker)[1] in extensions
            ]
            metadata_df = pd.concat(metadata_dfs, axis=0).reset_index(drop=True)
            metadata_df["region"] = region
            metadata_dict[region] = metadata_df
    else:
        region = "region"
        metadata_dfs = [
            parse_marker_func(os.path.join(marker_dir, marker))
            for marker in os.listdir(marker_dir)
            if os.path.splitext(marker)[1] in extensions
        ]
        metadata_df = pd.concat(metadata_dfs, axis=0).reset_index(drop=True)
        metadata_df["region"] = region
        metadata_dict[region] = metadata_df
    return metadata_dict


def organize_metadata_fusion(
    marker_dir: str, subfolders: bool = True, extensions: list[str] = [".tiff", ".tif"]
) -> dict[str, pd.DataFrame]:
    """
    Organize metadata from marker files for Fusion output.

    Args:
        marker_dir (str): Directory containing marker images or subdirectories of marker images.
        parse_marker_func (function): Function to parse individual marker image files into DataFrames.
        subfolders (bool): If True, marker files are in subfolders of marker_dir (e.g., one subfolder per region).
            If False, marker files are directly in marker_dir. Default is True.
        extensions (list): List of allowed file extensions. Default is [".tiff", ".tif"].

    Returns:
        dict: Dictionary with region names as keys and metadata DataFrames as values.
    """
    return _organize_metadata(marker_dir, parse_marker_fusion, subfolders, extensions)


def organize_metadata_keyence(
    marker_dir: str, subfolders: bool = True, extensions: list[str] = [".tiff", ".tif"]
) -> dict[str, pd.DataFrame]:
    """
    Organize metadata from marker files for Keyence output.

    Args:
        marker_dir (str): Directory containing marker images or subdirectories of marker images.
        parse_marker_func (function): Function to parse individual marker image files into DataFrames.
        subfolders (bool): If True, marker files are in subfolders of marker_dir (e.g., one subfolder per region).
            If False, marker files are directly in marker_dir. Default is True.
        extensions (list): List of allowed file extensions. Default is [".tiff", ".tif"].

    Returns:
        dict: Dictionary with region names as keys and metadata DataFrames as values.
    """
    return _organize_metadata(marker_dir, parse_marker_keyence, subfolders, extensions)
