import re

import pandas as pd
import tifffile
from IPython.display import display
from tifffile import tifffile
from tqdm import tqdm

########################################################################################################################
# summary marker
########################################################################################################################


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


########################################################################################################################
# organize marker_dict
########################################################################################################################


def organize_marker_dict(metadata_dict: dict[str, pd.DataFrame], region: str, marker_list: list[str]):
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
    for marker in tqdm(marker_list):
        marker_path = metadata_df["path"][metadata_df["marker"] == marker].item()
        marker_dict[marker] = tifffile.imread(marker_path)
    return marker_dict


########################################################################################################################
# display pixel size
########################################################################################################################


def display_pixel_size(metadata_dict: dict[str, pd.DataFrame], n: int = 1) -> None:
    """
    Display the unique pixel sizes from TIFF metadata.

    Parameters:
    metadata_dict (dict): Dictionary containing region names as keys and metadata DataFrames as values.
    n (int, optional): The number of rows to extract from each DataFrame. Default is 1.

    Returns:
    None: Displays a DataFrame of unique pixel sizes (width or height in micrometers) found in the TIFF files.
    """
    from pycodex.utils import get_tiff_size

    path_list = [path for metadata_df in metadata_dict.values() for path in metadata_df.iloc[:n]["path"]]
    size_df = []
    for path in tqdm(path_list):
        size_df.append(get_tiff_size(path))
    size_df = pd.DataFrame(size_df)
    size_df = size_df[["pixel_width_um", "pixel_height_um"]].drop_duplicates()
    display(size_df)
