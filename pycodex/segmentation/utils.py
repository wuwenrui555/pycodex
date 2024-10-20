import re

import pandas as pd
from IPython.display import display
import tifffile
from tqdm import tqdm

################################################################################
# tiff
################################################################################


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


def display_pixel_size(metadata_dict: dict[str, pd.DataFrame], n: int = 1) -> None:
    """
    Display the unique pixel sizes from TIFF metadata.

    Parameters:
    metadata_dict (dict): Dictionary containing region names as keys and metadata DataFrames as values.
    n (int, optional): The number of rows to extract from each DataFrame. Default is 1.

    Returns:
    None: Displays a DataFrame of unique pixel sizes (width or height in micrometers) found in the TIFF files.
    """
    path_list = [path for metadata_df in metadata_dict.values() for path in metadata_df.iloc[:n]["path"]]
    size_df = []
    for path in tqdm(path_list):
        size_df.append(get_tiff_size(path))
    size_df = pd.DataFrame(size_df)
    size_df = size_df[["pixel_width_um", "pixel_height_um"]].drop_duplicates()
    display(size_df)


################################################################################
# marker
################################################################################


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

    return renamed_list


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
