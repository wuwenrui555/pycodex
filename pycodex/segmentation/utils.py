import re

import tifffile


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
