import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.morphology
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay


def scale_marker(marker: str, marker_dict: dict[str, np.ndarray], scale: bool = True) -> np.ndarray:
    """
    Scale image of a specific marker.

    Args:
        marker (str): The name of the marker to be scaled.
        marker_dict (dict): Dictionary containing marker names askeys and corresponding images as values.
        scale (bool, optional): Whether to scale the image or not. Defaults to True.

    Returns:
        np.ndarray: Scaled image of the specified marker.
    """
    im_marker = marker_dict[marker]
    if scale:
        im_marker = (im_marker - im_marker.min()) / (im_marker.max() - im_marker.min())
    return im_marker


def scale_marker_sum(marker_list: list[str], marker_dict: dict[str : np.ndarray], scale: bool = True) -> np.ndarray:
    """
    Sum scaled images of specified markers.

    Args:
        marker_list (list): List of marker name to be scaled.
        marker_dict (dict): Dictionary containing marker names as keys and corresponding images as values.
        scale (bool, optional): Whether to scale the images or not. Defaults to True.

    Returns:
        np.ndarray: Summed and scaled image of the specified markers.
    """
    scale_marker_list = [scale_marker(marker, marker_dict, scale=scale) for marker in marker_list]
    scale_marker_sum = np.sum(scale_marker_list, axis=0)
    scale_marker_sum = (
        255 * (scale_marker_sum - scale_marker_sum.min()) / (scale_marker_sum.max() - scale_marker_sum.min())
    ).astype("uint8")
    return scale_marker_sum


def segmentation_mesmer(
    boundary_markers: list[str],
    internal_markers: list[str],
    marker_dict: dict[str : np.ndarray],
    pixel_size_um: float,
    scale: bool = True,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform segmentation (Mesmer) on a given image.

    Args:
        boundary_markers (list): List of boundary marker names.
        internal_markers (list): List of internal marker names.
        marker_dict (dict): Dictionary containing marker names as keys and corresponding images as values.
        pixel_size_um (float): Pixel size in micrometers.
        scale (bool, optional): Whether to scale the images or not. Defaults to True.
        maxima_threshold (float, optional): Maxima threshold, larger for fewer cells. Defaults to 0.075.
        interior_threshold (float, optional): Interior threshold, larger for larger cells. Defaults to 0.20.

    Returns:
        Tuple: Segmentation mask, RGB image, and overlay.
    """
    # Data for markers
    boundary_sum = scale_marker_sum(boundary_markers, marker_dict, scale=scale)
    internal_sum = scale_marker_sum(internal_markers, marker_dict, scale=scale)

    # Data for Mesmer
    seg_stack = np.stack((internal_sum, boundary_sum), axis=-1)
    seg_stack = np.expand_dims(seg_stack, 0)

    # Do segmentation
    mesmer = Mesmer()
    segmentation_mask = mesmer.predict(
        seg_stack,
        image_mpp=pixel_size_um,
        postprocess_kwargs_whole_cell={
            "maxima_threshold": maxima_threshold,
            "interior_threshold": interior_threshold,
        },
        compartment="nuclear",
    )
    rgb_image = create_rgb_image(seg_stack, channel_colors=["blue", "green"])
    overlay = make_outline_overlay(rgb_data=rgb_image, predictions=segmentation_mask)
    segmentation_mask = segmentation_mask[0, ..., 0]
    rgb_image = rgb_image[0, ...]
    overlay = overlay[0, ...]
    return segmentation_mask, rgb_image, overlay


def extract_single_cell_info(
    marker_dict: dict[str, np.ndarray], segmentation_mask: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract single cell information from a core.

    Args:
        marker_dict (dict): Dictionary containing marker names as keys and corresponding images as values.
        segmentation_mask (np.ndarray): Segmentation mask to extract single cell information.

    Returns:
        Tuple: DataFrames containing single cell data and size-scaled data.
    """
    marker_name = [marker for marker in marker_dict.keys() if marker != "Empty"]
    marker_array = [marker_dict[marker] for marker in marker_name]
    counts_no_noise = np.stack(marker_array, axis=2)

    # Calculate properties of each segmented region
    stats = skimage.measure.regionprops(segmentation_mask)
    label_num = len(stats)

    channel_num = len(marker_array)
    data = np.zeros((label_num, channel_num))  # Sum of intensities for each marker
    data_scale_size = np.zeros((label_num, channel_num))  # Mean intensity per cell size
    cell_sizes = np.zeros((label_num, 1))  # Area of each cell
    cell_props = np.zeros((label_num, 3))  # Label and centroid coordinates for each cell

    for i, region in enumerate(stats):
        cell_label = region.label
        # Extract the pixel values for the current region from the counts array
        label_counts = [counts_no_noise[coord[0], coord[1], :] for coord in region.coords]
        data[i] = np.sum(label_counts, axis=0)  # Sum of marker intensities
        data_scale_size[i] = data[i] / region.area  # Average intensity per unit area
        cell_sizes[i] = region.area  # Store the area of the cell
        cell_props[i] = [
            cell_label,
            region.centroid[0],
            region.centroid[1],
        ]  # Store label and centroid

    # Create DataFrames to store the cell data
    data_df = pd.DataFrame(data, columns=marker_name)
    data_full = pd.concat(
        [
            pd.DataFrame(cell_props, columns=["cellLabel", "Y_cent", "X_cent"]),
            pd.DataFrame(cell_sizes, columns=["cellSize"]),
            data_df,
        ],
        axis=1,
    )

    # Create DataFrames to store scaled cell data (intensity per unit size)
    data_scale_size_df = pd.DataFrame(data_scale_size, columns=marker_name)
    data_scale_size_full = pd.concat(
        [
            pd.DataFrame(cell_props, columns=["cellLabel", "Y_cent", "X_cent"]),
            pd.DataFrame(cell_sizes, columns=["cellSize"]),
            data_scale_size_df,
        ],
        axis=1,
    )
    return data_full, data_scale_size_full


# def segmentation_mesmer_object(marker_object):
#     """
#     Process a single TMA.

#     Args:
#         folder_object (str): Path to the folder containing TMA data.
#         path_parameter (str): Path to the parameter configuration file.
#         tag (str): Tag to be used for output folder.
#     """
#     name_object = Path(folder_object).name

#     folder_output = f"{folder_object}/segmentation/{tag}/"
#     os.makedirs(folder_output, exist_ok=True)

#     # Write parameter
#     config = load_config(path_parameter)
#     keys = [
#         "boundary",
#         "internal",
#         "scale",
#         "pixel_size_um",
#         "maxima_threshold",
#         "interior_threshold",
#     ]
#     config = {key: config.get(key) for key in keys}
#     with open(
#         f"{folder_output}/parameter_segmentation.json", "w", encoding="utf-8"
#     ) as file:
#         json.dump(config, file, indent=4, ensure_ascii=False)

#     marker_dict = load_tiff_markers(f"{folder_object}/marker")
#     logging.info("tiff loaded for segmentation")

#     # Segmentation
#     segmentation_mask, rgb_image, overlay = segmentation_mesmer(
#         config["boundary"],
#         config["internal"],
#         marker_dict,
#         config["scale"],
#         config["pixel_size_um"],
#         config["maxima_threshold"],
#         config["interior_threshold"],
#     )
#     segmentation_mask = segmentation_mask[0, ..., 0]
#     overlay = overlay[0, ...]
#     tifffile.imwrite(f"{folder_output}/mesmer_mask.tiff", segmentation_mask)
#     tifffile.imwrite(f"{folder_output}/mesmer_overlay.tiff", overlay)
#     logging.info(f"Segmentation completed for {name_object}")

#     # Extract single cell information
#     data_full, data_scale_size_full = extract_single_cell_info(
#         marker_dict, segmentation_mask
#     )
#     data_full.to_csv(f"{folder_output}/data.csv", index=False)
#     data_scale_size_full.to_csv(f"{folder_output}/dataScaleSize.csv", index=False)
#     logging.info(f"Single cell information extracted for {name_object}")
