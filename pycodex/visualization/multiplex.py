# %%
import numpy as np


def create_rgb_multiplex(
    img_dict,
    marker_color_dict,
    marker_cutoff_dict=None,
    markers_to_plot=None,
    blend_mode="add",
):
    """
    Create a multi-channel fluorescence composite RGB image from marker data.

    This function combines multiple fluorescence marker images into a single RGB
    composite image using specified colors and intensity thresholds for each marker.

    Parameters
    ----------
    img_dict : dict
        Dictionary mapping marker names to 2D numpy arrays containing image data.
    marker_color_dict : dict
        Dictionary mapping marker names to hex color strings (e.g., "#FF0000").
    marker_cutoff_dict : dict
        Dictionary mapping marker names to [min, max] threshold lists for intensity
        clipping.
    markers_to_plot : list
        List of marker names to include in the composite image.
    blend_mode : str, optional
        Layer blending mode - either 'add' (additive) or 'max' (maximum value).
        Default is 'add'.

    Returns
    -------
    numpy.ndarray
        Composite RGB image as uint8 array with shape (height, width, 3).
    """
    if marker_cutoff_dict is None:
        marker_cutoff_dict = {}
    if markers_to_plot is None:
        markers_to_plot = list(img_dict.keys())

    # Validate input parameters
    shapes = [img.shape for img in img_dict.values()]
    if len(set(shapes)) != 1:
        raise ValueError("All images in img_dict must have the same dimensions")
    else:
        height, width = shapes[0]

    # Initialize composite image as float32 for precise calculations
    rgb_composite = np.zeros((height, width, 3), dtype=np.float32)

    # Process each marker individually
    for marker in markers_to_plot:
        if marker not in img_dict:
            print(f"Warning: Marker '{marker}' not found in img_dict, skipping.")
            continue

        # Get marker data and parameters
        marker_img = img_dict[marker]
        color_hex = marker_color_dict[marker]
        intensity_min, intensity_max = marker_cutoff_dict.get(
            marker, [np.min(marker_img), np.max(marker_img)]
        )

        # Convert hex color to RGB values (0-255 range)
        rgb_color = np.array([int(color_hex[i : i + 2], 16) for i in (1, 3, 5)])

        # Normalize marker intensity to [0, 1] range
        # Use float32 to prevent integer division issues
        img_normalized = (marker_img.astype(np.float32) - intensity_min) / (
            intensity_max - intensity_min
        )
        img_normalized = np.clip(img_normalized, 0, 1)

        # Create colored layer for current marker
        # Broadcasting: (h, w, 1) * (3,) -> (h, w, 3)
        current_layer = np.expand_dims(img_normalized, axis=-1) * rgb_color

        # Blend current layer with composite using specified mode
        if blend_mode == "add":
            rgb_composite += current_layer
        elif blend_mode == "max":
            rgb_composite = np.maximum(rgb_composite, current_layer)
        else:
            raise ValueError("blend_mode must be either 'add' or 'max'")

    # Convert final composite to uint8 format for display
    # Clip values to valid RGB range [0, 255]
    rgb_composite = np.clip(rgb_composite, 0, 255).astype(np.uint8)

    return rgb_composite


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pyqupath.tiff import TiffZarrReader

    from pycodex.visualization.utils import ax_plot_legend, ax_plot_scalebar

    reader = TiffZarrReader.from_ometiff(
        "/mnt/nfs/home/wenruiwu/projects/pycodex/demo/data/segmentation/reg001/reg001.ome.tiff"
    )
    img_dict = reader.zimg_dict

    markers_to_plot = [
        "DAPI",
        "CD3e",
        "CD4",
        "CD8",
        # "CD68",
        # "CD163",
        "NaKATP",
    ]

    marker_color_dict = {
        "DAPI": "#999999",
        "CD3e": "#2CA02C",
        "CD4": "#007BFF",
        "CD8": "#D62728",
        "CD68": "#d817e6",
        "CD163": "#17e6bc",
        "NaKATP": "#FF7F0E",
    }

    percentile_ll = 1
    percentile_ul = 99
    marker_cutoff_dict = {
        marker: [
            np.percentile(img_dict[marker], percentile_ll),
            np.percentile(img_dict[marker], percentile_ul),
        ]
        for marker in markers_to_plot
    }

    rgb_composite = create_rgb_multiplex(
        img_dict,
        marker_color_dict,
        markers_to_plot=markers_to_plot,
        marker_cutoff_dict=marker_cutoff_dict,
        blend_mode="add",
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb_composite)
    ax_plot_scalebar(
        rgb_composite,
        mpp=0.37,
        bar_width_um=50,
        bar_height_px=10,
        color_rgb=(255, 255, 255),
        text_unit="μm",
        text_size=15,
        text_to_bar_px=10,
    )
    ax_plot_legend({marker: marker_color_dict[marker] for marker in markers_to_plot})
# %%
