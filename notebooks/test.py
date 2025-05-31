# %%
import logging
import numpy as np
import re
from pathlib import Path

import matplotlib.pyplot as plt
from pyqupath.tiff import TiffZarrReader

from pycodex.io import setup_gpu
from pycodex.segmentation import segmentation_mesmer
from pycodex.segmentation_mask import (
    create_rgb_segmentation_mask,
    plot_labels,
    find_label_boundaries,
)

setup_gpu("1")
# %%
unit_dir = Path("../demo/data/segmentation/reg001/")


ometiff_path = None
if ometiff_path is None:
    pattern = re.compile(r".*\.ome\.tif[f]?", re.IGNORECASE)
    ometiff_paths = [f for f in unit_dir.glob("*") if pattern.match(f.name)]
    if len(ometiff_paths) == 0:
        logging.error("No OME-TIFF file found in the directory.")
        raise FileNotFoundError("No OME-TIFF file found in the directory.")
    elif len(ometiff_paths) > 1:
        logging.error("Multiple OME-TIFF files found in the directory.")
        raise ValueError("Multiple OME-TIFF files found in the directory.")
    else:
        ometiff_path = ometiff_paths[0]
        tiff_reader = TiffZarrReader.from_ometiff(ometiff_path)
        marker_dict = tiff_reader.zimg_dict
        logging.info(f"OME-TIFF file loaded: {ometiff_path}.")

marker_dict = {k: v[300:500, 300:500] for k, v in marker_dict.items()}


internal_markers = ["DAPI"]
boundary_markers = ["CD45", "CD3e", "CD163", "NaKATP"]
thresh_q_min = 0
thresh_q_max = 0.99
thresh_otsu = False
scale = True
pixel_size_um = 0.5068164319979996
maxima_threshold = 0.075
interior_threshold = 0.20

# Write parameters
params = {
    "internal_markers": internal_markers,
    "boundary_markers": boundary_markers,
    "thresh_q_min": thresh_q_min,
    "thresh_q_max": thresh_q_max,
    "thresh_otsu": thresh_otsu,
    "scale": scale,
    "pixel_size_um": pixel_size_um,
    "maxima_threshold": maxima_threshold,
    "interior_threshold": interior_threshold,
    "compartment": "both",
}

# Segmentation
(
    segmentation_mask,
    internal_channel,
    boundary_channel,
    internal_dict,
    boundary_dict,
) = segmentation_mesmer(marker_dict=marker_dict, **params)

# %%
segmentation_mask_cell = segmentation_mask[0, :, :, 0]
segmentation_mask_nuclear = segmentation_mask[0, :, :, 1]

segmentation_mask_nuclear_fixed = (
    segmentation_mask_cell * segmentation_mask_nuclear.astype(bool)
)

labels_nuclear = np.unique(
    segmentation_mask_nuclear_fixed[segmentation_mask_nuclear_fixed != 0]
)
segmentation_mask_cell_both = segmentation_mask_cell.copy()
segmentation_mask_cell_both = segmentation_mask_cell_both * np.isin(
    segmentation_mask_cell, labels_nuclear
)

segmentation_mask_membrane_both = segmentation_mask_cell_both.copy()
segmentation_mask_membrane_both = (
    segmentation_mask_membrane_both - segmentation_mask_nuclear_fixed
)

# %%
rgb_cell = create_rgb_segmentation_mask(
    internal_channel,
    boundary_channel,
    outline=True,
    segmentation_mask=segmentation_mask_cell,
)
rgb_nuclear = create_rgb_segmentation_mask(
    internal_channel,
    boundary_channel,
    outline=True,
    segmentation_mask=segmentation_mask_nuclear,
)
rgb_nuclear_fixed = create_rgb_segmentation_mask(
    internal_channel,
    boundary_channel,
    outline=True,
    segmentation_mask=segmentation_mask_nuclear_fixed,
)
rgb_cell_both = create_rgb_segmentation_mask(
    internal_channel,
    boundary_channel,
    outline=True,
    segmentation_mask=segmentation_mask_cell_both,
)
# %%

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for ax in axes:
    ax.axis("off")
axes[0].imshow(rgb_cell)
plot_labels(segmentation_mask_cell, ax=axes[0])
axes[0].set_title("Cell Segmentation")

axes[1].imshow(rgb_nuclear)
plot_labels(segmentation_mask_nuclear, ax=axes[1])
axes[1].set_title("Nuclear Segmentation")

axes[2].imshow(rgb_nuclear_fixed)
plot_labels(segmentation_mask_nuclear_fixed, ax=axes[2])
axes[2].set_title("Nuclear Segmentation (fixed)")

axes[3].imshow(rgb_cell_both)
plot_labels(segmentation_mask_cell_both, ax=axes[3])
axes[3].set_title("Cell Segmentation (both)")

fig.tight_layout()

# %%
boundaries = find_label_boundaries(segmentation_mask_cell_both)

rgb_data = np.zeros(
    (segmentation_mask_cell_both.shape[0], segmentation_mask_cell_both.shape[1], 3),
    dtype=np.uint8,
)
rgb_data[segmentation_mask_membrane_both.astype(bool)] = (0, 255, 0)
rgb_data[segmentation_mask_nuclear_fixed.astype(bool)] = (0, 0, 255)

fig, axes = plt.subplots(2, 1, figsize=(5, 10))
axes = axes.flatten()
for ax in axes:
    ax.axis("off")
axes[0].imshow(rgb_data)
axes[0].set_title("Cell and Nuclear Segmentation")

rgb_data[boundaries.astype(bool)] = (255, 255, 255)
axes[1].imshow(rgb_data)
plot_labels(segmentation_mask_cell_both, ax=axes[1])
axes[1].set_title("Cell and Nuclear Segmentation with Boundaries")

fig.tight_layout()
# %%

internal_markers = ["DAPI"]
boundary_markers = ["CD45", "CD3e", "CD163", "NaKATP"]
thresh_q_min = 0
thresh_q_max = 0.99
thresh_otsu = False
scale = True
pixel_size_um = 0.5068164319979996
maxima_threshold = 0.075
interior_threshold = 0.20

# Write parameters
params = {
    "internal_markers": internal_markers,
    "boundary_markers": boundary_markers,
    "thresh_q_min": thresh_q_min,
    "thresh_q_max": thresh_q_max,
    "thresh_otsu": thresh_otsu,
    "scale": scale,
    "pixel_size_um": pixel_size_um,
    "maxima_threshold": maxima_threshold,
    "interior_threshold": interior_threshold,
    "compartment": "both",
}

# Segmentation
(
    segmentation_mask,
    internal_channel,
    boundary_channel,
    internal_dict,
    boundary_dict,
) = segmentation_mesmer(marker_dict=marker_dict, **params)

# Segmentation mask for cell, nuclear, and membrane compartments
segmentation_mask_cell = segmentation_mask[0, :, :, 0]
segmentation_mask_nuclear = segmentation_mask[0, :, :, 1]
segmentation_mask_nuclear = (
    segmentation_mask_nuclear.astype(bool) * segmentation_mask_cell
)
segmentation_mask_membrane = segmentation_mask_cell - segmentation_mask_nuclear


rgb_image_cell = create_rgb_segmentation_mask(
    internal_channel,
    boundary_channel,
    outline=True,
    segmentation_mask=segmentation_mask_cell,
)
rgb_image_nuclear = create_rgb_segmentation_mask(
    internal_channel,
    boundary_channel,
    outline=True,
    segmentation_mask=segmentation_mask_nuclear,
)
rgb_image_membrane = create_rgb_segmentation_mask(
    internal_channel,
    boundary_channel,
    outline=True,
    segmentation_mask=segmentation_mask_membrane,
)


labels_both = [
    label
    for label in np.unique(segmentation_mask_cell)
    if label in np.unique(segmentation_mask_nuclear)
]
boundaries = find_label_boundaries(segmentation_mask_cell)

rgb_image = np.zeros(
    (segmentation_mask_cell.shape[0], segmentation_mask_cell.shape[1], 3),
    dtype=np.uint8,
)
rgb_image[segmentation_mask_membrane.astype(bool)] = (0, 255, 0)
rgb_image[segmentation_mask_nuclear.astype(bool)] = (0, 0, 255)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()
for ax in axes:
    ax.axis("off")
axes[0].imshow(rgb_image)

rgb_image[np.isin(segmentation_mask_cell, labels_both, invert=True)] = (255, 0, 0)
rgb_image[boundaries.astype(bool)] = (255, 255, 255)
axes[1].imshow(rgb_image)

fig.tight_layout()
# %%
