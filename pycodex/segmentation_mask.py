import numpy as np
from joblib import delayed
from matplotlib.axes import Axes
from skimage.segmentation import find_boundaries
from tqdm import tqdm
from tqdm_joblib import ParallelPbar

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def find_boundaries_label(
    segmentation_mask: np.ndarray,
    label: int,
    connectivity: int = 1,
    mode: str = "inner",
) -> np.ndarray:
    """
    Get boundary for specific label in a segmentation mask.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        An array in which different regions are labeled with different integers.
    label : int
        The label for which to extract the boundary.
    connectivity : int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}
        How to mark the boundaries:
        - thick: any pixel not completely surrounded by pixels of the
          same label (defined by `connectivity`) is marked as a boundary.
          This results in boundaries that are 2 pixels thick.
        - inner: outline the pixels *just inside* of objects, leaving
          background pixels untouched.
        - outer: outline pixels in the background around object
          boundaries. When two objects touch, their boundary is also
          marked.
        - subpixel: return a doubled image, with pixels *between* the
          original pixels marked as boundary where appropriate.

    Returns
    -------
    tuple
        A tuple containing label, x_indices, and y_indices of the boundary pixels.
    """
    binary_mask = segmentation_mask == label
    y_indices, x_indices = np.where(binary_mask)
    x_min, x_max = x_indices.min(), x_indices.max() + 1
    y_min, y_max = y_indices.min(), y_indices.max() + 1
    binary_mask_label = binary_mask[y_min:y_max, x_min:x_max]
    binary_mask_label_expand = np.zeros(
        (binary_mask_label.shape[0] + 2, binary_mask_label.shape[1] + 2),
        dtype=bool,
    )
    binary_mask_label_expand[1:-1, 1:-1] = binary_mask_label
    boundary_label_expand = find_boundaries(
        binary_mask_label_expand, connectivity=connectivity, mode=mode
    )
    boundary_label = boundary_label_expand[1:-1, 1:-1]

    y_indices, x_indices = np.where(boundary_label)
    x_indices = x_indices + x_min
    y_indices = y_indices + y_min

    return label, x_indices, y_indices


def find_boundaries_joblib(
    segmentation_mask: np.ndarray,
    connectivity: int = 1,
    mode: str = "inner",
    n_jobs: int = 10,
):
    """
    Convert a segmentation mask to a boundary mask using parallel processing.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        An array in which different regions are labeled with different integers.
    connectivity : int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}
        How to mark the boundaries:
        - thick: any pixel not completely surrounded by pixels of the
          same label (defined by `connectivity`) is marked as a boundary.
          This results in boundaries that are 2 pixels thick.
        - inner: outline the pixels *just inside* of objects, leaving
          background pixels untouched.
        - outer: outline pixels in the background around object
          boundaries. When two objects touch, their boundary is also
          marked.
        - subpixel: return a doubled image, with pixels *between* the
          original pixels marked as boundary where appropriate.

    n_jobs : int, optional
        The number of parallel jobs to run. Default is 10.

    Returns
    -------
    np.ndarray
        A 2D numpy array with the boundaries of the segmentation mask overlaid.
    """
    # Extract unique labels from the mask
    labels = np.unique(segmentation_mask)
    labels = labels[labels != 0]

    boundaries = ParallelPbar(
        desc="Mask to boundaries",
        bar_format=TQDM_FORMAT,
    )(n_jobs=n_jobs)(
        delayed(find_boundaries_label)(
            segmentation_mask, label, connectivity=connectivity, mode=mode
        )
        for label in labels
    )

    segmentation_boundary = np.zeros_like(segmentation_mask, dtype=int)
    for label, x_indices, y_indices in tqdm(boundaries, bar_format=TQDM_FORMAT):
        segmentation_boundary[y_indices, x_indices] = label

    return segmentation_boundary


def plot_labels(
    segmentation_mask: np.ndarray,
    ax: Axes,
    color: str = "white",
    fontsize: int = 8,
    ha: str = "center",
    va: str = "center",
    bbox: dict = dict(facecolor="black", alpha=0.5, edgecolor="none"),
):
    """
    Plot labels from a segmentation mask.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        An array in which different regions are labeled with different integers.
    ax : matplotlib.axes.Axes
        The axes on which to plot the labels.
    color : str, optional
        Color of the text labels. Default is "white".
    fontsize : int, optional
        Font size of the text labels. Default is 8.
    ha : str, optional
        Horizontal alignment of the text labels. Default is "center".
    va : str, optional
        Vertical alignment of the text labels. Default is "center".
    bbox : dict, optional
        A dictionary defining the bounding box properties for the text labels.
        Default is a semi-transparent black box with no edge color.
    """

    labels = np.unique(segmentation_mask)
    labels = labels[labels != 0]
    for label in labels:
        y_indices, x_indices = np.where(segmentation_mask == label)
        x_mid = (x_indices.min() + x_indices.max()) // 2
        y_mid = (y_indices.min() + y_indices.max()) // 2
        ax.text(
            x_mid,
            y_mid,
            str(label),
            color=color,
            fontsize=fontsize,
            ha=ha,
            va=va,
            bbox=bbox,
        )
