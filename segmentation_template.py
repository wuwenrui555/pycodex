# %%
import colorsys
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import tifffile

from pycodex.io import setup_gpu
from pycodex.segmentation import (
    generate_segmentation_mask_geojson,
    mask_to_geojson_joblib,
    run_segmentation_mesmer_cell,
)
from tqdm import tqdm

TQDM_FORMAT = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def generate_distinct_colors(
    n: int, saturation: float = 0.7, value: float = 0.95
) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct colors using HSV color space.

    This function generates colors by:
    1. Using the golden ratio to space hues evenly
    2. Using fixed saturation and value to ensure good visibility
    3. Avoiding similar hues when colors are adjacent

    Parameters
    ----------
    n : int
        Number of colors to generate
    saturation : float, optional
        Color saturation (0-1), default 0.7
    value : float, optional
        Color value/brightness (0-1), default 0.95

    Returns
    -------
    List[Tuple[int, int, int]]
        List of RGB color tuples with values 0-255
    """
    colors = []
    golden_ratio = 0.618033988749895  # Golden ratio conjugate

    # Start with primary colors for small n
    primary_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    if n <= len(primary_colors):
        return primary_colors[:n]

    # Use golden ratio method for larger n
    hue = 0
    for i in range(n):
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to 0-255 range
        rgb_int = tuple(int(255 * x) for x in rgb)
        colors.append(rgb_int)
        # Use golden ratio to space hues evenly
        hue = (hue + golden_ratio) % 1.0

    return colors


def assign_bright_colors(
    labels: List[Union[str, int]],
) -> dict[Union[str, int], Tuple[int, int, int]]:
    """
    Assign bright RGB colors to variable values.

    Ensures good distinction between colors. Similar to QuPath's implementation
    but with improved color separation.

    Parameters
    ----------
    labels : List[Union[str, int]]
        List of labels that need distinct colors

    Returns
    -------
    dict[Union[str, int], Tuple[int, int, int]]
        Dictionary mapping each label to an RGB color tuple
    """
    n_colors = len(labels)
    colors = generate_distinct_colors(n_colors)
    return dict(zip(labels, colors))


def update_geojson_classification(
    geojson_f: Union[Path, str],
    output_f: Union[Path, str],
    name_dict: dict[Union[str, int], Union[str, int]],
    color_dict: Optional[dict[Union[str, int], tuple[int, int, int]]] = None,
) -> None:
    """
    Update classification names and colors in a GeoJSON file.

    Parameters
    ----------
    geojson_f : Union[Path, str]
        Input GeoJSON file path
    output_f : Union[Path, str]
        Output GeoJSON file path
    name_dict : Union[dict[str, str], dict[int, str]]
        Dictionary mapping original names to new classification names
    color_dict : Optional[dict[Union[str, int], tuple[int, int, int]]], optional
        Dictionary mapping classification names to RGB colors.
        If not provided, colors will be automatically assigned.
    """
    # Read input GeoJSON
    with open(geojson_f, "r") as f:
        geojson_data = json.load(f)

    # Generate colors if not provided
    if color_dict is None:
        unique_names = list(set(name_dict.values()))
        color_dict = assign_bright_colors(unique_names)

    # Update features
    features = geojson_data["features"]
    for feature in features:
        properties = feature["properties"]
        if "name" in properties:
            orig_name = properties["name"]
            if orig_name in name_dict:
                new_name = name_dict[orig_name]
                properties["classification"] = {
                    "name": new_name,
                    "color": color_dict[new_name],
                }
        properties["isLocked"] = True
    # Write output
    output_path = Path(output_f)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson_data, f)


# %%
################################################################################
# geojson
################################################################################

if False:
    output_root = "/mnt/nfs/storage/wenruiwu_temp/20250513_huaying_geojson"
    annotation_df = pd.read_csv(
        "/mnt/nfs/home/huayingqiu/hodgkinebvmibi/rebuttal/src/cluster_042825/annotated/annotation_no_artifact.csv"
    )
    seg_mask_dir = Path(
        "/mnt/nfs/home/huayingqiu/pipeline_test/cHL_revision/output/seg_mask_0.3_0.01_OG_correct"
    )
    seg_mask_files = [
        f for f in seg_mask_dir.glob("*.tiff") if not f.name.endswith("overlay.tiff")
    ]

    for seg_mask_file in tqdm(
        seg_mask_files, desc="Processing segmentation masks", bar_format=TQDM_FORMAT
    ):
        tma_core = seg_mask_file.name.split(".", 1)[0]
        output_dir = Path(output_root) / tma_core
        output_dir.mkdir(parents=True, exist_ok=True)

        segmentation_mask = tifffile.imread(seg_mask_file)
        output_f = output_dir / f"{tma_core}.geojson"
        mask_to_geojson_joblib(
            segmentation_mask,
            output_f,
            n_jobs=16,
            batch_size=100,
        )

        name_dict = (
            annotation_df.query("TMA_core == @tma_core")
            .set_index("cellLabel")["Annotation"]
            .to_dict()
        )
        for k, v in name_dict.items():
            if v == "Tumor":
                name_dict[k] = "Tumor"
            else:
                name_dict[k] = "non_Tumor"
        name_dict = {str(k): v for k, v in name_dict.items()}

        update_geojson_classification(
            geojson_f=output_f,
            output_f=output_dir / f"{tma_core}_tumor.geojson",
            name_dict=name_dict,
        )


# %%
################################################################################
# segmentation
################################################################################

if False:
    setup_gpu("1")

    # one folder for each unit, within the folder there should be one ome.tiff file
    unit_dir = "/path/to/unit_dir"
    internal_markers = ["DAPI"]
    boundary_markers = ["CD45", "CD3e", "CD163", "NaKATP"]
    pixel_size_um = 0.5068164319979996
    maxima_threshold = 0.075
    interior_threshold = 0.20
    # whether to perform Otsu thresholding for internal and boundary markers
    thresh_otsu = False
    # not important >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    thresh_q_min = 0
    thresh_q_max = 0.99
    scale = True
    ometiff_path = None
    # not important <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # name of subfolder under unit_dir to store segmentation results
    tag = "some_tag"
    run_segmentation_mesmer_cell(
        unit_dir=unit_dir,
        internal_markers=internal_markers,
        boundary_markers=boundary_markers,
        thresh_q_min=thresh_q_min,
        thresh_q_max=thresh_q_max,
        thresh_otsu=thresh_otsu,
        scale=scale,
        pixel_size_um=pixel_size_um,
        maxima_threshold=maxima_threshold,
        interior_threshold=interior_threshold,
        tag=tag,
        ometiff_path=ometiff_path,
    )
    # whether to generate a geojson file for the segmentation mask
    if False:
        generate_segmentation_mask_geojson(unit_dir, tag)
