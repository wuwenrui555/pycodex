import itertools
import logging
import os
import pickle as pkl
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from IPython.display import display
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from pycodex import io, metadata

########################################################################################################################
# SIFH
########################################################################################################################


class SIFTMatcher:
    """
    A class for SIFT (Scale-Invariant Feature Transform) feature matching between two images.
    """

    def __init__(self, im_src, im_dst, nfeatures=10000, step=8):
        """
        Initializes the SIFT matcher object with source and destination images.

        Args:
            im_src (ndarray): Source image in grayscale where keypoints and descriptors will be detected.
            im_dst (ndarray): Destination image in grayscale for matching keypoints from the source image.
            nfeatures (int, optional): The maximum number of features to retain in SIFT detection. Default is 10000.
            step (int, optional): Step size for downsampling the images to speed up computation. Default is 8.
        """
        if im_src.dtype.name == "uint16":
            im_src = (im_src / 256).astype("uint8")
        if im_dst.dtype.name == "uint16":
            im_dst = (im_dst / 256).astype("uint8")
        self.im_src = im_src[::step, ::step]
        self.im_dst = im_dst[::step, ::step]
        self.step = step
        self.nfeatures = nfeatures

        print("Initialize SIFT detector")
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        print("Find keypoints and descriptors")
        self.kp1, self.des1 = self.sift.detectAndCompute(self.im_src, None)
        self.kp2, self.des2 = self.sift.detectAndCompute(self.im_dst, None)
        print("Initialize matcher")
        self.matcher = cv2.BFMatcher()
        print("Match descriptors")
        self.matches = self.matcher.knnMatch(self.des1, self.des2, k=2)

    def find_kp_intersections(self, ratio_min=0.1, ratio_max=0.9, ratio_step=0.05):
        """
        Get the number of keypoint connection intersections across different Lowe ratio thresholds to determine the
        optimal value that reduces intersecting matches.

        Args:
            ratio_min (float): The minimum ratio to start with. Default is 0.1.
            ratio_max (float): The maximum ratio to end with. Default is 0.9.
            ratio_step (float): The step size for changing the ratio. Default is 0.05.
        """

        def ccw(A, B, C):
            """
            Checks if three points A, B, C are arranged in a counterclockwise order.

            Args:
                A (tuple): The first point (x, y).
                B (tuple): The second point (x, y).
                C (tuple): The third point (x, y).

            Returns:
                bool: True if the points are in counterclockwise order, otherwise False.
            """
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def is_intersect(p1, q1, p2, q2):
            """
            Checks if two line segments (p1, q1) and (p2, q2) intersect.

            Args:
                p1 (tuple): The starting point of the first line segment (x, y).
                q1 (tuple): The ending point of the first line segment (x, y).
                p2 (tuple): The starting point of the second line segment (x, y).
                q2 (tuple): The ending point of the second line segment (x, y).

            Returns:
                bool: True if the two line segments intersect, otherwise False.
            """
            # Check if the endpoints of the two line segments are on different sides
            return ccw(p1, p2, q2) != ccw(q1, p2, q2) and ccw(p1, q1, p2) != ccw(p1, q1, q2)

        # Iterate over different ratio thresholds
        intersections = []
        for ratio in tqdm(np.arange(ratio_min, ratio_max, ratio_step)):
            good_matches = [m for m, n in self.matches if m.distance <= ratio * n.distance]
            n_combination = 0
            n_intersection = 0

            # Check if any match lines intersect
            for m1, m2 in itertools.combinations(good_matches, 2):
                p1, q1 = self.kp1[m1.queryIdx].pt, self.kp2[m1.trainIdx].pt
                p2, q2 = self.kp1[m2.queryIdx].pt, self.kp2[m2.trainIdx].pt
                if is_intersect(p1, q1, p2, q2):
                    n_intersection += 1
                n_combination += 1
            if n_combination > 0:
                intersections.append([ratio, n_intersection, n_combination, n_intersection / n_combination])
        self.intersections = pd.DataFrame(
            intersections,
            columns=["ratio", "n_intersection", "n_combination", "percentage"],
        )

    def plot_kp_intersections(self, figsize=(10, 5)):
        """
        Plots the percentage of intersecting matches for different Lowe's ratio thresholds.

        Args:
            figsize (tuple): The size of the figure for the plot. Default is (10, 5).

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        ratios = self.intersections["ratio"]
        percentages = self.intersections["percentage"]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(ratios, percentages, marker="o", linestyle="-", color="b")
        ax.set_xticks(ratios)
        ax.set_xlabel("Lowe's Ratio Threshold")
        ax.set_ylabel("Intersection Percentage")
        ax.set_title("SIFT Matcher Intersections")

        plt.tight_layout()
        plt.close(fig)

        return fig

    def set_lowe_ratio_threshold(self, ratio_threshold=0.60, figsize=(15, 15)):
        """
        Filters the matches based on Lowe's ratio test and plots the matching result.

        Args:
            ratio_threshold (float, optional): A threshold value for Lowe's ratio test to determine good matches.
                Matches with a ratio of the distance between the closest and second-closest neighbors
                lower than this threshold will be considered good matches. Default is 0.60.
            figsize (tuple, optional): The size of the figure for plotting the matched features. Default is (15, 15).

        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        print("Apply Lowe's ratio threshold")
        self.ratio_threshold = ratio_threshold
        self.good_matches = [m for m, n in self.matches if m.distance / n.distance < ratio_threshold]

        # Get the coordinates for lines to be drawn
        img_matches = cv2.drawMatches(
            img1=self.im_src,
            keypoints1=self.kp1,
            img2=self.im_dst,
            keypoints2=self.kp2,
            matches1to2=self.good_matches,
            outImg=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_matches)        
        plt.close(fig)
        
        return fig

    def compute_affine_matrix(self, ratio_threshold=None):
        """
        Computes the affine transformation matrix between the source and destination images.

        Args:
            ratio_threshold (float, optional): A threshold value for Lowe's ratio test to determine good matches.
                Default is the value set by set_lowe_ratio_threshold().

        Returns:
            tuple: A tuple containing:
                - ndarray: The 2x3 affine transformation matrix from source image to destination image.
                - ndarray: The 2x3 inverse affine transformation matrix from destination image to source image.
        """
        if ratio_threshold is None:
            ratio_threshold = self.ratio_threshold
        self.good_matches = [m for m, n in self.matches if m.distance / n.distance < ratio_threshold]

        print("Extract matched keypoints")
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
        print("Compute affine transformation")
        H, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        H[0, 2] *= self.step
        H[1, 2] *= self.step
        H_inverse = np.linalg.inv(np.vstack((H, [0, 0, 1])))[:2, :]

        return H, H_inverse


def apply_affine_transformation(im_src: np.ndarray, output_shape: tuple[int, int], H_inverse: np.ndarray):
    """
    Applies an affine transformation to warp the source image to the destination image's coordinate space.

    Args:
        im_src (ndarray): The source image to be transformed.
        output_shape (tuple): Shape of output image (height, width), which is the shape of destination image generally.
        H_inverse (ndarray): The inverse affine transformation matrix from destination image to source image.

    Returns:
        tuple: A tuple containing:
            - ndarray: The transformed source image.
            - ndarray: A boolean mask marking out-of-bounds regions (where value == 0).
    """
    # Generate grid coordinates for the destination image
    dst_y, dst_x = np.indices(output_shape)
    dst_coords = np.stack([dst_x.ravel(), dst_y.ravel(), np.ones(dst_x.size)])

    # Map destination coordinates to source coordinates using the inverse affine matrix
    src_coords = H_inverse @ dst_coords
    src_x = src_coords[0, :].reshape(output_shape)
    src_y = src_coords[1, :].reshape(output_shape)

    # Check for out-of-bounds coordinates before interpolation
    valid_mask = (src_x >= 0) & (src_x < im_src.shape[1]) & (src_y >= 0) & (src_y < im_src.shape[0])

    # Initialize the transformed image with zeros (for out-of-bounds regions)
    transformed_im_src = np.zeros(output_shape, dtype=im_src.dtype)

    # order=1: bilinear interpolation is used to estimate the pixel value non-integer coordinates
    # mode="constant", cval=0: fills the out-of-bounds coordinates with a constant 0.
    transformed_im_src[valid_mask] = map_coordinates(
        im_src, [src_y[valid_mask], src_x[valid_mask]], order=1, mode="constant", cval=0
    )

    # Create a mask for out-of-bounds regions (where valid_mask is False)
    blank_mask = ~valid_mask

    # Ensure the transformed image values are within the valid range and convert to uint16
    transformed_im_src = np.clip(transformed_im_src, 0, 65535).astype(np.uint16)

    return transformed_im_src, blank_mask


def apply_blank_mask(im, blank_mask):
    """
    Applies a blank mask to an image, setting masked regions to 0.

    Args:
        im (ndarray): The input image to be masked.
        blank_mask (ndarray): A boolean mask where True indicates out-of-bounds regions.

    Returns:
        ndarray: The masked image with out-of-bounds regions set to 0.
    """
    im[blank_mask] = 0
    masked_im = np.clip(im, 0, 65535).astype(np.uint16)

    return masked_im

########################################################################################################################
# Align Images of Different Runs
########################################################################################################################


def sift_align_src_on_dst_coordinate(
    id: str,
    dir_dst: str,
    dir_src: str,
    path_parameter: str,
    dir_output: str,
    src_rot90cw: int = 0,
    src_hflip: bool = False,
    name_output_dst: str = "dst",
    name_output_src: str = "src",
):
    """
    Aligns and processes images from two separate runs (align source image on coordinates of destination image).

    Parameters
    ----------
    id: str
        Unique identifier for the alignment.
    dir_dst : str
        Path to the directory containing destination images.
    dir_src : str
        Path to directory containing source images will be aligned to the coordinates of the destination images.
    path_parameter : str
        Path to the `sift_parameter.pkl` file storing parameters for the SIFT alignment process, generated iteratively
        using the `01_sift_plot_src_on_dst_coordinate.ipynb` notebook.
    dir_output : str
        Path to directory where the alignment results are saved.
    src_rot90cw : int, optional
        Number of times to rotate the source images 90 degrees clockwise. Defaults to 0 (no rotation).
    src_hflip : bool, optional
        If True, horizontally flips the source images. Defaults to False.
    name_output_dst : str, optional
        The name of subfolder under `dir_output` for saving output of the processed destination images. Defaults to "dst".
    name_output_src : str, optional
        The name of subfolder under `dir_output` for saving output of the aligned source images. Defaults to "src".

    Returns
    -------
    None
    """

    try:
        # Load SIFT parameters
        with open(path_parameter, "rb") as f:
            data = pkl.load(f)
        logging.info("SIFT parameters loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Parameter file not found: {path_parameter}")
        return

    # Load marker lists and metadata
    dst_metadata_dict = io.organize_metadata_fusion(dir_dst, subfolders=False)
    dst_unique_markers, _, _, _ = metadata.summary_markers(dst_metadata_dict)
    src_metadata_dict = io.organize_metadata_fusion(dir_src, subfolders=False)
    src_unique_markers, _, _, _ = metadata.summary_markers(src_metadata_dict)

    # Rename markers to avoid duplicates
    dst_unique_markers_renamed = io.rename_duplicate_markers(dst_unique_markers)
    src_unique_markers_renamed = io.rename_duplicate_markers(src_unique_markers)

    # Load source and destination image file information
    def load_tiff_files(directory: str) -> dict[str, str]:
        """
        Helper function to load TIFF image file names from a specified directory.

        Parameters
        ----------
        directory : str
            Path to the directory containing the image files.

        Returns
        -------
        dict[str, str]
            A dictionary where the keys are the file names (without extensions)
            and the values are the full file names (with extensions) of TIFF files
            in the directory.
        """
        tiff_files = {
            os.path.splitext(file)[0]: file
            for file in os.listdir(directory)
            if os.path.splitext(file)[1].lower() in [".tiff", ".tif"]
        }
        return tiff_files

    src_files_dict = load_tiff_files(dir_src)
    dst_files_dict = load_tiff_files(dir_dst)

    # Align and process images
    def process_image(marker, path_marker, output_path, transform=True):
        """
        Helper function to align and process images with optional transformations.

        Parameters:
        -----------
        marker : str
            The name or identifier for the marker image being processed.
        path_marker : str
            The file path to the marker image to be processed.
        output_path : str
            The file path where the processed image will be saved.
        transform : bool, optional
            If True, the function applies transformations (rotation, flip, affine transformations, and masking).
            If False, only a blank mask is applied. Default is True.

        Returns:
        --------
        None
        """
        if os.path.exists(path_marker):
            im = tifffile.imread(path_marker)
            if transform:
                effective_rotation = src_rot90cw % 4
                if effective_rotation != 0:
                    rotateCode = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}[
                        effective_rotation
                    ]
                    im = cv2.rotate(im, rotateCode)
                if src_hflip:
                    im = cv2.flip(im, 1)
                logging.info(f"{marker}: image loaded and transformed")

                im_warped, _ = apply_affine_transformation(im, data["output_shape"], data["H_inverse"])
                logging.info(f"{marker}: affine transformation applied")

                im_masked = apply_blank_mask(im_warped, data["blank_mask"])
                logging.info(f"{marker}: blank mask applied")
            else:
                im_masked = apply_blank_mask(im, data["blank_mask"])
                logging.info(f"{marker}: blank mask applied without transformation")
            tifffile.imwrite(output_path, im_masked)
            logging.info(f"{marker}: saved successfully to {output_path}")
        else:
            logging.warning(f"File not found: {path_marker}")

    ## Source images
    dir_output_src = os.path.join(dir_output, id, name_output_src)
    os.makedirs(dir_output_src, exist_ok=True)
    for i, marker in tqdm(enumerate(src_unique_markers), desc="Source images", total=len(src_unique_markers)):
        path_marker = os.path.join(dir_src, src_files_dict.get(marker, ""))
        path_output = os.path.join(dir_output_src, f"{src_unique_markers_renamed[i]}.tiff")
        process_image(marker, path_marker, path_output, transform=True)

    ## Destination images
    dir_output_dst = os.path.join(dir_output, id, name_output_dst)
    os.makedirs(dir_output_dst, exist_ok=True)
    for i, marker in tqdm(enumerate(dst_unique_markers), desc="Destination images", total=len(dst_unique_markers)):
        path_marker = os.path.join(dir_dst, dst_files_dict.get(marker, ""))
        path_output = os.path.join(dir_output_dst, f"{dst_unique_markers_renamed[i]}.tiff")
        process_image(marker, path_marker, path_output, transform=False)


def display_aligned_markers(dir_id: str):
    """
    Displays the aligned markers of different runs.

    Parameters
    ----------
    dir_id : str
        Path to the directory containing the aligned markers.

    Returns
    -------
    None
    """
    # Load the metadata
    metadata_id = io.organize_metadata_keyence(dir_id, subfolders=True)
    df_metadata_id = pd.concat([value for key, value in metadata_id.items()])
    df_metadata_id.rename(
        columns={
            "region": "run",
            "_region": "region",
        },
        inplace=True,
    )

    # Pivot the DataFrame
    df_pivoted = df_metadata_id.pivot_table(
        index=["run", "region", "cycle"],
        columns="channel",
        values="marker",
        aggfunc=",".join,
    ).fillna("")
    df_pivoted.columns.name = None
    df_pivoted.reset_index(inplace=True)
    df_pivoted.rename(
        columns={
            "ch001": "ch001 (DAPI)",
            "ch002": "ch002 (AF488)",
            "ch003": "ch003 (ATTO550/cy3)",
            "ch004": "ch004 (AF647/cy5)",
        },
        inplace=True,
    )

    # Display the DataFrame by run
    for run in sorted(df_pivoted["run"].unique()):
        display(df_pivoted[df_pivoted["run"] == run])


def export_marker_metadata_keyence(dir_region: str, dir_output: str):
    """
    Exports marker metadata to an Excel file with multiple sheets.

    This function processes metadata for a given region directory, organizes the marker information,
    and writes the data into an Excel file. Each run's metadata is saved as a separate sheet, and a
    summary sheet for marker order is included.

    Parameters
    ----------
    dir_region : str
        The path to the region directory containing marker metadata for processing.
    dir_output : str
        The path to the output directory where the Excel file will be saved.

    Returns
    -------
    None
        The function saves an Excel file with multiple sheets to the specified output directory.

    Notes
    -----
    - Each sheet corresponds to a run and contains the following columns:
        - `marker`: Marker names.
        - `is_blank`: Boolean flag indicating whether the marker is a blank.
        - `is_dapi`: Boolean flag indicating whether the marker is a DAPI marker.
        - `is_marker`: Boolean flag indicating whether the marker is a valid marker (not blank or DAPI).
        - `is_kept`: An empty column for user-defined marker retention.
    - The `marker_order` sheet contains column headers for each run for user-defined marker order.
    """
    os.makedirs(dir_output, exist_ok=True)

    region_id = os.path.basename(dir_region)
    path_excel = os.path.join(dir_output, f"{region_id}.xlsx")
    with pd.ExcelWriter(path_excel) as writer:
        # sheet for marker metadata of each run
        marker_metadata = io.organize_metadata_keyence(dir_region, subfolders=True)
        run_list = sorted(marker_metadata.keys())
        for run in run_list:
            df_marker = marker_metadata[run]
            df_marker = df_marker.drop(columns=["region"])
            df_marker["marker_name"] = df_marker["marker"]
            df_marker["marker"] = run + "-" + df_marker["marker"]
            # add flag columns
            df_marker["is_blank"] = df_marker["marker_name"].str.contains("blank", case=False, na=False)
            df_marker["is_dapi"] = df_marker["marker_name"].str.match(r"Ch\d+Cy\d+", flags=re.IGNORECASE)
            df_marker["is_marker"] = (~df_marker["is_blank"]) & (~df_marker["is_dapi"])
            df_marker["is_kept"] = ""
            # save to Excel with the run name as the sheet name
            df_marker.to_excel(writer, sheet_name=run, index=False)


def export_marker_metadata_fusion(dir_region: str, dir_output: str):
    """
    Exports marker metadata to an Excel file with multiple sheets.

    This function processes metadata for a given region directory, organizes the marker information,
    and writes the data into an Excel file. Each run's metadata is saved as a separate sheet, and a
    summary sheet for marker order is included.

    Parameters
    ----------
    dir_region : str
        The path to the region directory containing marker metadata for processing.
    dir_output : str
        The path to the output directory where the Excel file will be saved.

    Returns
    -------
    None
        The function saves an Excel file with multiple sheets to the specified output directory.

    Notes
    -----
    - Each sheet corresponds to a run and contains the following columns:
        - `marker`: Marker names.
        - `is_blank`: Boolean flag indicating whether the marker is a blank.
        - `is_dapi`: Boolean flag indicating whether the marker is a DAPI marker.
        - `is_marker`: Boolean flag indicating whether the marker is a valid marker (not blank or DAPI).
        - `is_kept`: An empty column for user-defined marker retention.
    - The `marker_order` sheet contains column headers for each run for user-defined marker order.
    """
    os.makedirs(dir_output, exist_ok=True)

    region_id = os.path.basename(dir_region)
    path_excel = os.path.join(dir_output, f"{region_id}.xlsx")
    with pd.ExcelWriter(path_excel) as writer:
        # sheet for marker metadata of each run
        marker_metadata = io.organize_metadata_fusion(dir_region, subfolders=True)
        run_list = sorted(marker_metadata.keys())
        for run in run_list:
            df_marker = marker_metadata[run]
            df_marker = df_marker.drop(columns=["region"])
            df_marker["marker_name"] = df_marker["marker"]
            df_marker["marker"] = run + "-" + df_marker["marker"]
            # add flag columns
            df_marker["is_blank"] = df_marker["marker_name"].str.contains("blank", case=False, na=False)
            df_marker["is_dapi"] = df_marker["marker_name"].str.contains("dapi", case=False, na=False)
            df_marker["is_marker"] = (~df_marker["is_blank"]) & (~df_marker["is_dapi"])
            df_marker["is_kept"] = ""
            # save to Excel with the run name as the sheet name
            df_marker.to_excel(writer, sheet_name=run, index=False)