import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from tqdm import tqdm

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
                Default is 0.60.

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


def apply_affine_transformation(im_src, im_dst, H_inverse):
    """
    Applies an affine transformation to warp the source image to the destination image's coordinate space.

    Args:
        im_src (ndarray): The source image to be transformed.
        im_dst (ndarray): The destination image (used for shape reference).
        H_inverse (ndarray): The inverse affine transformation matrix from destination image to source image.

    Returns:
        tuple: A tuple containing:
            - ndarray: The transformed source image.
            - ndarray: A boolean mask marking out-of-bounds regions (where value == 0).
    """
    # Generate grid coordinates for the destination image
    output_shape = im_dst.shape
    dst_y, dst_x = np.indices(output_shape)
    dst_coords = np.stack([dst_x.ravel(), dst_y.ravel(), np.ones(dst_x.size)])

    # Map destination coordinates to source coordinates using the inverse affine matrix
    src_coords = H_inverse @ dst_coords
    src_x = src_coords[0, :].reshape(output_shape)
    src_y = src_coords[1, :].reshape(output_shape)

    # Add 1 to the source image to distinguish valid pixels from out-of-bounds areas (value == 0)
    im_src += 1

    # order=1: bilinear interpolation is used to estimate the pixel value non-integer coordinates
    # mode="constant", cval=0: fills the out-of-bounds coordinates with a constant 0.
    transformed_im_src = map_coordinates(im_src, [src_y, src_x], order=1, mode="constant", cval=0)

    # Create a mask indicating out-of-bounds regions in the transformed image
    blank_mask = transformed_im_src == 0

    # Subtract 1 from the transformed image to restore original value range and convert to uint16 range [0, 65535]
    transformed_im_src -= 1
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
