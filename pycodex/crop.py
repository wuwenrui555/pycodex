import math

import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
# crop regions with qupath measurement
########################################################################################################################


def parse_crop_data(crop_data_path):
    """
    Parse tab-separated text data to extract centroids, areas, perimeters, and names of shapes.

    Parameters:
    text (str): The tab-separated input data as a string.

    Returns:
    tuple: Lists of centroids, areas, perimeters, and names.
    """
    # Initialize lists to store parsed data
    centroids = []
    areas = []
    perimeters = []
    names = []

    with open(crop_data_path, "r") as f:
        crop_data = f.read().splitlines()
    columns = crop_data[0].split("\t")

    # Find the column indices for the required fields
    centroid_x_index = columns.index("Centroid X µm")
    centroid_y_index = columns.index("Centroid Y µm")
    area_index = columns.index("Area µm^2")
    perimeter_index = columns.index("Perimeter µm")
    name_index = columns.index("Name")

    # Parse each line after the header
    for line in crop_data[1:]:
        fields = line.split("\t")
        centroid_x = float(fields[centroid_x_index])
        centroid_y = float(fields[centroid_y_index])
        area = float(fields[area_index])
        perimeter = float(fields[perimeter_index])
        name = str(fields[name_index])

        # Append parsed values to respective lists
        centroids.append((centroid_x, centroid_y))
        areas.append(area)
        perimeters.append(perimeter)
        names.append(name)

    return centroids, areas, perimeters, names


def calculate_rectangle(
    centroid_x, centroid_y, area, perimeter, name, pixel_width, pixel_height
):
    """
    Calculate the bounding box coordinates of a rectangle based on centroid, area, and perimeter.

    Parameters:
    centroid_x (float): X-coordinate of the centroid in micrometers.
    centroid_y (float): Y-coordinate of the centroid in micrometers.
    area (float): Area of the shape in square micrometers.
    perimeter (float): Perimeter of the shape in micrometers.
    name (str): 如果不为空，则与x轴平行的边为长边
    pixel_width (float): Width of a pixel in micrometers.
    pixel_height (float): Height of a pixel in micrometers.

    Returns:
    tuple: Coordinates of the bounding box (x1, y1, x2, y2).
    """
    # Convert area and perimeter from micrometers to pixels
    area_px = area / (pixel_width * pixel_height)
    perimeter_px = perimeter / ((pixel_width + pixel_height) / 2)

    # Calculate half perimeter and the discriminant for solving the quadratic equation
    P_half = perimeter_px / 2
    discriminant = P_half**2 - 4 * area_px

    # Solve for width and height based on the discriminant
    if discriminant < 0:
        raise ValueError("Invalid shape dimensions, unable to calculate rectangle.")

    if name:
        width_px = (P_half - math.sqrt(discriminant)) / 2
    else:
        width_px = (P_half + math.sqrt(discriminant)) / 2
    height_px = P_half - width_px

    # Convert centroid coordinates to pixels
    centroid_x_px = centroid_x / pixel_width
    centroid_y_px = centroid_y / pixel_height

    # Calculate bounding box coordinates
    x1 = centroid_x_px - width_px / 2
    y1 = centroid_y_px - height_px / 2
    x2 = centroid_x_px + width_px / 2
    y2 = centroid_y_px + height_px / 2

    return int(x1), int(y1), int(x2), int(y2)


def crop_and_display(
    image, centroids, areas, perimeters, names, pixel_width, pixel_height
):
    """
    Crop and display regions from an image based on centroid, area, and perimeter data.

    Parameters:
    image (ndarray): The input image.
    centroids (list): List of centroid coordinates (tuples).
    areas (list): List of areas corresponding to each shape.
    perimeters (list): List of perimeters corresponding to each shape.
    names (list): List of names corresponding to each shape.
    pixel_width (float): Width of a pixel in micrometers.
    pixel_height (float): Height of a pixel in micrometers.

    Returns:
    list: List of bounding box coordinates for each cropped region.
    """
    rectangles = []

    # Iterate over all centroids and calculate bounding boxes
    for i in range(len(centroids)):
        centroid_x, centroid_y = centroids[i]
        area = areas[i]
        perimeter = perimeters[i]
        name = names[i]

        # Calculate bounding box coordinates
        x1, y1, x2, y2 = calculate_rectangle(
            centroid_x, centroid_y, area, perimeter, name, pixel_width, pixel_height
        )
        rectangles.append((x1, y1, x2, y2))

        # Crop the image using calculated bounding box and display
        cropped_img = image[0, y1:y2, x1:x2]

        plt.imshow(np.log1p(cropped_img), cmap="gray")
        plt.title(f"reg{i + 1:03}")
        plt.show()

    return rectangles


########################################################################################################################
# crop image into into equal-sized blocks
########################################################################################################################


def crop_image_into_blocks(
    shape: tuple[int, int], max_block_size: int = 3000
) -> dict[str, np.ndarray]:
    """
    Divide a large image into equal-sized blocks with dimensions <= max_block_size.

    Args:
        shape (tuple[int, int]):
            A tuple representing the height and width of the input image (e.g., (height, width)).
        max_block_size (int):
            The maximum allowed size for any block along both dimensions (height and width).
            Defaults to 3000.

    Returns:
        dict[str, np.ndarray]: A dictionary where each key is a subregion name (e.g., "sub001") and the value is a
        tuple containing: (x_beg, x_end, y_beg, y_end) - The start and end coordinates for the block.
    """
    height, width = shape

    # Find the optimal number of blocks along each dimension
    num_blocks_y = (height + max_block_size - 1) // max_block_size  # Ceiling division
    num_blocks_x = (width + max_block_size - 1) // max_block_size

    # Calculate the size of each block to ensure equal sizes
    block_height = height // num_blocks_y
    block_width = width // num_blocks_x

    # Ensure all blocks are of equal size by using slicing
    xy_limits = {}
    idx = 1
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            y_beg = i * block_height
            y_end = (i + 1) * block_height if i != num_blocks_y - 1 else height
            x_beg = j * block_width
            x_end = (j + 1) * block_width if j != num_blocks_x - 1 else width
            xy_limits[f"sub{idx:03d}"] = (x_beg, x_end, y_beg, y_end)
            idx += 1
    return xy_limits


def plot_block_labels(
    image: np.ndarray, xy_limits: dict[str, tuple[int, int, int, int]]
) -> plt.Figure:
    """
    Plot the given image with bounding boxes and labels inside each block.

    Args:
        image (np.ndarray):
            The input 2D image to display (e.g., segmentation mask).
        xy_limits (dict[str, tuple[int, int, int, int]]):
            A dictionary where each key is a subregion name (e.g., "sub001") and the value is a tuple containing:
            (x_beg, x_end, y_beg, y_end) - The start and end coordinates for the block.

    Returns:
        plt.Figure: The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")

    for label, (x_beg, x_end, y_beg, y_end) in xy_limits.items():
        # Draw a rectangle (bounding box)
        rect = plt.Rectangle(
            (x_beg, y_beg),
            x_end - x_beg,
            y_end - y_beg,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add label text inside the rectangle (centered)
        ax.text(
            (x_beg + x_end) / 2,
            (y_beg + y_end) / 2,
            label,
            color="white",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )

    plt.axis("on")
    plt.close(fig)

    return fig
