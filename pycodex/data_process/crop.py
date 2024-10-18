import math

import matplotlib.pyplot as plt
import numpy as np


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
        plt.title(f"reg{i+1:03}")
        plt.show()

    return rectangles
