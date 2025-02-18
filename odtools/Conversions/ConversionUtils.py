import random
import shutil
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mung.node import Node


def get_num_pixels(filepath):
    return Image.open(filepath).size


def split_dataset(
        data: list[tuple[Path, Path]],
        split_ratio=0.9,
        seed=42
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """
    Split dataset into train and test sets based on given ratio.

    :param data: The dataset to be split.
    :param split_ratio: The ratio of train and test sets.
    :param seed: Seed for the random number generator.
    :return: Train and test sets as two lists of tuples (image, annotation).
    """
    # shuffle by given seed
    random.Random(seed).shuffle(data)
    # actually split the dataset
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    return train_data, val_data


def find_convex_hull(binary_array: np.ndarray, show_plot: bool = False) -> list:
    """
    Given a 2D ndarray of 0s and 1s, find and show_plot its convex hull.

    :param binary_array: mask of 0s and 1s
    :param show_plot: whether to show the convex hull
    """

    # binary array -> uint8 path_to_image
    binary_image = (binary_array * 255).astype(np.uint8)
    # find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        warnings.warn("No contours found.")
        return []
    # take first contour
    contour = contours[0]
    # find convex hull
    hull = cv2.convexHull(contour, clockwise=True)

    if show_plot:
        # plot binary array
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_array, cmap='gray')
        # plot contour
        plt.scatter(contour[:, 0, 0], contour[:, 0, 1], color='blue', label='Contour')
        # plot convex hull points
        plt.plot(hull[:, 0, 0], hull[:, 0, 1], color='red', linewidth=2, label='Convex Hull')

        plt.title('Convex Hull of Binary Array')
        plt.legend()

        plt.show()

    # flatten outputted list
    return [x[0] for x in hull]


def draw_shapes_on_image(image_path: str | Path, coordinates_list: list[tuple[int, int]]):
    """
    Draws multiple 2D shapes from lists of relative coordinates onto a given path_to_image.

    :param image_path: path to path_to_image
    :param coordinates_list: list of relative coordinates
    """

    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    for relative_coordinates in coordinates_list:
        # convert from relative to absolute
        absolute_coordinates = [(int(x * img_width), int(y * img_height)) for x, y in relative_coordinates]

        # first line is blue, to communicate mask orientation
        if len(absolute_coordinates) > 1:
            cv2.line(img, absolute_coordinates[0], absolute_coordinates[1], color=(255, 0, 0), thickness=2,
                     lineType=cv2.LINE_AA)

        # remaining lines
        for j in range(1, len(absolute_coordinates) - 1):
            cv2.line(img, absolute_coordinates[j], absolute_coordinates[j + 1], color=(0, 0, 255), thickness=2)

        # connect first and last
        cv2.line(img, absolute_coordinates[0], absolute_coordinates[-1], color=(0, 0, 255), thickness=2,
                 lineType=cv2.LINE_AA)

        # draw first point brow, orientation
        x, y = absolute_coordinates[0]
        cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

    # show path_to_image
    cv2.namedWindow("Segmentation masks visualization", cv2.WINDOW_NORMAL)
    cv2.imshow("Segmentation masks visualization", img)
    # resize to fit
    cv2.resizeWindow("Segmentation masks visualization", 800, 600)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mung_segmentation_to_absolute_coordinates(node: Node) -> list[tuple[int, int]]:
    # if mask is None, then the segmentation mask is the whole bounding box (via documentation)
    if node.mask is None:
        shifted_coordinates = [
            (node.left, node.top),
            (node.left, node.bottom),
            (node.right, node.bottom),
            (node.right, node.top),
        ]
        return shifted_coordinates

    # get mask coordinates
    coordinates = find_convex_hull(node.mask)
    # at least three coordinates are required to form a mask.
    # if this is not true, use whole bounding box as a mask
    if len(coordinates) < 3:
        shifted_coordinates = [
            (node.left, node.top),
            (node.left, node.bottom),
            (node.right, node.bottom),
            (node.right, node.top),
        ]
    else:
        # shift retrieved coordinates and make them relative
        shifted_coordinates = [
            (int(point[0] + node.left), int(point[1] + node.top)) for point in coordinates]

    return shifted_coordinates


def copy_and_resize_image(image_path: Path | str, output_path: Path | str, max_size: int = None) -> None:
    """
    Resizes path_to_image so that its larger side equals max_size while maintaining aspect ratio.
    Uses bilinear interpolation for resizing.

    :param image_path: path to path_to_image
    :param output_path: path to output path_to_image
    :param max_size: maximum size of path_to_image
    """

    if max_size is None:
        shutil.copy(image_path, output_path)
        return

    with Image.open(image_path) as img:
        # Get original dimensions
        width, height = img.size

        # Determine the scaling factor to resize the larger side to max_size
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)

        # Resize the path_to_image using bilinear interpolation
        img_resized = img.resize((new_width, new_height), Image.BILINEAR)

        # Save the resized path_to_image back to the same path
        img_resized.save(output_path)
