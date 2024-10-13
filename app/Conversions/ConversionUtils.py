import random
import shutil
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mung.node import Node

from .COCO import MungToCOCO
from .YOLO import MungToYOLO


def format_dataset(
        images_path: Path,
        annotations_path: Path,
        dataset_path: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        output_format: str = "coco",
        mode: str = None,
        split_ratio: float = 0.9,
        resize: int = None,
        seed: int = 42,
        image_format: str = "jpg"
) -> None:
    """
    Finds all images and annotations inside given files
    and processes them according to the split ratio and output format.

    :param images_path: path to directory with images
    :param annotations_path: path to directory with labels
    :param dataset_path: path to final dataset, all data will be outputted here

    :param class_reference_table: dictionary, a function that assigns class id by class name
    :param class_output_names: list of class names

    :param output_format: "coco" or "yolo", defines output format
    :param mode: detection or segmentation
    :param split_ratio: train/test split ratio

    :param resize: resizes images so that the longer side is this many pixels long
    :param seed: seed for dataset shuffling
    :param image_format: format in which the images are saved
    """
    # check parameters
    if mode is not None and output_format == "coco":
        warnings.warn("COCO format exports in both modes (\"detection\", \"segmentation\")at the same time.")
    if mode is not None and mode not in ["detection", "segmentation"]:
        raise ValueError("mode must be either \"detection\" or \"segmentation\"")
    if mode is None:
        mode = "detection"

    if output_format not in ["coco", "yolo"]:
        raise ValueError("output_format must be either \"coco\" or \"yolo\"")

    # load data from given paths
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.xml")))
    data = list(zip(images, annotations))

    # dump everything into one directory
    if split_ratio == 1.0:
        # set up folders
        images_dir = dataset_path / "images"
        annot_dir = dataset_path / "labels"

        images_dir.mkdir(exist_ok=True, parents=True)
        annot_dir.mkdir(exist_ok=True, parents=True)

        if output_format == "coco":
            MungToCOCO.process_mung_batch_to_coco(
                data,
                (images_dir, annot_dir),
                class_reference_table=class_reference_table,
                class_output_names=class_output_names,
                image_format=image_format,
                resize=resize,
            )
        elif output_format == "yolo":
            MungToYOLO.process_mung_batch_to_yolo(
                data,
                (images_dir, annot_dir),
                class_reference_table=class_reference_table,
                class_output_names=class_output_names,
                image_format=image_format,
                resize=resize,
                mode=mode,
            )

    # split to train/test
    else:
        # set up folders
        train_images_dir = dataset_path / "images" / "train"
        val_images_dir = dataset_path / "images" / "val"
        train_annot_dir = dataset_path / "labels" / "train"
        val_labels_dir = dataset_path / "labels" / "val"

        train_images_dir.mkdir(exist_ok=True, parents=True)
        val_images_dir.mkdir(exist_ok=True, parents=True)
        train_annot_dir.mkdir(exist_ok=True, parents=True)
        val_labels_dir.mkdir(exist_ok=True, parents=True)

        # split
        train_data, val_data = split_dataset(data, split_ratio=split_ratio, seed=seed)

        if output_format == "coco":
            # train
            MungToCOCO.process_mung_batch_to_coco(
                train_data,
                (train_images_dir, train_annot_dir),
                class_reference_table=class_reference_table,
                class_output_names=class_output_names,
                image_format=image_format,
                resize=resize,
            )
            # val
            MungToCOCO.process_mung_batch_to_coco(
                val_data,
                (val_images_dir, val_labels_dir),
                class_reference_table=class_reference_table,
                class_output_names=class_output_names,
                image_format=image_format,
                resize=resize,
            )
        elif output_format == "yolo":
            # train
            MungToYOLO.process_mung_batch_to_yolo(
                train_data,
                (train_images_dir, train_annot_dir),
                class_reference_table=class_reference_table,
                class_output_names=class_output_names,
                image_format=image_format,
                resize=resize,
                mode=mode,
            )
            # val
            MungToYOLO.process_mung_batch_to_yolo(
                val_data,
                (val_images_dir, val_labels_dir),
                class_reference_table=class_reference_table,
                class_output_names=class_output_names,
                image_format=image_format,
                resize=resize,
                mode=mode,
            )


def get_num_pixels(filepath):
    return Image.open(filepath).size


def split_dataset(data: list[tuple[Path, Path]], split_ratio=0.9, seed=42):
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

    # binary array -> uint8 image
    binary_image = (binary_array * 255).astype(np.uint8)
    # find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found.")
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
    Draws multiple 2D shapes from lists of relative coordinates onto a given image.

    :param image_path: path to image
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

    # show image
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
    Resizes an image so that its larger side equals max_size while maintaining aspect ratio.
    Uses bilinear interpolation for resizing.

    :param image_path: path to image
    :param output_path: path to output image
    :param max_size: maximum size of image
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

        # Resize the image using bilinear interpolation
        img_resized = img.resize((new_width, new_height), Image.BILINEAR)

        # Save the resized image back to the same path
        img_resized.save(output_path)
