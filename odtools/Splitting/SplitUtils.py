import json
from importlib import resources as impresources
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import data
from ..Conversions.BoundingBox import BoundingBox

inp_file = impresources.files(data) / "colors.json"
with inp_file.open("rt") as f:
    colors = json.load(f)["colors"]


def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def draw_rectangles_on_image(
        image: str | Path | cv2.Mat,
        rectangles: list[BoundingBox],
        thickness: int = 5,
        color: tuple[int, int, int] | list[tuple[int, int, int]] = None,
        shift_based_on_thickness: bool = False,
        output_path: str = None,
        show: bool = False
):
    """
    Draws a list of annotations on the given path_to_image.

    :param image: image or path to image
    :param rectangles: list of BoundingBox objects to display
    :param thickness: drawn rectangle thickness
    :param color: if not None, this color is applied to every rectangle
    otherwise each rectangle is assigned a unique color
    :param shift_based_on_thickness: whether the shift outline by "thickness" numer of pixels, better visualization
    :param output_path: path to store the final image at
    :param show: whether to show the final image or not
    """
    # Load the image using OpenCV
    if isinstance(image, Path) or isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    # draw annotations
    for i, rectangle in enumerate(rectangles):
        (x1, y1, x2, y2) = rectangle.coordinates()

        # set or unique color
        if color is not None:
            c_color = color
        else:
            c_color = hex_to_rgb(colors[i])

        # draw rectangle with predefined color
        if shift_based_on_thickness:
            cv2.rectangle(img, (x1 + thickness, y1 + thickness), (x2 - thickness, y2 - thickness),
                          color=c_color, thickness=thickness)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color=c_color, thickness=thickness)

    # optional show
    if show:
        # convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # display
        Image.fromarray(img_rgb).show()

    # optional save
    if output_path is not None:
        cv2.imwrite(output_path, img)

    return img


def find_overlaps(box_matrix: list[list[BoundingBox]]) -> list[list[BoundingBox]]:
    """
    Given a matrix of overlapping boxes returns subpages that do not overlap.

    :param box_matrix: overlapping Rectangles
    :return: non-overlapping Rectangles
    """
    width = len(box_matrix[0])
    height = len(box_matrix)
    new_boxes = []
    for row in range(height):
        new_row = []
        for col in range(width):
            cx1, cy1, cx2, cy2 = box_matrix[row][col].coordinates()
            # if not found, leave coordinates the same
            lim_left, lim_bottom, lim_right, lim_top = cx1, cy1, cx2, cy2
            # find limitations imposed by boxes around,
            # split the space in between the limits between the two boxes
            if col - 1 >= 0:
                lim_left = box_matrix[row][col - 1].right
            if col + 1 < width:
                lim_right = box_matrix[row][col + 1].left
            if row - 1 >= 0:
                lim_bottom = box_matrix[row - 1][col].bottom
            if row + 1 < height:
                lim_top = box_matrix[row + 1][col].top
            new_row.append(
                BoundingBox(
                    cx1 + (abs(cx1 - lim_left) // 2),
                    cy1 + (abs(cy1 - lim_bottom) // 2),
                    cx2 - (abs(cx2 - lim_right) // 2),
                    cy2 - (abs(cy2 - lim_top) // 2))
            )
        new_boxes.append(new_row)
    return new_boxes


def visualize_cutouts(
        image_path: str,
        rectangles: list[list[BoundingBox]],
        masks: list[list[BoundingBox]] = None,
        spacing: int = 10,
        opacity: float = 0.5,
        output_path: str = None
):
    """
    Shows path_to_image as tiles based on the given list of annotations and visualizes it with gray spaces in between.
    Shows non-overlapping areas.

    :param image_path: path to path_to_image
    :param rectangles: list of subpages
    :param masks: non-overlapping parts of subpages
    :param spacing: gray spacing around subpages
    :param opacity: opacity of overlapping parts
    :param output_path: path to store the path_to_image at
    """
    # Load the path_to_image
    img = cv2.imread(image_path)

    # define gray
    gray_color = (200, 200, 200)

    # calculate overall dimensions
    rows, columns = len(rectangles), len(rectangles[0])
    (x1, y1, x2, y2) = rectangles[0][0].coordinates()
    grid_width = (x2 - x1) * columns + (columns + 1) * spacing
    grid_height = (y2 - y1) * rows + (rows + 1) * spacing

    # blank canvas
    grid_img = np.full((grid_height, grid_width, 3), gray_color, dtype=np.uint8)

    # loop through the tiles and add them to the grid path_to_image
    for row in range(rows):
        for col in range(columns):
            (x1, y1, x2, y2) = rectangles[row][col].coordinates()

            # take tile from original path_to_image
            tile = img[y1:y2, x1:x2]

            tile_height, tile_width = tile.shape[:2]

            # Create a gray background tile the same size as the cropped path_to_image tile
            gray_tile = np.full((tile_height, tile_width, 3), gray_color, dtype=np.uint8)

            # Blend the tile with the gray background to achieve reduced opacity
            blended_tile = cv2.addWeighted(tile, opacity, gray_tile, 1 - opacity, 0)

            y_offset = row * (y2 - y1) + (row + 1) * spacing
            x_offset = col * (x2 - x1) + (col + 1) * spacing

            # place tile on grid
            grid_img[y_offset:y_offset + (y2 - y1), x_offset:x_offset + (x2 - x1)] = blended_tile

            if masks is not None:
                (m1, n1, m2, n2) = masks[row][col].coordinates()
                mask = img[n1:n2, m1:m2]

                # if path_to_image is corner piece, shift it accordingly
                if col == 0:
                    left_offset = 0
                elif col == columns - 1:
                    left_offset = ((x2 - x1) - (m2 - m1))
                else:
                    left_offset = ((x2 - x1) - (m2 - m1)) // 2

                if row == 0:
                    top_offset = 0
                elif row == rows - 1:
                    top_offset = ((y2 - y1) - (n2 - n1))
                else:
                    top_offset = ((y2 - y1) - (n2 - n1)) // 2

                grid_img[y_offset + top_offset:y_offset + (n2 - n1) + top_offset,
                x_offset + left_offset:x_offset + (m2 - m1) + left_offset] = mask

    # convert for display
    grid_img_rgb = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
    # display
    Image.fromarray(grid_img_rgb).show()
    # optional save
    if output_path is not None:
        cv2.imwrite(output_path, grid_img)
