import json
import math
import warnings
from importlib import resources as impresources
from typing import Self

import cv2
import numpy as np
from matplotlib import pyplot as plt

from .. import data
from ..Conversions.COCO.Interfaces import ICOCOAnnotation
from .Interfaces import IRectangle

inp_file = impresources.files(data) / "colors.json"
with inp_file.open("rt") as f:
    colors = json.load(f)["colors"]


def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class Rectangle(IRectangle):
    """
    Stores coordinates of a rectangle as absolute values.
    """

    def __init__(self, left, top, right, bottom):
        super().__init__(left, top, right, bottom)

    def coordinates(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    def intersects(self, other: Self) -> bool:
        """
        Returns true if rectangles intersect.

        :param other: other rectangle to check intersection with
        :return: True if rectangles intersect
        """
        # check if the rectangles overlap both horizontally and vertically
        return (
                self.left <= other.right
                and self.right >= other.left
                and self.top <= other.bottom
                and self.bottom >= other.top
        )

    def intersection_area(self, other: Self) -> int:
        """
        via: https://stackoverflow.com/a/27162334
        :param other: other rectangle to check intersection with
        :return: True if rectangles intersect
        """
        dx = min(self.right, other.right) - max(self.left, other.left)
        dy = min(self.bottom, other.bottom) - max(self.top, other.top)
        if dx >= 0 and dy >= 0:
            return dx * dy
        return 0

    def area(self) -> int:
        return (self.bottom - self.top) * (self.right - self.left)

    def size(self):
        """
        :return: (width, height)
        """
        return self.right - self.left, self.bottom - self.top

    def __str__(self):
        return f"Rectangle({self.left}, {self.top}, {self.right}, {self.bottom})"

    @classmethod
    def from_coco_annotation(cls, annot: ICOCOAnnotation) -> Self:
        return Rectangle(
            annot.left,
            annot.top,
            annot.left + annot.width,
            annot.top + annot.height
        )


def draw_rectangles_on_image(
        image_path: str | cv2.Mat,
        rectangles: list[Rectangle],
        thickness: int = 5,
        color: tuple[int, int, int] = None,
        shift_based_on_thickness: bool = False,
        output_path: str = None,
        loaded: bool = False
) -> None:
    """
    Draws a list of rectangles on the given image.

    :param image_path: path to image
    :param rectangles: list of Rectangle objects to display
    :param thickness: drawn rectangle thickness
    :param color: if not None, this color is applied to every rectangle,
    otherwise each rectangle is assigned a unique color
    :param shift_based_on_thickness: whether the shift outline by "thickness" numer of pixels, better visualization
    :param output_path: path to store the image at
    :param loaded: if true, passed image is already loaded as cv2 image
    """
    # Load the image using OpenCV
    if loaded:
        img = image_path
    else:
        img = cv2.imread(image_path)
    # draw rectangles
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

    # convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # show image
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis('off')  # hide axis
    plt.show()

    if output_path is not None:
        cv2.imwrite(output_path, img)


def create_split_box_matrix(
        image_size: tuple[int, int],
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
) -> list[list[Rectangle]]:
    """
    Based on the `split_section_to_starts` creates Rectangle subpages that have `window_size` dimensions
    and neighbouring rectangles overlap by minimum of `overlap_ratio`.

    :param image_size: (width, height) of image
    :param window_size: (width, height) of sliding window, output rectangles
    :param overlap_ratio:
    :return: Rectangle subpages
    """
    img_width, img_height = image_size
    win_width, win_height = window_size

    left_starts = split_section_to_starts(img_width, win_width, int(win_width * overlap_ratio), adjust_last=True)
    top_starts = split_section_to_starts(img_height, win_height, int(win_height * overlap_ratio), adjust_last=True)

    boxes_matrix = []
    for top in top_starts:
        row = []
        for left in left_starts:
            row.append(Rectangle(left, top, left + win_width, top + win_height))
        boxes_matrix.append(row)

    return boxes_matrix


def find_overlaps(box_matrix: list[list[Rectangle]]) -> list[list[Rectangle]]:
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
            l, t, r, b = cx1, cy1, cx2, cy2
            # find limitations imposed by boxes around,
            # split the space in between the limits between the two boxes
            if col - 1 >= 0:
                l = box_matrix[row][col - 1].right
            if col + 1 < width:
                r = box_matrix[row][col + 1].left
            if row - 1 >= 0:
                t = box_matrix[row - 1][col].bottom
            if row + 1 < height:
                b = box_matrix[row + 1][col].top
            new_row.append(
                Rectangle(
                    cx1 + (abs(cx1 - l) // 2),
                    cy1 + (abs(cy1 - t) // 2),
                    cx2 - (abs(cx2 - r) // 2),
                    cy2 - (abs(cy2 - b) // 2))
            )
        new_boxes.append(new_row)
    return new_boxes


def split_section_to_starts(
        total_length: int,
        section_length: int,
        min_overlap: int,
        adjust_last: bool = False,
) -> list[int]:
    """
    Generates a list of start positions so that section of given length cover the whole length
    with constant overlap that is greater or equal to min_overlap.

    :param total_length: total length to cover
    :param section_length: length of section
    :param min_overlap: minimum overlap between sections
    :param adjust_last: adjusts last box to span exactly to the end (rounding to ints may cause shifts)
    :return: list of start positions
    """
    if section_length > total_length:
        warnings.warn("Section length is bigger than the total length. Returning [0].")
        return [0]

    # maximum number of sections based on the minimal overlap
    number_of_sections = math.ceil((total_length - min_overlap) / (section_length - min_overlap))
    # calculate the step
    step = (total_length - section_length) / (number_of_sections - 1)
    # generate positions
    output = [int(i * step) for i in range(number_of_sections)]

    if adjust_last:
        output[-1] = total_length - section_length

    return output


def visualize_cutouts(
        image_path: str,
        rectangles: list[list[Rectangle]],
        masks: list[list[Rectangle]] = None,
        spacing: int = 10,
        opacity: float = 0.5,
        output_path: str = None
):
    """
    Shows image as tiles based on the given list of rectangles and visualizes it with gray spaces in between.
    Shows non-overlapping areas.

    :param image_path: path to image
    :param rectangles: list of subpages
    :param masks: non-overlapping parts of subpages
    :param spacing: gray spacing around subpages
    :param opacity: opacity of overlapping parts
    :param output_path: path to store the image at
    """
    # Load the image
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

    # loop through the tiles and add them to the grid image
    for row in range(rows):
        for col in range(columns):
            (x1, y1, x2, y2) = rectangles[row][col].coordinates()

            # take tile from original image
            tile = img[y1:y2, x1:x2]

            tile_height, tile_width = tile.shape[:2]

            # Create a gray background tile the same size as the cropped image tile
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

                # if image is corner piece, shift it accordingly
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

    # show image
    plt.figure(figsize=(10, 6))
    plt.imshow(grid_img_rgb)
    plt.axis('off')
    plt.show()

    if output_path is not None:
        cv2.imwrite(output_path, grid_img)
