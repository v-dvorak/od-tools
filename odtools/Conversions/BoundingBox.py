from typing import Self, Optional

import numpy as np

from .direction import Direction


class BoundingBox:
    """
    Stores coordinates of a rectangle as absolute values.
    """

    def __init__(self, left, top, right, bottom):
        self.segmentation = None
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.width = self.right - self.left
        self.height = self.bottom - self.top

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                    self.left == other.left
                    and self.top == other.top
                    and self.right == other.right
                    and self.bottom == other.bottom
            )
        return False
    
    def __str__(self):
        return f"Rectangle(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"

    def coordinates(self) -> tuple[int, int, int, int]:
        """
        Returns the coordinates of a rectangle as a tuple (left, top, right, bottom).
        """
        return self.left, self.top, self.right, self.bottom

    def xyxy(self) -> tuple[int, int, int, int]:
        """
        Returns the coordinates of a rectangle as a tuple (left, top, right, bottom).
        """
        return self.left, self.top, self.right, self.bottom

    def xcycwh(self) -> tuple[int, int, int, int]:
        """
        Returns the coordinates of a rectangle as a tuple (x_center, y_center, width, height).
        """
        return self.left + self.width // 2, self.top + self.height // 2, self.width, self.height

    def center(self) -> tuple[float, float]:
        """
        Returns the height and width coordinates of the center of the bounding box.
        """
        return self.top + self.height / 2, self.left + self.width / 2

    def intersects(self, other: Self, direction: Optional[Direction] = None) -> bool:
        """
        Returns true if annotations intersect.

        :param other: other rectangle to check intersection with
        :param direction: direction
        :return: True if annotations intersect
        """
        # check if the annotations overlap both horizontally and vertically
        match direction:
            case None:
                return (
                        self.left <= other.right
                        and self.right >= other.left
                        and self.top <= other.bottom
                        and self.bottom >= other.top
                )
            case Direction.VERTICAL:
                return (
                        self.top <= other.bottom
                        and self.bottom >= other.top
                )
            case Direction.HORIZONTAL:
                return (
                        self.left <= other.right
                        and self.right >= other.left
                )
            case _:
                raise TypeError(f"Invalid direction: {direction}")

    def intersection_area(self, other: Self) -> int:
        """
        via: https://stackoverflow.com/a/27162334
        :param other: other rectangle to check intersection with
        :return: True if bounding boxes intersect
        """
        dx = min(self.right, other.right) - max(self.left, other.left)
        dy = min(self.bottom, other.bottom) - max(self.top, other.top)
        if dx > 0 and dy > 0:
            return dx * dy
        return 0

    def area(self) -> int:
        return (self.bottom - self.top) * (self.right - self.left)

    def size(self):
        """
        :return: (width, height)
        """
        return self.right - self.left, self.bottom - self.top

    def shift(self, left_shift: int = 0, top_shift: int = 0) -> None:
        self.left += left_shift
        self.right += left_shift
        self.top += top_shift
        self.bottom += top_shift

    def shift_copy(self, left_shift: int = 0, top_shift: int = 0) -> "BoundingBox":
        return BoundingBox(
            self.left + left_shift,
            self.top + top_shift,
            self.right + left_shift,
            self.bottom + top_shift
        )

    @classmethod
    def from_ltwh(cls, left, top, width, height) -> Self:
        return cls(left, top, left + width, top + height)

    @classmethod
    def from_list_of_boxes(cls, bboxes: list[Self]) -> Self:
        """
        Returns a bounding box that encapsulates all the given bounding boxes.

        :param bboxes: list of bounding boxes
        """
        if len(bboxes) == 0:
            raise ValueError("Empty list")

        return cls(
            min(bboxes, key=lambda b: b.left).left,
            min(bboxes, key=lambda b: b.top).top,
            max(bboxes, key=lambda b: b.right).right,
            max(bboxes, key=lambda b: b.bottom).bottom
        )

    def is_fully_inside(self, other: Self, direction: Optional[Direction] = None) -> bool:
        """
        Returns true if THIS bounding box is fully inside the OTHER bounding box.
        If directions is specified, returns true if THIS is inside the vertical/horizontal strip defined by the OTHER.

        :param other: "bigger" rectangle
        :param direction: "vertical", "horizontal" or None (for both)
        :return: rectangle1 is fully inside rectangle2
        """
        if direction is None:
            return (
                    self.left >= other.left and
                    self.top >= other.top and
                    self.right <= other.right and
                    self.bottom <= other.bottom
            )
        elif direction == Direction.HORIZONTAL:
            return (
                    other.left <= self.left and self.right <= other.right
            )
        elif direction == Direction.VERTICAL:
            return (
                    other.top <= self.top and self.bottom <= other.bottom
            )
        else:
            raise NotImplementedError()

    def _2D_iou(self, other: Self) -> float:
        intersection_area = self.intersection_area(other)
        if intersection_area <= 0:
            return 0.0

        union = self.area() + other.area() - intersection_area
        if union <= 0:
            return 0.0

        return intersection_area / union

    def _horizontal_iou(self, other: Self) -> float:
        overlap_start = max(self.left, other.left)
        overlap_end = min(self.right, other.right)

        union_start = min(self.left, other.left)
        union_end = max(self.right, other.right)

        overlap = overlap_end - overlap_start
        union = union_end - union_start
        if overlap <= 0 or union <= 0:
            return 0

        return overlap / union

    def _vertical_iou(self, other: Self) -> float:
        overlap_start = max(self.top, other.top)
        overlap_end = min(self.bottom, other.bottom)

        union_start = min(self.top, other.top)
        union_end = max(self.bottom, other.bottom)

        overlap = overlap_end - overlap_start
        union = union_end - union_start
        if overlap <= 0 or union <= 0:
            return 0

        return overlap / union

    def intersection_over_union(self, other: Self, direction: Optional[Direction] = None) -> float:
        if direction is None:
            return self._2D_iou(other)
        elif direction == Direction.HORIZONTAL:
            return self._horizontal_iou(other)
        elif direction == Direction.VERTICAL:
            return self._vertical_iou(other)
        else:
            raise NotImplementedError()

    def center_distance(self, bbox2: Self, direction: Optional[Direction] = None) -> float:
        c_v1, c_h1 = self.center()
        c_v2, c_h2 = bbox2.center()
        if direction is None:
            return np.sqrt((c_v1 - c_v2) ** 2 + (c_h1 - c_h2) ** 2)
        elif direction == Direction.HORIZONTAL:
            return abs(c_h1 - c_h2)
        elif direction == Direction.VERTICAL:
            return abs(c_v1 - c_v2)
        else:
            raise NotImplementedError(f"Not implemented for direction {direction}")



if __name__ == "__main__":
    from unittest import TestCase, main
    
    class TestIoU(TestCase):

        def test_iou_identical_boxes(self):
            b1 = BoundingBox(0, 0, 10, 10)
            b2 = BoundingBox(0, 0, 10, 10)
            assert b1.intersection_over_union(b2) == 1.0


        def test_iou_no_overlap(self):
            b1 = BoundingBox(0, 0, 10, 10)
            b2 = BoundingBox(20, 20, 30, 30)
            assert b1.intersection_over_union(b2) == 0.0


        def test_iou_partial_overlap(self):
            b1 = BoundingBox(0, 0, 10, 10)
            b2 = BoundingBox(5, 5, 15, 15)

            # Intersection area = 5*5 = 25
            # Union = 100 + 100 - 25 = 175
            expected = 25 / 175

            self.assertAlmostEqual(b1.intersection_over_union(b2), expected)


        def test_iou_touching_edges(self):
            b1 = BoundingBox(0, 0, 10, 10)
            b2 = BoundingBox(10, 0, 20, 10)
            assert b1.intersection_over_union(b2) == 0.0


        def test_iou_one_inside_another(self):
            b1 = BoundingBox(0, 0, 10, 10)
            b2 = BoundingBox(2, 2, 8, 8)

            # Intersection = 6*6 = 36
            # Union = 100
            expected = 36 / 100

            self.assertAlmostEqual(b1.intersection_over_union(b2), expected)


        def test_iou_symmetry(self):
            b1 = BoundingBox(0, 0, 10, 10)
            b2 = BoundingBox(5, 5, 15, 15)

            self.assertAlmostEqual(
                b1.intersection_over_union(b2),
                b2.intersection_over_union(b1)
            )


        def test_iou_zero_area_box(self):
            b1 = BoundingBox(0, 0, 0, 0)
            b2 = BoundingBox(0, 0, 10, 10)

            assert b1.intersection_over_union(b2) == 0.0

    main()