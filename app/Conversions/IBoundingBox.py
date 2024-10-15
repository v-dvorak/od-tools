from typing import Self

from ..Conversions.COCO.Interfaces import ICOCOAnnotation


class IBoundingBox:
    """
    Stores coordinates of a rectangle as absolute values.
    """

    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def coordinates(self) -> tuple[int, int, int, int]:
        """
        Returns the coordinates of a rectangle as a tuple (left, top, right, bottom).
        """
        raise NotImplementedError()

    def intersects(self, other: Self) -> bool:
        """
        Returns true if annotations intersect.

        :param other: other rectangle to check intersection with
        :return: True if annotations intersect
        """
        raise NotImplementedError()

    def is_fully_inside(self, other: Self) -> bool:
        """
        Returns true if THIS bounding box is fully inside other bounding box.
        :param other: "bigger" rectangle
        :return: rectangle1 is fully inside rectangle2
        """
        raise NotImplementedError()

    def intersection_area(self, other: Self) -> int:
        """
        via: https://stackoverflow.com/a/27162334
        :param other: other rectangle to check intersection with
        :return: True if bounding boxes intersect
        """
        raise NotImplementedError()

    def area(self) -> int:
        raise NotImplementedError()

    def size(self):
        """
        :return: (width, height)
        """
        raise NotImplementedError()

    def __str__(self):
        return f"Rectangle({self.left}, {self.top}, {self.right}, {self.bottom})"

    @classmethod
    def from_coco_annotation(cls, annot: ICOCOAnnotation) -> Self:
        raise NotImplementedError()
