from abc import ABC, abstractmethod
from enum import Enum
from typing import Self


class Direction(Enum):
    VERTICAL = 0
    HORIZONTAL = 1


class IBoundingBox(ABC):
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

    @abstractmethod
    def coordinates(self) -> tuple[int, int, int, int]:
        """
        Returns the coordinates of a rectangle as a tuple (left, top, right, bottom).
        """
        pass

    @abstractmethod
    def xyxy(self) -> tuple[int, int, int, int]:
        """
        Returns the coordinates of a rectangle as a tuple (left, top, right, bottom).
        """
        pass

    @abstractmethod
    def xcycwh(self) -> tuple[int, int, int, int]:
        """
        Returns the coordinates of a rectangle as a tuple (x_center, y_center, width, height).
        """
        pass

    @abstractmethod
    def center(self) -> tuple[float, float]:
        """
        Returns the height and width coordinates of the center of the bounding box.
        """
        pass

    @abstractmethod
    def intersects(self, other: Self) -> bool:
        """
        Returns true if annotations intersect.

        :param other: other rectangle to check intersection with
        :return: True if annotations intersect
        """
        pass

    @abstractmethod
    def is_fully_inside(self, other: Self, direction: Direction = None) -> bool:
        """
        Returns true if THIS bounding box is fully inside the OTHER bounding box.
        If directions is specified, returns true if THIS is inside the vertical/horizontal strip defined by the OHTER.

        :param other: "bigger" rectangle
        :param direction: "vertical", "horizontal" or None (for both)
        :return: rectangle1 is fully inside rectangle2
        """
        pass

    @abstractmethod
    def intersection_area(self, other: Self) -> int:
        """
        via: https://stackoverflow.com/a/27162334
        :param other: other rectangle to check intersection with
        :return: True if bounding boxes intersect
        """
        pass

    @abstractmethod
    def area(self) -> int:
        pass

    @abstractmethod
    def size(self):
        """
        :return: (width, height)
        """
        pass

    def __str__(self):
        return f"Rectangle({self.left}, {self.top}, {self.right}, {self.bottom})"

    @classmethod
    @abstractmethod
    def from_ltwh(cls, left, top, width, height) -> Self:
        pass

    @abstractmethod
    def shift(self, left_shift: int = 0, top_shift: int = 0) -> None:
        pass

    @abstractmethod
    def shift_copy(self, left_shift: int = 0, top_shift: int = 0) -> Self:
        pass

    @abstractmethod
    def intersection_over_union(self, other: Self, direction: Direction = None) -> float:
        pass

    @abstractmethod
    def center_distance(self, bbox2: Self, direction: Direction = None) -> float:
        pass
