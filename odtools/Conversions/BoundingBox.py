from typing import Self

import numpy as np

from .IBoundingBox import IBoundingBox, Direction


class BoundingBox(IBoundingBox):
    """
    Stores coordinates of a rectangle as absolute values.
    """

    def __init__(self, left, top, right, bottom):
        super().__init__(left, top, right, bottom)

    def coordinates(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    def xyxy(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    def xcycwh(self) -> tuple[int, int, int, int]:
        return self.left + self.width // 2, self.top + self.height // 2, self.width, self.height

    def center(self) -> tuple[float, float]:
        return self.top + self.height / 2, self.left + self.width / 2

    def intersects(self, other: Self, direction: Direction = None) -> bool:
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

    def shift(self, left_shift: int = 0, top_shift: int = 0) -> None:
        self.left += left_shift
        self.right += left_shift
        self.top += top_shift
        self.bottom += top_shift

    def shift_copy(self, left_shift: int = 0, top_shift: int = 0) -> Self:
        return BoundingBox(self.left + left_shift, self.top + top_shift, self.width, self.height)

    @classmethod
    def from_ltwh(cls, left, top, width, height) -> Self:
        return BoundingBox(left, top, left + width, top + height)

    def is_fully_inside(self, other: Self, direction: Direction = None) -> bool:
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
        if intersection_area == 0:
            return 0

        area1 = self.area()
        area2 = other.area()

        return intersection_area / (area1 + area2 - intersection_area)

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

    def intersection_over_union(self, other: Self, direction: Direction = None) -> float:
        if direction is None:
            return self._2D_iou(other)
        elif direction == Direction.HORIZONTAL:
            return self._horizontal_iou(other)
        elif direction == Direction.VERTICAL:
            return self._vertical_iou(other)
        else:
            raise NotImplementedError()

    def center_distance(self, bbox2: Self, direction: Direction = None) -> float:
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
