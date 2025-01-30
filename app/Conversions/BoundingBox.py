from typing import Self

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

    def intersects(self, other: Self) -> bool:
        # check if the annotations overlap both horizontally and vertically
        return (
                self.left <= other.right
                and self.right >= other.left
                and self.top <= other.bottom
                and self.bottom >= other.top
        )

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

    def intersection_over_union(self, other: Self) -> float:
        int_area = self.intersection_area(other)
        area1 = self.area()
        area2 = other.area()

        return int_area / (area1 + area2 - int_area)
