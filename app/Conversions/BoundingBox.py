from typing import Self

from .IBoundingBox import IBoundingBox


class BoundingBox(IBoundingBox):
    """
    Stores coordinates of a rectangle as absolute values.
    """

    def __init__(self, left, top, right, bottom):
        super().__init__(left, top, right, bottom)

    def coordinates(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

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

    def is_fully_inside(self, other: Self) -> bool:
        fully_inside = (
                self.left >= other.left and
                self.top >= other.top and
                self.right <= other.right and
                self.bottom <= other.bottom
        )
        return fully_inside
        # this might be useful later
        # fully_outside = (
        #     rectangle1.right <= rectangle2.left or
        #     rectangle1.left >= rectangle2.right or
        #     rectangle1.bottom <= rectangle2.top or
        #     rectangle1.top >= rectangle2.bottom
        # )

    def intersection_over_union(self, other: Self) -> float:
        int_area = self.intersection_area(other)
        area1 = self.area()
        area2 = other.area()

        return int_area / (area1 + area2 - int_area)
