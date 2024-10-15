from typing import Self

from .COCO.Interfaces import ICOCOAnnotation
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

    @classmethod
    def from_coco_annotation(cls, annot: ICOCOAnnotation) -> Self:
        return BoundingBox(
            annot.left,
            annot.top,
            annot.left + annot.width,
            annot.top + annot.height
        )

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
