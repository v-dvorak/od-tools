from odtools import Annotation, BoundingBox


class VerticalLine(BoundingBox):
    def __init__(self, left: int, top: int, bottom: int):
        super().__init__(left, top, left, bottom)

    @classmethod
    def from_bbox(cls, bbox: BoundingBox):
        return VerticalLine(
            (bbox.left + bbox.right) // 2,
            bbox.top,
            bbox.bottom
        )

    @classmethod
    def from_annotation(cls, annotation: Annotation):
        return VerticalLine.from_bbox(annotation.bbox)
