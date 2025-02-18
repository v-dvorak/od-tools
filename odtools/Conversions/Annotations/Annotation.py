from typing import Self

from mung.io import Node

from .Interfaces import IAnnotation, AnnotationType
from .. import ConversionUtils
from ..BoundingBox import BoundingBox


class Annotation(IAnnotation):
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int,
                 segmentation: list[tuple[int, int]] | None, confidence: float = 1.0,
                 an_type: AnnotationType = AnnotationType.GROUND_TRUTH):
        super().__init__(class_id, left, top, width, height, segmentation, confidence=confidence, an_type=an_type)
        self.bbox = BoundingBox.from_ltwh(left, top, width, height)  # Python shenanigans
        if segmentation is None:
            self.segmentation = self.segmentation_from_bounding_box(self.bbox)

    def set_image_name(self, image_name: str):
        self.image_name = image_name

    def get_image_name(self) -> str:
        return self.image_name

    def get_class_id(self) -> int:
        return self.class_id

    @classmethod
    def from_mung_node(cls, class_id: int, node: Node, an_type: AnnotationType = AnnotationType.GROUND_TRUTH) -> Self:
        return cls(
            class_id,
            node.left, node.top, node.width, node.height,
            ConversionUtils.mung_segmentation_to_absolute_coordinates(node),
            an_type=an_type
        )

    @staticmethod
    def bounding_box_from_segmentation(
            segm: list[tuple[int, int]]
    ):
        left = min(segm, key=lambda x: x[0])[0]
        top = min(segm, key=lambda x: x[1])[1]
        right = max(segm, key=lambda x: x[0])[0]
        bottom = max(segm, key=lambda x: x[1])[1]
        return left, top, right, bottom

    @staticmethod
    def segmentation_from_bounding_box(
            bbox: BoundingBox
    ):
        x1, y1, x2, y2 = bbox.xyxy()
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def intersects(self, other: Self) -> bool:
        other: Annotation
        return self.bbox.intersects(other.bbox)

    def adjust_position(self, left_shift: int = 0, top_shift: int = 0) -> None:
        self.bbox.shift(left_shift, top_shift)

    def adjust_position_copy(self, left_shift: int, top_shift: int) -> Self:
        if self.segmentation is not None:
            new_segmentation = [(x + left_shift, y + top_shift) for x, y in self.segmentation]
        else:
            new_segmentation = None

        return Annotation(
            self.class_id,

            self.bbox.left + left_shift,
            self.bbox.top + top_shift,
            self.bbox.width,
            self.bbox.height,

            new_segmentation,
            confidence=self.confidence,
            an_type=self.an_type
        )
