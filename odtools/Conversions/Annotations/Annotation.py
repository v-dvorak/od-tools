from typing import Self, Optional, Any
import numpy as np

from mung.io import Node

from .annotation_type import AnnotationType
from .. import ConversionUtils
from ..BoundingBox import BoundingBox, Direction



class Annotation:
    def __init__(
        self,
        class_id: int,
        left: int,
        top: int,
        width: int,
        height: int,
        mask: np.ndarray | None,
        confidence: float = 1.0,
        an_type: AnnotationType = AnnotationType.GROUND_TRUTH,
    ):
        self.class_id = class_id
        self.bbox: BoundingBox = BoundingBox.from_ltwh(
            left=left, top=top, width=width, height=height
        )
        self.mask = mask
        self.confidence = confidence
        self.an_type = an_type
        self.image_name: str = None # type: ignore


    def __str__(self):
        return f"({self.class_id=}, {self.bbox.left}, {self.bbox.top}, {self.bbox.width}, {self.bbox.height}, {self.bbox.segmentation})"

    @classmethod
    def from_bbox(
        cls,
        class_id: int,
        bbox: BoundingBox,
        segmentation: np.ndarray | None = None,
        confidence: float = 1.0,
        an_type: AnnotationType = AnnotationType.GROUND_TRUTH,
    ) -> Self:
        return cls(
            class_id,
            bbox.left,
            bbox.top,
            bbox.width,
            bbox.height,
            mask=segmentation,
            confidence=confidence,
            an_type=an_type,
        )

    def set_image_name(self, image_name: str):
        self.image_name = image_name

    def get_image_name(self) -> str:
        return self.image_name

    def get_class_id(self) -> int:
        return self.class_id

    @classmethod
    def from_mung_node(
        cls,
        class_id: int,
        node: Node,
        an_type: AnnotationType = AnnotationType.GROUND_TRUTH,
    ) -> Self:
        """
        Creates a new Annotation object from Mung Node.

        :param class_id: class id
        :param node: Mung Node
        :return: new Annotation object
        """
        return cls(
            class_id,
            node.left,
            node.top,
            node.width,
            node.height,
            mask=node.mask,
            an_type=an_type,
        )

    @staticmethod
    def bounding_box_from_segmentation(segm: list[tuple[int, int]]) -> tuple[int, int ,int, int]:
        """
        Returns the bounding box of the given segmentation.

        :param segm: list of segmentation coordinates
        :return: left, top, width, height
        """
        left = min(segm, key=lambda x: x[0])[0]
        top = min(segm, key=lambda x: x[1])[1]
        right = max(segm, key=lambda x: x[0])[0]
        bottom = max(segm, key=lambda x: x[1])[1]
        return left, top, right, bottom

    @staticmethod
    def segmentation_from_bounding_box(bbox: BoundingBox):
        """
        Returns the segmentation coordinates of the given bounding box.

        :param bbox: bounding box
        :return: segmentation as list[tuple[int, int]]
        """
        x1, y1, x2, y2 = bbox.xyxy()
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def intersects(self, other: Self, direction: Optional[Direction] = None) -> bool:
        """
        Returns true if two Annotation objects intersect, else false.

        :param other: other Annotation object
        """
        return self.bbox.intersects(other.bbox, direction=direction)

    def intersection_over_union(
        self, other: Self, direction: Optional[Direction] = None
    ) -> float:
        return self.bbox.intersection_over_union(other.bbox, direction=direction)

    def adjust_position(self, left_shift: int = 0, top_shift: int = 0) -> None:
        """
        Adjusts classes position in place.

        :param left_shift: pixel shift to the left
        :param top_shift: pixel shift to the top
        """
        self.bbox.shift(left_shift, top_shift)
    
    def adjust_position_copy(self, left_shift: int, top_shift: int) -> "Annotation":
        """
        Creates a new Annotation object with adjusted position.

        :param left_shift: pixel shift to the left
        :param top_shift: pixel shift to the top
        :return: new Annotation object with adjusted coordinates
        """
        # if self.mask is not None:
        #     new_segmentation = [
        #         (x + left_shift, y + top_shift) for x, y in self.mask
        #     ]
        # else:
        #     new_segmentation = None

        return Annotation(
            self.class_id,
            self.bbox.left + left_shift,
            self.bbox.top + top_shift,
            self.bbox.width,
            self.bbox.height,
            self.mask,
            confidence=self.confidence,
            an_type=self.an_type,
        )

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Annotation) and id(self) == id(other)
