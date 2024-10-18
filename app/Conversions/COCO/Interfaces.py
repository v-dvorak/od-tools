from typing import Self

from mung.node import Node
from ultralytics.engine.results import Results

from odmetrics.bounding_box import ValBoundingBox


class ICOCOAnnotation:
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int,
                 segmentation: list[tuple[int, int]], confidence: float = 1.0):
        self.class_id = class_id
        self.bbox = None  # Python shenanigans
        self.segmentation = segmentation
        self.confidence = confidence

    def __str__(self):
        return f"({self.class_id=}, {self.bbox.left}, {self.bbox.top}, {self.bbox.width}, {self.bbox.height}, {self.bbox.segmentation})"

    def adjust_position_copy(self, left_shift: int, top_shift: int) -> Self:
        raise NotImplementedError()

    def adjust_position(self, left_shift: int, top_shift: int) -> None:
        raise NotImplementedError()

    @classmethod
    def from_mung_node(cls, clss: int, node: Node) -> Self:
        raise NotImplementedError()

    def intersects(self, other: Self) -> bool:
        raise NotImplementedError()

    def to_val_box(self, image_id: str, ground_truth: bool = False) -> ValBoundingBox:
        raise NotImplementedError()


class ICOCOFullPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            annotations: list[list[ICOCOAnnotation]],
            class_names: list[str]
    ):
        """
        Stores all subpages inside a single page (path_to_image).
        The subpages are stored in a list of lists
        where each list corresponds to single class id.

        :param image_size: path_to_image image_size, (width, height)
        :param annotations: list of COCOAnnotation
        :param class_names: list of class names
        """
        self.size = image_size
        self.class_names = class_names
        self.annotations: list[list[ICOCOAnnotation]] = annotations

    def __str__(self):
        return f"({self.class_names=}, {self.size=}, {self.annotations})"

    def all_annotations(self) -> list[ICOCOAnnotation]:
        raise NotImplementedError()

    def annotation_count(self) -> int:
        raise NotImplementedError()

    @classmethod
    def from_yolo_result(cls, result: Results) -> Self:
        raise NotImplementedError()

    def to_eval_format(self) -> list[tuple[list[int], float, int]]:
        raise NotImplementedError()


class ICOCOSplitPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            subpages: list[list[ICOCOFullPage]],
            class_names: list[str],
            splits
    ):
        self.size = image_size
        self.subpages = subpages
        self.class_names = class_names
        self.splits = splits
