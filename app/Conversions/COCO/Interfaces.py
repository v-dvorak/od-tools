from pathlib import Path
from typing import Self

from mung.node import Node
from ultralytics.engine.results import Results

from ..Formats import OutputFormat


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

    def save_to_file(self, output_dir: Path, dato_name: Path | str, output_format: OutputFormat):
        raise NotImplementedError()

    @classmethod
    def from_list_of_coco_annotations(cls, image_size: tuple[int, int], annotations: list[ICOCOAnnotation],
                                      class_names: list[str]) -> Self:
        raise NotImplementedError()

    def cut_off_predictions_too_close_to_edge(
            self,
            edge_offset: int = 20,
            edge_tile: tuple[bool, bool, bool, bool] = (True, True, True, True),
            verbose: bool = False
    ) -> None:
        """
        Removes page's annotations that are to close to the edge.

        :param edge_offset: offset of the edge in pixels
        :param edge_tile: boolean indicating if the edge should be removed, (left, top, right, bottom) edges
        :param verbose: boolean indicating if the edge should be removed
        """
        raise NotImplementedError()

    def adjust_position_for_all_annotations(self, shift_left: int = 0, shift_top: int = 0) -> None:
        raise NotImplementedError()

    @staticmethod
    def resolve_overlaps_for_list_of_annotations(
            annotations: list[ICOCOAnnotation],
            iou_threshold: float = 0.25,
            inside_threshold: float = 0.0,
            verbose: bool = False,
    ) -> list[ICOCOAnnotation]:
        raise NotImplementedError()

    def all_annotations(self) -> list[ICOCOAnnotation]:
        raise NotImplementedError()

    def annotation_count(self) -> int:
        raise NotImplementedError()

    @classmethod
    def from_yolo_result(cls, result: Results) -> Self:
        raise NotImplementedError()

    def to_eval_format(self) -> list[tuple[list[int], float, int]]:
        raise NotImplementedError()

    def resolve_overlaps_with_other_page(
            self,
            other: Self,
            inside_threshold: float = 0.0,
            iou_threshold: float = 0.25,
            verbose: bool = False
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def resolve_matrix_of_pages(
            subpages=list[list[Self]],
            inside_threshold: float = 0.0,
            iou_threshold: float = 0.25,
            verbose: bool = False,
    ) -> None:
        raise NotImplementedError()

    @classmethod
    def from_mung_file(
            cls,
            annot_path: Path,
            image_size: tuple[int, int],
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> Self:
        """
        Loads a page of annotations from a MuNG annotation file.

        :param annot_path: path to mung file
        :param image_size: image size (width, height)
        :param class_reference_table: class reference table
        :param class_output_names: class output names

        :return: ICOCOFullPage
        """
        raise NotImplementedError()

    @classmethod
    def from_mung(
            cls,
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> Self:
        """
        Loads a page of annotations from an image and a MuNG annotation file.

        :param annot_path: path to mung file
        :param image_path: path to image
        :param class_reference_table: class reference table
        :param class_output_names: class output names

        :return: ICOCOFullPage
        """
        raise NotImplementedError()

    @classmethod
    def from_coco_file(
            cls,
            file_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> Self:
        """
        Loads a page of annotations from a COCO annotation file.

        :param file_path: path to COCO annotation file
        :param class_reference_table: class reference table
        :param class_output_names: class output names
        :return: ICOCOFullPage
        """


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
