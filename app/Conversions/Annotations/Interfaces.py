from enum import Enum
from pathlib import Path
from typing import Generator
from typing import Self

from mung.node import Node
from ultralytics.engine.results import Results

from ..Formats import OutputFormat, InputFormat
from ..IBoundingBox import IBoundingBox


class AnnotationType(Enum):
    GROUND_TRUTH = 1
    PREDICTION = 2


class IAnnotation:
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int,
                 segmentation: list[tuple[int, int]] | None, confidence: float = 1.0,
                 an_type: AnnotationType = AnnotationType.GROUND_TRUTH):
        self.class_id = class_id
        self.bbox = None  # Python shenanigans
        self.segmentation = segmentation
        self.confidence = confidence
        self.an_type = an_type
        self.image_name: str = None

    def __str__(self):
        return f"({self.class_id=}, {self.bbox.left}, {self.bbox.top}, {self.bbox.width}, {self.bbox.height}, {self.bbox.segmentation})"

    def adjust_position_copy(self, left_shift: int, top_shift: int) -> Self:
        """
        Creates a new Annotation object with adjusted position.

        :param left_shift: pixel shift to the left
        :param top_shift: pixel shift to the top
        :return: new Annotation object with adjusted coordinates
        """
        raise NotImplementedError()

    def adjust_position(self, left_shift: int, top_shift: int) -> None:
        """
        Adjusts classes position in place.

        :param left_shift: pixel shift to the left
        :param top_shift: pixel shift to the top
        """
        raise NotImplementedError()

    @classmethod
    def from_mung_node(cls, class_id: int, node: Node) -> Self:
        """
        Creates a new Annotation object from Mung Node.

        :param class_id: class id
        :param node: Mung Node
        :return: new Annotation object
        """
        raise NotImplementedError()

    def intersects(self, other: Self) -> bool:
        """
        Returns true if two Annotation objects intersect, else false.

        :param other: other Annotation object
        """
        raise NotImplementedError()

    @staticmethod
    def bounding_box_from_segmentation(segm: list[tuple[int, int]]) -> IBoundingBox:
        """
        Returns the bounding box of the given segmentation.

        :param segm: list of segmentation coordinates
        :return: IBoundingBox
        """
        raise NotImplementedError()

    @staticmethod
    def segmentation_from_bounding_box(bbox: IBoundingBox) -> list[tuple[int, int]]:
        """
        Returns the segmentation coordinates of the given bounding box.

        :param bbox: bounding box
        :return: segmentation as list[tuple[int, int]]
        """
        raise NotImplementedError()

    def set_image_name(self, image_name: str):
        """
        Sets the image name.

        :param image_name: unique image name
        """
        raise NotImplementedError()

    def get_image_name(self) -> str:
        """
        Gets the image name.

        :return: image name
        """
        raise NotImplementedError()

    def get_class_id(self) -> int:
        """
        Gets the class id.

        :return: class id
        """
        raise NotImplementedError()


class IFullPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            annotations: list[list[IAnnotation]],
            class_names: list[str]
    ):
        """
        Stores all subpages inside a single page (path_to_image).
        The subpages are stored in a list of lists
        where each list corresponds to single class id.

        :param image_size: path_to_image image_size, (width, height)
        :param annotations: list of Annotation
        :param class_names: list of class names
        """
        self.size = image_size
        self.class_names = class_names
        self.annotations: list[list[IAnnotation]] = annotations

    def __str__(self):
        return f"({self.class_names=}, {self.size=}, {self.annotations})"

    def save_to_file(self, output_dir: Path, dato_name: Path | str, output_format: OutputFormat) -> None:
        """
        Based on OutputFormat saves FullPage to the output directory.

        :param output_dir: output directory
        :param dato_name: output file name, without extension
        :param output_format: output format
        """
        raise NotImplementedError()

    @classmethod
    def from_list_of_coco_annotations(cls, image_size: tuple[int, int], annotations: list[IAnnotation],
                                      class_names: list[str]) -> Self:
        """
        Creates a new FullPage object from a list of annotations.

        :param image_size: path_to_image image_size, (width, height)
        :param annotations: list of Annotation
        :param class_names: list of class names
        :return: new FullPage object
        """
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
        """
        Adjusts the position of all annotations by given left and top shift.

        :param shift_left: left shift of the annotations
        :param shift_top: top shift of the annotations
        """
        raise NotImplementedError()

    def all_annotations(self) -> Generator[IAnnotation, None, None]:
        """
        Creates a generator of all Annotations in FullPage.

        :return: generator of Annotations
        """
        raise NotImplementedError()

    def annotation_count(self) -> int:
        """
        Returns the total number of annotations.
        """
        raise NotImplementedError()

    @classmethod
    def from_yolo_result(cls, result: Results) -> Self:
        """
        Transforms YOLO predictions into an FullPage object.

        :param result: YOLO predictions
        :return: FullPage object
        """
        raise NotImplementedError()

    def resolve_overlaps_with_other_page(
            self,
            other: Self,
            inside_threshold: float = 0.0,
            iou_threshold: float = 0.25,
            verbose: bool = False
    ) -> None:
        """
        Removes overlapping annotations from two different pages based on IoU.
        Annotations with higher confidence are kept, resolution is done in place - original objects are changed.

        :param other: another FullPage object
        :param inside_threshold: how much object has to be inside other object to trigger resolving
        :param iou_threshold: how big IoU has to be to trigger resolving
        :param verbose: make script verbose
        """
        raise NotImplementedError()

    def resolve_overlaps_inside_self(
            self,
            inside_threshold: float = 0.0,
            iou_threshold: float = 0.25,
            verbose: bool = False
    ):
        """
        Removes overlapping annotations inside single page based on IoU.
        Annotations with higher confidence are kept, resolution is done in place - original objects are changed.

        :param inside_threshold: how much object has to be inside other object to trigger resolving
        :param iou_threshold: how big IoU has to be to trigger resolving
        :param verbose: make script verbose
        """
        raise NotImplementedError()

    @staticmethod
    def resolve_matrix_of_pages(
            subpages=list[list[Self]],
            inside_threshold: float = 0.0,
            iou_threshold: float = 0.25,
            verbose: bool = False,
    ) -> None:
        """
        Given a matrix of subpages, resolves their overlaps with smart algorithm.

        :param subpages: matrix of subpages
        :param inside_threshold: how much object has to be inside other object to trigger resolving
        :param iou_threshold: how big IoU has to be to trigger resolving
        :param verbose: make script verbose
        """
        raise NotImplementedError()

    @classmethod
    def load_from_file(
            cls,
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            input_format: InputFormat
    ) -> Self:
        """
        Loads a single page of annotations from given file in specified format.

        :param annot_path: path to file
        :param image_path: path to image
        :param class_reference_table: class reference table
        :param class_output_names: class output names
        :param input_format: input format
        :return: IFullPage
        """
        raise NotImplementedError()

    @classmethod
    def from_yolo_detection(
            cls,
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> Self:
        """
        Loads a single page of annotations from a YOLO detection annotation file.

        :param annot_path: path to file
        :param image_path: path to image
        :param class_reference_table: class reference table
        :param class_output_names: class output names
        :return: IFullPage
        """
        raise NotImplementedError()

    @staticmethod
    def from_yolo_segmentation(
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> "IFullPage":
        """
        Loads a single page of annotations from a YOLO segmentation annotation file.

        :param annot_path: path to file
        :param image_path: path to image
        :param class_reference_table: class reference table
        :param class_output_names: class output names
        :return: IFullPage
        """
        raise NotImplementedError()

    @classmethod
    def combine_multiple_pages_and_resolve(
            cls,
            subpages: list[Self],
            splits: list[list[IBoundingBox]],
            iou_threshold: float = 0.25,
            edge_offset: int = 20,
            verbose: bool = False,
    ) -> Self:
        """
        Combines multiple pages into a single page.

        :param subpages: list of subpages
        :param splits: matrix of splits
        :param iou_threshold: how big IoU has to be to trigger resolving
        :param edge_offset: offset from edges, anything outside the edge will be dropped from final page
        :param verbose: make script verbose
        :return: IFullPage
        """
        raise NotImplementedError()

    def extend_page(self, new_page: Self):
        """
        Adds annotations and class names from give page into the page.

        :param new_page: page to source new annotations from
        """
        raise NotImplementedError()


class ISplitPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            subpages: list[list[IFullPage]],
            class_names: list[str],
            splits
    ):
        self.size = image_size
        self.subpages = subpages
        self.class_names = class_names
        self.splits = splits
