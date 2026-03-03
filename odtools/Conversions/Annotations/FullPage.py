from pathlib import Path
from typing import Generator, Self, Optional
from ultralytics.engine.results import Results

from .Annotation import Annotation
from .annotation_type import AnnotationType
from ..Formats import InputFormat, OutputFormat
from ...Conversions.BoundingBox import BoundingBox
from .format_helpers import _COCOHelper, _MuNGHelper, _YOLOHelper


class FullPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            annotations: list[list[Annotation]],
            class_names: list[str]
    ):
        """
        Stores all subpages inside a single page (path_to_image).
        The subpages are stored in a list of lists
        where each list corresponds to single class id.

        :param image_size: image size, (width, height)
        :param annotations: list of Annotation
        :param class_names: list of class names
        """
        self.size = image_size
        self.class_names = class_names
        self.annotations: list[list[Annotation]] = annotations

    def __str__(self):
        return f"({self.class_names=}, {self.size=}, {self.annotations})"

    @staticmethod
    def _sort_annotations_by_class(annotations: list[Annotation], class_count: int) -> list[list[Annotation]]:
        output = [[] for _ in range(class_count)]
        for annot in annotations:
            output[annot.class_id].append(annot)

        return output

    @classmethod
    def from_list_of_coco_annotations(
            cls,
            image_size: tuple[int, int],
            annotations: list[Annotation],
            class_names: list[str]
    ) -> Self:
        """
        Creates a new FullPage object from a list of annotations.

        :param image_size: path_to_image image_size, (width, height)
        :param annotations: list of Annotation
        :param class_names: list of class names
        :return: new FullPage object
        """
        return cls(
            image_size,
            cls._sort_annotations_by_class(annotations, len(class_names)),
            class_names
        )

    def all_annotations(self) -> Generator[Annotation, None, None]:
        """
        Creates a generator of all Annotations in FullPage.

        :return: generator of Annotations
        """
        for row in self.annotations:
            for annotation in row:
                yield annotation

    def annotation_count(self) -> int:
        return sum([len(self.annotations[i]) for i in range(len(self.annotations))])

    def adjust_position_for_all_annotations(self, shift_left: int = 0, shift_top: int = 0) -> None:
        """
        Adjusts the position of all annotations by given left and top shift.

        :param shift_left: left shift of the annotations
        :param shift_top: top shift of the annotations
        """
        for annotation in self.all_annotations():
            annotation.adjust_position(shift_left, shift_top)

    @classmethod
    def load_from_file(
            cls,
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            input_format: InputFormat,
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> "FullPage":
        """
        Loads a single page of annotations from given file in specified format.

        :param annot_path: path to file
        :param image_path: path to image
        :param class_reference_table: class reference table
        :param class_output_names: class output names
        :param input_format: input format
        :return: FullPage
        """
        match input_format:
            case InputFormat.COCO:
                return _COCOHelper.from_coco_file(
                    annot_path,
                    class_reference_table,
                    class_output_names,
                    an_type=an_type
                )
            case InputFormat.MUNG:
                return _MuNGHelper.from_mung(
                    annot_path,
                    image_path,
                    class_reference_table,
                    class_output_names,
                    an_type=an_type
                )
            case InputFormat.YOLO_DETECTION:
                return _YOLOHelper.from_yolo_detection(
                    annot_path,
                    image_path,
                    class_reference_table,
                    class_output_names,
                    an_type=an_type
                )
            case InputFormat.YOLO_SEGMENTATION:
                return _YOLOHelper.from_yolo_segmentation(
                    annot_path,
                    image_path,
                    class_reference_table,
                    class_output_names,
                    an_type=an_type
                )
            case _:
                raise ValueError(f"Unsupported input format: {input_format}")

    def save_to_file(
            self,
            output_dir: Path,
            dato_name: Path | str,
            output_format: OutputFormat,
    ) -> None:
        """
        Based on OutputFormat saves FullPage to the output directory.

        :param output_dir: output directory
        :param dato_name: output file name, without extension
        :param output_format: output format
        """
        match output_format:
            case OutputFormat.COCO:
                _COCOHelper.save_annotation(
                    self,
                    output_dir / f"{dato_name}.{output_format.to_annotation_extension()}",
                )
            case OutputFormat.YOLO_DETECTION:
                _YOLOHelper.save_yolo_detection(
                    self,
                    output_dir / f"{dato_name}.{output_format.to_annotation_extension()}",
                )
            case OutputFormat.MUNG:
                _MuNGHelper.save_annotation(
                    self,
                    output_dir / f"{dato_name}.{output_format.to_annotation_extension()}"
                )
            case _:
                raise NotImplementedError()

    @classmethod
    def from_yolo_result(cls, result: Results, wanted_ids: Optional[list[int]] = None) -> Self:
        """
        Transforms YOLO predictions into an FullPage object.

        :param result: YOLO predictions
        :param wanted_ids: list of class IDs that will be retrieved, if None all are retrieved
        :return: FullPage object
        """
        if wanted_ids is None:
            class_count = len(result.names)
            class_names = [result.names[i] for i in range(class_count)]
        else:
            class_count = len(wanted_ids)
            class_names = [result.names[i] for i in wanted_ids]
            # map wanted IDs to indexes in output list
            id_mapping = {w_id: index for index, w_id in enumerate(wanted_ids)}

        predictions = [[] for _ in range(class_count)]
        
        assert result.boxes is not None

        for i in range(len(result.boxes.xywh)):
            class_id = int(result.boxes.cls[i])
            # check if this particular prediction should be further processed based on its ID
            if wanted_ids is None or class_id in wanted_ids:
                x_center, y_center, width, height = (
                    float(result.boxes.xywh[i, 0]),
                    float(result.boxes.xywh[i, 1]),
                    float(result.boxes.xywh[i, 2]),
                    float(result.boxes.xywh[i, 3])
                )
                # get index in output list
                if wanted_ids is None:
                    output_index = class_id
                else:
                    output_index = id_mapping[class_id] # type: ignore

                predictions[output_index].append(
                    Annotation(
                        int(result.boxes.cls[i]),
                        round(x_center - width / 2),
                        round(y_center - height / 2),
                        round(width),
                        round(height),
                        # TODO: what to do with segmentation?
                        segmentation=None,
                        confidence=float(result.boxes.conf[i]),
                        an_type=AnnotationType.PREDICTION
                    )
                )
        # original shape is stored as (height, width) in YOLO
        return cls((result.orig_shape[1], result.orig_shape[0]), predictions, class_names)

    # region Resolve overlaps
    def cut_off_predictions_too_close_to_edge(
            self, edge_offset: int = 20,
            edge_tile: tuple[bool, bool, bool, bool] = (True, True, True, True),
            verbose: bool = False
    ) -> None:
        """
        Removes page's annotations that are to close to the edge.

        :param edge_offset: offset of the edge in pixels
        :param edge_tile: boolean indicating if the edge should be removed, (left, top, right, bottom) edges
        :param verbose: boolean indicating if the edge should be removed
        """
        width, height = self.size

        border = BoundingBox(
            0 + edge_offset if edge_tile[0] else 0,
            0 + edge_offset if edge_tile[1] else 0,
            width - edge_offset if edge_tile[2] else width,
            height - edge_offset if edge_tile[3] else height
        )

        new_annotations = []
        for class_annotations in self.annotations:
            new_c_a = []
            for annot in class_annotations:
                current_rectangle = annot.bbox
                if current_rectangle.is_fully_inside(border):
                    new_c_a.append(annot)
                # else: cut it
            new_annotations.append(new_c_a)

        old_count = self.annotation_count()

        self.annotations = new_annotations

        if verbose:
            print(f"Cut off {old_count - self.annotation_count()} out of {old_count}")


    @classmethod
    def combine_multiple_pages_and_resolve(
            cls,
            subpages: list[Self],
            splits: list[list[BoundingBox]],
            edge_offset: int = 20,
            iou_threshold: float = 0.25,
            verbose: bool = False,
    ) -> "FullPage":
        """
        Combines multiple pages into a single page.

        :param subpages: list of subpages
        :param splits: matrix of splits
        :param iou_threshold: how big IoU has to be to trigger resolving
        :param edge_offset: offset from edges, anything outside the edge will be dropped from final page
        :param verbose: make script verbose
        :return: FullPage
        """

        for i, (subpage, split) in enumerate(zip(subpages, [x for xs in splits for x in xs])):
            subpage: FullPage
            split: BoundingBox

            # cut predictions on edges
            if edge_offset > 0:
                x, y = i % len(splits[0]), i // len(splits[0])
                subpage.cut_off_predictions_too_close_to_edge(
                    edge_offset=edge_offset,
                    edge_tile=(
                        x != 0,
                        y != 0,
                        x != len(splits[0]) - 1,
                        y != len(splits) - 1,
                    ),
                    verbose=verbose
                )
            # shift annotations based in their absolute position in image
            subpage.adjust_position_for_all_annotations(split.left, split.top)

        # retrieve important values without actually passing them as arguments
        # class names from first class
        class_names = subpages[0].class_names
        # the last split is also de facto the bottom right corner of the image,
        # we can retrieve image image_size from here,
        last_split: BoundingBox = splits[-1][-1]

        def _apply_nms_single_class(annotations: list[Annotation], iou_threshold: float) -> list[Annotation]:
            annotations = sorted(annotations, key=lambda a: a.confidence, reverse=True)
            kept = []

            while annotations:
                current = annotations.pop(0)
                kept.append(current)

                annotations = [
                    a for a in annotations
                    if current.intersection_over_union(a) < iou_threshold
                ]

            return kept

        completed_annotations = [[] for _ in range(len(class_names))]
        for subpage in subpages:
            for annotation in subpage.all_annotations():
                completed_annotations[annotation.class_id].append(annotation)
        
        complete_page = FullPage((last_split.right, last_split.bottom), completed_annotations, class_names)

        complete_page.annotations = [
            _apply_nms_single_class(annots, iou_threshold)
            for annots in complete_page.annotations
        ]

        return complete_page

    # endregion
    def extend_page(self, new_page: Self):
        """
        Adds annotations and class names from give page into the page.

        :param new_page: page to source new annotations from
        """
        if self.size[0] != new_page.size[0] or self.size[1] != new_page.size[1]:
            raise ValueError(f"Image sizes do not match: {self.size} != {new_page.size}")

        self.annotations += new_page.annotations
        self.class_names += new_page.class_names

