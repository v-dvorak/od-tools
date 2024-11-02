import json
from json import JSONEncoder
from pathlib import Path
from typing import Generator
from typing import Self

import numpy as np
from mung.io import read_nodes_from_file
from ultralytics.engine.results import Results

from .Annotation import Annotation
from .Interfaces import IAnnotation, IFullPage
from .. import ConversionUtils
from ..Formats import InputFormat, OutputFormat
from ...Splitting.SplitUtils import BoundingBox


class FullPage(IFullPage):
    def __init__(self, image_size: tuple[int, int], annotations: list[list[IAnnotation]], class_names: list[str]):
        super().__init__(image_size, annotations, class_names)

    @staticmethod
    def _sort_annotations_by_class(annotations: list[Annotation], class_count: int) -> list[list[Annotation]]:
        output = [[] for _ in range(class_count)]
        for annot in annotations:
            output[annot.class_id].append(annot)

        return output

    @classmethod
    def from_list_of_coco_annotations(cls, image_size: tuple[int, int], annotations: list[Annotation],
                                      class_names: list[str]) -> Self:
        return cls(
            image_size,
            cls._sort_annotations_by_class(annotations, len(class_names)),
            class_names
        )

    @classmethod
    def load_from_file(
            cls,
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            input_format: InputFormat
    ) -> Self:
        if input_format == InputFormat.COCO:
            return cls.from_coco_file(
                annot_path,
                class_reference_table,
                class_output_names
            )
        elif input_format == InputFormat.MUNG:
            return cls.from_mung(
                annot_path,
                image_path,
                class_reference_table,
                class_output_names
            )
        elif input_format == InputFormat.YOLO_DETECTION:
            return cls.from_yolo_detection(
                annot_path,
                image_path,
                class_reference_table,
                class_output_names
            )
        elif input_format == InputFormat.YOLO_SEGMENTATION:
            return cls.from_yolo_segmentation(
                annot_path,
                image_path,
                class_reference_table,
                class_output_names
            )
        else:
            raise ValueError()

    def save_to_file(
            self,
            output_dir: Path,
            dato_name: Path | str,
            output_format: OutputFormat,
    ):
        if output_format == OutputFormat.COCO:
            with open(output_dir / (dato_name + f".{output_format.to_annotation_extension()}"), "w") as f:
                json.dump(self, f, indent=4, cls=COCOFullPageEncoder)
        else:
            raise NotImplementedError()

    @classmethod
    def from_yolo_detection(
            cls,
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> Self:
        # TODO: manage class filtering
        image_width, image_height = ConversionUtils.get_num_pixels(image_path)
        annots = []

        with open(annot_path, "r") as file:
            for line in file:
                annots.append(cls._parse_single_line_yolo_detection(line, image_width, image_height))

        return cls.from_list_of_coco_annotations(
            (image_width, image_width),
            annots,
            class_output_names
        )

    @classmethod
    def from_yolo_segmentation(
            cls,
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> Self:
        image_width, image_height = ConversionUtils.get_num_pixels(image_path)
        annots = []

        with open(annot_path, "r") as file:
            for line in file:
                annots.append(cls._parse_single_line_yolo_segmentation(line, image_width, image_height))

        return cls.from_list_of_coco_annotations(
            (image_width, image_width),
            annots,
            class_output_names
        )

    def to_eval_format(self) -> list[tuple[list[int], float, int]]:
        output = []
        for annotation in self.all_annotations():
            output.append(annotation.to_val_box())
        return output

    def all_annotations(self) -> Generator[Annotation, None, None]:
        for row in self.annotations:
            for annotation in row:
                yield annotation

    def annotation_count(self) -> int:
        return sum([len(self.annotations[i]) for i in range(len(self.annotations))])

    @classmethod
    def from_mung_file(
            cls,
            annot_path: Path,
            image_size: tuple[int, int],
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> Self:
        nodes = read_nodes_from_file(annot_path.__str__())

        annots = []
        # process each node
        for node in nodes:
            if node.class_name in class_reference_table:
                annots.append(Annotation.from_mung_node(class_reference_table[node.class_name], node))

        # create single page
        full_page = FullPage.from_list_of_coco_annotations(image_size, annots, class_output_names)
        return full_page

    @classmethod
    def from_mung(
            cls,
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ):
        image_size = ConversionUtils.get_num_pixels(image_path)
        return FullPage.from_mung_file(
            annot_path,
            image_size,
            class_reference_table,
            class_output_names
        )

    @classmethod
    def from_coco_file(
            cls,
            file_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str]
    ) -> Self:
        with open(file_path.__str__(), "r") as file:
            data = json.load(file)
        image_width, image_height = data["width"], data["height"]
        annots = [[] for _ in range(len(class_output_names))]
        for class_name in class_reference_table.keys():
            for annot in data[class_name]:
                # process coordinates
                left = annot["left"]
                top = annot["top"]
                width = annot["width"]
                height = annot["height"]

                # process segmentation
                if annot["segmentation"] is None:
                    segm = None
                else:
                    i = 0
                    segm = []
                    while i + 1 < len(annot["segmentation"][0]):
                        segm.append((int(annot["segmentation"][0][i]), int(annot["segmentation"][0][i + 1])))
                        i += 2

                # save parsed annotation
                annots[class_reference_table[class_name]].append(
                    Annotation(class_reference_table[class_name], left, top, width, height, segm)
                )

        return cls((image_width, image_height), annots, class_output_names)

    @staticmethod
    def _parse_single_line_yolo_detection(line: str, image_width: int, image_height: int) -> Annotation:
        """
        From YOLO detection output_format to `Annotation`.

        :param line: single line of detection in YOLO output_format
        :param image_width: image width
        :param image_height: image height
        :return: Annotation
        """
        # parse data
        parts = line.strip().split()
        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert normalized coordinates to pixel values
        left = (center_x * image_width) - (width * image_width) / 2
        top = (center_y * image_height) - (height * image_height) / 2
        width_pixels = width * image_width
        height_pixels = height * image_height

        return Annotation(class_id, int(left), int(top), int(width_pixels), int(height_pixels), None)

    @staticmethod
    def _parse_single_line_yolo_segmentation(
            line: str,
            image_width: int,
            image_height: int
    ) -> Annotation:
        parts = line.strip().split()
        assert len(parts) > 2 and len(parts) % 2 == 1

        class_id = int(parts[0])

        segm = []
        i = 0
        # process every point of segmentation
        while i + 1 < len(parts[1:]):
            x, y = int(float(parts[i]) * image_width), int(float(parts[i + 1]) * image_height)
            segm.append((x, y))
            i += 2

        l, t, w, h = Annotation._bounding_box_from_segmentation(segm)

        return Annotation(class_id, l, t, w, h, segm)

    @classmethod
    def from_yolo_result(cls, result: Results) -> Self:
        class_count = len(result.names)
        predictions = [[] for _ in range(class_count)]
        class_names = [result.names[i] for i in range(class_count)]
        for i in range(len(result.boxes.xywh)):
            x_center, y_center, width, height = (int(result.boxes.xywh[i, 0]), int(result.boxes.xywh[i, 1]),
                                                 int(result.boxes.xywh[i, 2]), int(result.boxes.xywh[i, 3]))
            predictions[int(result.boxes.cls[i])].append(
                Annotation(
                    int(result.boxes.cls[i]),
                    x_center - width // 2,
                    y_center - height // 2,
                    width,
                    height,
                    # TODO: what to do with segmentation?
                    segmentation=None,
                    confidence=float(result.boxes.conf[i])
                )
            )
        return cls(result.orig_shape, predictions, class_names)

    def cut_off_predictions_too_close_to_edge(
            self, edge_offset: int = 20,
            edge_tile: tuple[bool, bool, bool, bool] = (True, True, True, True),
            verbose: bool = False
    ) -> None:
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

        if verbose:
            old_count = self.annotation_count()

        # important, should not be affected by verbose
        self.annotations = new_annotations

        if verbose:
            print(f"Cut off {old_count - self.annotation_count()} out of {old_count}")

    def adjust_position_for_all_annotations(self, shift_left: int = 0, shift_top: int = 0) -> None:
        for annotation in self.all_annotations():
            annotation.adjust_position(shift_left, shift_top)

    @staticmethod
    def resolve_overlaps_for_list_of_annotations(
            annotations: list[Annotation],
            iou_threshold: float = 0.25,
            inside_threshold: float = 0.0,
            verbose: bool = False,
    ) -> list[Annotation]:
        """
        By finding overlaps and classifying them, tries to resolve predicted bounding boxes
        for a list of COCOAnnotations.

        Detects if IoU of two bboxes is greater than given limit, eliminates duplicated bboxes.
        Detects if bbox is mostly inside other bounding box, eliminates splitting of bbox into multiple smaller ones.

        :param annotations: list of annotations to resolve
        :param iou_threshold: sets threshold for when are two bounding boxes considered to be significantly overlapping
        :param inside_threshold: sets threshold for when smaller box is considered to be inside other bounding box
        :param verbose: makes script verbose
        """
        # sort by confidence, higher confidence first
        annotations.sort(key=lambda x: x.confidence, reverse=True)

        # annotations that passed the vibe check,
        # they do not overlap with already chosen annotations etc.
        cleared_annotations = []

        for current_annot in annotations:
            intersects = False
            for selected_annot in cleared_annotations:
                if current_annot.bbox.intersects(selected_annot.bbox) and (
                        (
                                # detects if IoU of two bboxes is greater than limit
                                # eliminates duplicated bboxes
                                0 < iou_threshold < current_annot.bbox.intersection_over_union(selected_annot.bbox)
                        ) or (
                                # detects if currently investigated bbox is mostly inside other annotation
                                # eliminates splitting of bbox into multiple smaller ones
                                0 < inside_threshold <
                                current_annot.bbox.intersection_area(selected_annot.bbox) / current_annot.bbox.area()
                        )
                ):
                    intersects = True
                    if verbose:
                        print("------INTERSECTION---------")
                        print(current_annot.bbox)
                        print(selected_annot.bbox)
                        print(f"{current_annot.confidence} vs {selected_annot.confidence}")

                    break
            if not intersects:
                cleared_annotations.append(current_annot)

        return cleared_annotations

    def resolve_overlaps_with_other_page(
            self,
            other: Self,
            inside_threshold: float = 0.0,
            iou_threshold: float = 0.25,
            verbose: bool = False
    ) -> None:
        other: FullPage
        for class_id in range(len(self.annotations)):
            resolved1, resolved2 = FullPage._resolve_overlaps_smart(
                self.annotations[class_id],
                other.annotations[class_id],
                inside_threshold=inside_threshold,
                iou_threshold=iou_threshold,
                verbose=verbose,
            )
            self.annotations[class_id] = resolved1
            other.annotations[class_id] = resolved2

    @staticmethod
    def resolve_matrix_of_pages(
            subpages=list[list[Self]],
            inside_threshold: float = 0.0,
            iou_threshold: float = 0.25,
            verbose: bool = False,
    ) -> None:
        subpages: list[list[FullPage]]

        vectors = [(1, 0), (1, 1), (0, 1)]
        for row in range(len(subpages)):
            for col in range(len(subpages[0])):
                for dx, dy in vectors:
                    if row + dx < len(subpages) and col + dy < len(subpages[0]):
                        subpages[row][col].resolve_overlaps_with_other_page(
                            subpages[row + dx][col + dy],
                            inside_threshold=inside_threshold,
                            iou_threshold=iou_threshold,
                            verbose=verbose,
                        )

    @staticmethod
    def _resolve_overlaps_smart(
            first: list[Annotation],
            second: list[Annotation],

            inside_threshold: float = 0.0,
            iou_threshold: float = 0.25,
            verbose: bool = False,
    ) -> tuple[list[Annotation], list[Annotation]]:
        if len(first) == 0 or len(second) == 0:
            return first, second

        # create vector of indexes
        look_up = np.vstack((np.column_stack((np.zeros(len(first), dtype=int), np.arange(len(first)))),
                             np.column_stack((np.ones(len(second), dtype=int), np.arange(len(second))))))

        sorted_indexes = np.argsort([-annot.confidence for annot in (first + second)])

        look_up = look_up[sorted_indexes]

        cleared_annotations1 = []
        cleared_annotations2 = []

        for index in look_up:
            box_id, annot_id = index[0], index[1]

            intersects = False
            if box_id == 0:
                to_compare = cleared_annotations2
                to_save = cleared_annotations1
                current_annot = first[annot_id]
            else:
                to_compare = cleared_annotations1
                to_save = cleared_annotations2
                current_annot = second[annot_id]

            for other_annot in to_compare:
                if current_annot.bbox.intersects(other_annot.bbox) and (
                        (
                                # detects if IoU of two bboxes is greater than limit
                                # eliminates duplicated bboxes
                                0 < iou_threshold < current_annot.bbox.intersection_over_union(other_annot.bbox)
                        ) or (
                                # detects if currently investigated bbox is mostly inside other annotation
                                # eliminates splitting of bbox into multiple smaller ones
                                0 < inside_threshold <
                                current_annot.bbox.intersection_area(other_annot.bbox) / current_annot.bbox.area()
                        )
                ):
                    intersects = True
                    if verbose:
                        print("------INTERSECTION---------")
                        print(current_annot.bbox)
                        print(other_annot.bbox)
                        print(f"{current_annot.confidence} vs {other_annot.confidence}")

                    break
            if not intersects:
                to_save.append(current_annot)
                # if box_id == 0:
                #     cleared_annotations1.append(current_annot)
                # else:
                #     cleared_annotations2.append(current_annot)
        return cleared_annotations1, cleared_annotations2

    @classmethod
    def combine_multiple_pages_and_resolve(
            cls,
            subpages: list[Self],
            splits: list[list[BoundingBox]],
            edge_offset: int = 20,
            verbose: bool = False,
    ) -> Self:
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

        # resolve overlaps
        matrix = list(np.reshape(subpages, (len(splits), len(splits[0]))))
        FullPage.resolve_matrix_of_pages(matrix, inside_threshold=0, iou_threshold=0.25, verbose=verbose)

        # dump all annotations into a single matrix
        completed_annotations = [[] for _ in range(len(class_names))]
        for subpage in subpages:
            for annotation in subpage.all_annotations():
                completed_annotations[annotation.class_id].append(annotation)
        # #
        # # resolve overlaps etc.
        # completed_annotations = [FullPage.resolve_overlaps_for_list_of_annotations(class_annotations) for
        #                          class_annotations in completed_annotations]

        return FullPage((last_split.left, last_split.bottom), completed_annotations, class_names)


class COCOFullPageEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, FullPage):
            output = {
                # "source": obj.source,
                "width": obj.size[0],
                "height": obj.size[1],
            }
            for i in range(len(obj.class_names)):
                output[obj.class_names[i]] = obj.annotations[i]
            return output
        elif isinstance(obj, Annotation):
            return COCOAnnotationEncoder().default(obj)

        return super().default(obj)


class COCOAnnotationEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Annotation):
            # flatten
            segm = []
            for x, y in obj.segmentation:
                segm.append(x)
                segm.append(y)

            return {
                "left": obj.bbox.left,
                "top": obj.bbox.top,
                "width": obj.bbox.width,
                "height": obj.bbox.height,
                "segmentation": [segm],
            }
        return super().default(obj)