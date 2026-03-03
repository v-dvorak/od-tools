import json
from json import JSONEncoder
from pathlib import Path
from typing import Generator
from typing import Self

import numpy as np
from mung.io import read_nodes_from_file
from ultralytics.engine.results import Results

from .Annotation import Annotation
from .Interfaces import IAnnotation, IFullPage, AnnotationType
from .. import ConversionUtils
from ..Formats import InputFormat, OutputFormat
from ...Conversions.BoundingBox import BoundingBox



class FullPage(IFullPage):
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
        return cls(
            image_size,
            cls._sort_annotations_by_class(annotations, len(class_names)),
            class_names
        )

    def all_annotations(self) -> Generator[Annotation, None, None]:
        for row in self.annotations:
            for annotation in row:
                yield annotation

    def annotation_count(self) -> int:
        return sum([len(self.annotations[i]) for i in range(len(self.annotations))])

    def adjust_position_for_all_annotations(self, shift_left: int = 0, shift_top: int = 0) -> None:
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
    ) -> Self:
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
                from mung.graph import Node
                from mung.io import write_nodes_to_file
                from itertools import chain
                nodes = []
                id_ = 0
                for annot in chain.from_iterable(self.annotations):
                    nodes.append(Node(
                        id_,
                        self.class_names[annot.class_id],
                        top=annot.bbox.top,
                        left=annot.bbox.left,
                        width=annot.bbox.width,
                        height=annot.bbox.height,
                        mask=np.ones((annot.bbox.height, annot.bbox.width)),
                        data={"confidence": annot.confidence}
                    ))
                    id_ += 1
                write_nodes_to_file(nodes, str(output_dir / f"{dato_name}.{output_format.to_annotation_extension()}"))
            case _:
                raise NotImplementedError()

    @classmethod
    def from_yolo_result(cls, result: Results, wanted_ids: list[int] = None) -> Self:
        if wanted_ids is None:
            class_count = len(result.names)
            class_names = [result.names[i] for i in range(class_count)]
        else:
            class_count = len(wanted_ids)
            class_names = [result.names[i] for i in wanted_ids]
            # map wanted IDs to indexes in output list
            id_mapping = {w_id: index for index, w_id in enumerate(wanted_ids)}

        predictions = [[] for _ in range(class_count)]

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
                    output_index = id_mapping[class_id]

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
        if self.size[0] != new_page.size[0] or self.size[1] != new_page.size[1]:
            raise ValueError(f"Image sizes do not match: {self.size} != {new_page.size}")

        self.annotations += new_page.annotations
        self.class_names += new_page.class_names


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


# region Helpers

class _COCOHelper:
    @staticmethod
    def from_coco_file(
            file_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> FullPage:
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
                    Annotation(class_reference_table[class_name], left, top, width, height, segm, an_type=an_type)
                )

        return FullPage((image_width, image_height), annots, class_output_names)

    @staticmethod
    def save_annotation(
            page: FullPage,
            output_path: Path
    ) -> None:
        with open(output_path, "w") as f:
            json.dump(page, f, indent=4, cls=COCOFullPageEncoder)


class _MuNGHelper:
    @staticmethod
    def from_mung(
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> FullPage:
        image_size = ConversionUtils.get_num_pixels(image_path)
        return _MuNGHelper.from_mung_file(
            annot_path,
            image_size,
            class_reference_table,
            class_output_names,
            an_type=an_type
        )

    @staticmethod
    def from_mung_file(
            annot_path: Path,
            image_size: tuple[int, int],
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> FullPage:
        nodes = read_nodes_from_file(annot_path.__str__())

        annots = []
        # process each node
        for node in nodes:
            if node.class_name in class_reference_table:
                annots.append(Annotation.from_mung_node(class_reference_table[node.class_name], node, an_type=an_type))

        # create single page
        full_page = FullPage.from_list_of_coco_annotations(image_size, annots, class_output_names)
        return full_page


class _YOLOHelper:
    @staticmethod
    def from_yolo_detection(
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> FullPage:
        # TODO: manage class filtering
        image_width, image_height = ConversionUtils.get_num_pixels(image_path)
        annots = []

        with open(annot_path, "r") as file:
            for line in file:
                annots.append(_YOLOHelper._parse_single_line_yolo_detection(
                    line,
                    image_width,
                    image_height,
                    an_type=an_type
                ))

        return FullPage.from_list_of_coco_annotations(
            (image_width, image_width),
            annots,
            class_output_names
        )

    @staticmethod
    def from_yolo_segmentation(
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> FullPage:
        image_width, image_height = ConversionUtils.get_num_pixels(image_path)
        annots = []

        with open(annot_path, "r") as file:
            for line in file:
                annots.append(_YOLOHelper._parse_single_line_yolo_segmentation(
                    line,
                    image_width,
                    image_height,
                    an_type=an_type
                ))

        return FullPage.from_list_of_coco_annotations(
            (image_width, image_width),
            annots,
            class_output_names
        )

    @staticmethod
    def _parse_single_line_yolo_detection(
            line: str,
            image_width: int,
            image_height: int,
            an_type: AnnotationType.GROUND_TRUTH
    ) -> Annotation:
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

        return Annotation(class_id, int(left), int(top), int(width_pixels), int(height_pixels), None, an_type=an_type)

    @staticmethod
    def _parse_single_line_yolo_segmentation(
            line: str,
            image_width: int,
            image_height: int,
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
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

        l, t, w, h = Annotation.bounding_box_from_segmentation(segm)

        return Annotation(class_id, l, t, w, h, segm, an_type=an_type)

    @staticmethod
    def save_yolo_detection(
            page: FullPage,
            output_path: Path,
    ) -> None:
        with open(output_path, "w") as file:
            for annotation in page.all_annotations():
                file.write(_YOLOHelper._serialize_detection(page.size, annotation))
                file.write("\n")

    @staticmethod
    def _serialize_detection(image_size: tuple[int, int], annotation: Annotation) -> str:
        """
        Return normalized YOLO detection format: `class_id x_center y_center width height`.

        :param image_size: image size (width, height)
        :param annotation: annotation
        :return: serialized annotation in YOLO format
        """
        im_width, im_height = image_size
        xc, yc, w, h = annotation.bbox.xcycwh()
        return f"{annotation.class_id} {xc / im_width:.6f} {yc / im_height:.6f} {w / im_width:.6f} {h / im_height:.6f}"

# endregion
