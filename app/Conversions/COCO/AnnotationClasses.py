from json import JSONEncoder
from typing import Generator
from typing import Self

from mung.node import Node
from ultralytics.engine.results import Results

from odmetrics.bounding_box import ValBoundingBox
from odmetrics.utils.enumerators import (BBFormat, BBType, CoordinatesType)
from .Interfaces import ICOCOAnnotation, ICOCOFullPage, ICOCOSplitPage
from .. import ConversionUtils
from ...Splitting.SplitUtils import BoundingBox


class COCOAnnotation(ICOCOAnnotation):
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int,
                 segmentation: list[tuple[int, int]], confidence: float = 1.0):
        super().__init__(class_id, left, top, width, height, segmentation, confidence=confidence)
        self.bbox = BoundingBox.from_ltwh(left, top, width, height)  # Python shenanigans

    def to_val_box(self, image_id: str, ground_truth: bool = False) -> ValBoundingBox:
        """
        Converts the annotation to a format suitable for evaluation with `pycocotools`.

        :return:
        """
        return ValBoundingBox(
            image_id,
            self.class_id,
            self.bbox.coordinates(),
            type_coordinates=CoordinatesType.ABSOLUTE,
            img_size=None,
            bb_type=BBType.GROUND_TRUTH if ground_truth else BBType.DETECTED,
            confidence=None if ground_truth else self.confidence,
            format=BBFormat.XYX2Y2
        )

    @classmethod
    def from_mung_node(cls, clss: int, node: Node) -> Self:
        return cls(
            clss,
            node.left, node.top, node.width, node.height,
            ConversionUtils.mung_segmentation_to_absolute_coordinates(node)
        )

    def intersects(self, other: Self) -> bool:
        other: COCOAnnotation
        return self.bbox.intersects(other.bbox)

    def adjust_position(self, left_shift: int = 0, top_shift: int = 0) -> None:
        """
        Adjusts classes position in place.

        :param left_shift: pixel shift to the left
        :param top_shift: pixel shift to the top
        """
        self.bbox.shift(left_shift, top_shift)

    def adjust_position_copy(self, left_shift: int, top_shift: int) -> Self:
        """
        Creates a new COCOAnnotation object with adjusted position.

        :param left_shift: pixel shift to the left
        :param top_shift: pixel shift to the top
        :return: new COCOAnnotation object with adjusted coordinates
        """
        if self.segmentation is not None:
            new_segmentation = [(x + left_shift, y + top_shift) for x, y in self.segmentation]
        else:
            new_segmentation = None

        return COCOAnnotation(
            self.class_id,

            self.bbox.left,
            self.bbox.right,
            self.bbox.width,
            self.bbox.height,

            new_segmentation,
            confidence=self.confidence
        )


class COCOFullPage(ICOCOFullPage):
    def __init__(self, image_size: tuple[int, int], annotations: list[list[ICOCOAnnotation]], class_names: list[str]):
        super().__init__(image_size, annotations, class_names)

    @staticmethod
    def _sort_annotations_by_class(annotations: list[COCOAnnotation], class_count: int) -> list[list[COCOAnnotation]]:
        output = [[] for _ in range(class_count)]
        for annot in annotations:
            output[annot.class_id].append(annot)

        return output

    @classmethod
    def from_list_of_coco_annotations(cls, image_size: tuple[int, int], annotations: list[COCOAnnotation],
                                      class_names: list[str]) -> Self:
        return cls(
            image_size,
            cls._sort_annotations_by_class(annotations, len(class_names)),
            class_names
        )

    def to_eval_format(self) -> list[tuple[list[int], float, int]]:
        output = []
        for annotation in self.all_annotations():
            output.append(annotation.to_val_box())
        return output

    def all_annotations(self) -> Generator[COCOAnnotation, None, None]:
        for row in self.annotations:
            for annotation in row:
                yield annotation

    def annotation_count(self) -> int:
        return sum([len(self.annotations[i]) for i in range(len(self.annotations))])

    @classmethod
    def from_yolo_result(cls, result: Results) -> Self:
        class_count = len(result.names)
        predictions = [[] for _ in range(class_count)]
        class_names = [result.names[i] for i in range(class_count)]
        for i in range(len(result.boxes.xywh)):
            x_center, y_center, width, height = (int(result.boxes.xywh[i, 0]), int(result.boxes.xywh[i, 1]),
                                                 int(result.boxes.xywh[i, 2]), int(result.boxes.xywh[i, 3]))
            predictions[int(result.boxes.cls[i])].append(
                COCOAnnotation(
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

    def cut_off_predictions_too_close_to_edge(self, edge_offset: int = 20, verbose: bool = False) -> None:
        width, height = self.size

        border = BoundingBox(0 + edge_offset, 0 + edge_offset, width - edge_offset, height - edge_offset)

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
            annotations: list[COCOAnnotation],
            iou_threshold: float = 0.25,
            inside_threshold: float = 0.0,
            verbose: bool = False,
    ) -> list[COCOAnnotation]:
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
                if current_annot.bbox.intersection_area(selected_annot.bbox) and (
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

    @classmethod
    def combine_multiple_pages_and_resolve(
            cls,
            subpages: list[Self],
            splits: list[list[BoundingBox]],
            edge_offset: int = 20,
            verbose: bool = False,
    ) -> Self:

        for subpage, split in zip(subpages, [x for xs in splits for x in xs]):
            subpage: COCOFullPage
            split: BoundingBox

            # cut predictions on edges
            if edge_offset > 0:
                subpage.cut_off_predictions_too_close_to_edge(edge_offset=edge_offset, verbose=verbose)

            # shift annotations based in their absolute position in image
            subpage.adjust_position_for_all_annotations(split.left, split.top)

        # retrieve important values without actually passing them as arguments
        # class names from first class
        class_names = subpages[0].class_names
        # the last split is also de facto the bottom right corner of the image,
        # we can retrieve image image_size from here,
        last_split: BoundingBox = splits[-1][-1]

        # dump all annotations into a single matrix
        completed_annotations = [[] for _ in range(len(class_names))]
        for subpage in subpages:
            for annotation in subpage.all_annotations():
                completed_annotations[annotation.class_id].append(annotation)

        # resolve overlaps etc.
        completed_annotations = [COCOFullPage.resolve_overlaps_for_list_of_annotations(class_annotations) for
                                 class_annotations in completed_annotations]

        return COCOFullPage((last_split.left, last_split.bottom), completed_annotations, class_names)


class COCOFullPageEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, COCOFullPage):
            output = {
                # "source": obj.source,
                "width": obj.size[0],
                "height": obj.size[1],
            }
            for i in range(len(obj.class_names)):
                output[obj.class_names[i]] = obj.annotations[i]
            return output
        elif isinstance(obj, COCOAnnotation):
            return COCOAnnotationEncoder().default(obj)

        return super().default(obj)


class COCOAnnotationEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, COCOAnnotation):
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


class COCOSplitPage(ICOCOSplitPage):
    def __init__(
            self,
            image_size: tuple[int, int],
            subpages: list[list[COCOFullPage]],
            class_names: list[str],
            splits: list[list[BoundingBox]]
    ):
        super().__init__(
            image_size,
            subpages,
            class_names,
            splits
        )

    @classmethod
    def from_coco_full_page(
            cls,
            full_page: COCOFullPage,
            splits: list[list[BoundingBox]],
            inside_threshold: float = 1.0
    ) -> Self:
        cutouts = []
        for row in splits:
            cutout_row = []
            for cutout in row:
                intersecting_annotations = []

                for annotation_class in full_page.annotations:
                    class_annots = []

                    for annotation in annotation_class:
                        # TODO: resolve "outside cutout", make bbox smaller

                        # DEBUG
                        # print(f"AoI is: {rec.intersection_area(cutout) / rec.area():.4f}", end=" ")
                        # if rec.intersection_area(cutout) / rec.area() >= inside_threshold:
                        #     print("ACCEPT")
                        # else:
                        #     print("reject")

                        if (annotation.bbox.intersects(cutout) and
                                annotation.bbox.intersection_area(cutout) / annotation.bbox.area() >= inside_threshold):
                            class_annots.append(annotation.adjust_position_copy(- cutout.left, - cutout.top))
                    intersecting_annotations.append(class_annots)

                cutout_row.append(COCOFullPage(
                    cutout.size(),
                    intersecting_annotations,
                    full_page.class_names
                ))

            cutouts.append(cutout_row)

        return cls(
            full_page.size,
            cutouts,
            full_page.class_names,
            splits
        )
