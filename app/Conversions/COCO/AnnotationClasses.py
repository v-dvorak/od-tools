from json import JSONEncoder
from typing import Self

from mung.node import Node
from ultralytics.engine.results import Results

from .Interfaces import ICOCOAnnotation, ICOCOFullPage, ICOCOSplitPage
from .. import ConversionUtils
from ...Splitting.SplitUtils import BoundingBox


class COCOAnnotation(ICOCOAnnotation):
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int,
                 segmentation: list[tuple[int, int]], confidence: float = 1.0):
        super().__init__(class_id, left, top, width, height, segmentation, confidence=confidence)

    @classmethod
    def from_mung_node(cls, clss: int, node: Node) -> Self:
        return cls(
            clss,
            node.left, node.top, node.width, node.height,
            ConversionUtils.mung_segmentation_to_absolute_coordinates(node)
        )

    def intersect(self, other: Self) -> bool:
        other: COCOAnnotation
        return (
                self.left <= other.left + self.width
                and self.left + self.width >= other.left
                and self.top <= other.top + self.height
                and self.top + self.height >= other.top
        )

    def to_rectangle(self):
        return BoundingBox(self.left, self.top, self.left + self.width, self.top + self.height)

    def adjust_position(self, left_shift: int = 0, top_shift: int = 0) -> None:
        """
        Adjusts classes position in place.

        :param left_shift: pixel shift to the left
        :param top_shift: pixel shift to the top
        """
        self.left += left_shift
        self.top += top_shift

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
            self.left + left_shift,
            self.top + top_shift,
            self.width,
            self.height,
            new_segmentation,
            confidence=self.confidence
        )


class COCOFullPage(ICOCOFullPage):
    def __init__(self, image_size: tuple[int, int], annotations: list[list[ICOCOAnnotation]], class_names: list[str]):
        super().__init__(image_size, annotations, class_names)

    # TODO: change annotation implementation from (left, top, width, height) to BoundingBox
    # -> reuse rectangle functions, single attribute inside class
    # -> maybe rename it to BoundingBox

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
                "left": obj.left,
                "top": obj.top,
                "width": obj.width,
                "height": obj.height,
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
                        rec = BoundingBox.from_coco_annotation(annotation)
                        # TODO: resolve "outside cutout", make bbox smaller

                        # DEBUG
                        # print(f"AoI is: {rec.intersection_area(cutout) / rec.area():.4f}", end=" ")
                        # if rec.intersection_area(cutout) / rec.area() >= inside_threshold:
                        #     print("ACCEPT")
                        # else:
                        #     print("reject")

                        if (rec.intersects(cutout) and
                                rec.intersection_area(cutout) / rec.area() >= inside_threshold):
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
