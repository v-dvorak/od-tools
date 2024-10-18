from odmetrics.bounding_box import ValBoundingBox
from odmetrics.utils.enumerators import (BBFormat, BBType, CoordinatesType)
from .Interfaces import IYOLODetection, IYOLOSegmentation, IYOLOFullPageDetection, IYOLOFullPageSegmentation
from ..COCO.Interfaces import ICOCOFullPage, ICOCOAnnotation


class YOLODetection(IYOLODetection):
    def __init__(self, class_id: int, x_center: float, y_center: float, width: float, height: float,
                 confidence: float = 1.0):
        super().__init__(class_id, x_center, y_center, width, height, confidence)

    @classmethod
    def from_coco_annotation(cls, annot: ICOCOAnnotation, image_size: tuple[int, int]):
        img_width, img_height = image_size
        return YOLODetection(
            annot.class_id,
            (annot.bbox.left + annot.bbox.width / 2) / img_width,
            (annot.bbox.top + annot.bbox.height / 2) / img_height,
            annot.bbox.width / img_width,
            annot.bbox.height / img_height,
        )

    def to_val_box(self, image_id: int | str, image_size: tuple[int, int],
                   ground_truth: bool = False) -> ValBoundingBox:
        return ValBoundingBox(
            image_id,
            self.class_id,
            (self.x_center, self.y_center, self.width, self.height),
            type_coordinates=CoordinatesType.RELATIVE,
            img_size=image_size,
            bb_type=BBType.GROUND_TRUTH if ground_truth else BBType.DETECTED,
            confidence=None if ground_truth else self.confidence,
            format=BBFormat.YOLO
        )


class YOLOFullPageDetection(IYOLOFullPageDetection):
    def __init__(self, image_size: tuple[int, int], annotations: list[YOLODetection]):
        super().__init__(image_size, annotations)

    @classmethod
    def from_coco_page(cls, page: ICOCOFullPage):
        return cls(
            page.size,
            [
                YOLODetection.from_coco_annotation(annot, page.size)
                for annot_class in page.annotations
                for annot in annot_class
            ]
        )

    @classmethod
    def from_yolo_file(cls, file_path: str, image_size: tuple[int, int]):
        parsed_data = []
        with open(file_path, "r") as file:
            for line in file:
                values = line.strip().split()
                class_id = int(values[0])
                x = float(values[1])
                y = float(values[2])
                w = float(values[3])
                h = float(values[4])
                parsed_data.append(YOLODetection(class_id, x, y, w, h))
        return cls(image_size, parsed_data)


class YOLOSegmentation(IYOLOSegmentation):
    def __init__(self, class_id: int, coordinates: list[tuple[float, float]], confidence: float = 1.0):
        super().__init__(class_id, coordinates, confidence)

    @classmethod
    def from_coco_annotation(cls, annot: ICOCOAnnotation, image_size: tuple[int, int]):
        width, height = image_size
        return cls(
            annot.class_id,
            [(x / width, y / height) for (x, y) in annot.segmentation],
        )


class YOLOFullPageSegmentation(IYOLOFullPageSegmentation):
    def __init__(self, image_size: tuple[int, int], annotations: list[YOLOSegmentation]):
        super().__init__(image_size, annotations)

    @classmethod
    def from_coco_page(cls, page: ICOCOFullPage):
        return cls(
            page.size,
            [
                YOLOSegmentation.from_coco_annotation(annot, page.size)
                for annot_class in page.annotations
                for annot in annot_class
            ]
        )
