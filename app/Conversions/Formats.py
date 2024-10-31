from enum import Enum


class InputFormat(Enum):
    MUNG = 1,
    COCO = 2,
    YOLO_DETECTION = 3,
    YOLO_SEGMENTATION = 4

    @staticmethod
    def from_string(input_format: str):
        input_format = input_format.lower()
        if input_format == "mung":
            return InputFormat.MUNG
        elif input_format == "coco":
            return InputFormat.COCO
        elif input_format == "yolod":
            return InputFormat.YOLO_DETECTION
        elif input_format == "yolos":
            return InputFormat.YOLO_SEGMENTATION
        else:
            return ValueError(f"Invalid input format: {input_format}")

    def to_annotation_extension(self) -> str:
        input_format: InputFormat
        if self == InputFormat.MUNG:
            return "xml"
        elif self == InputFormat.COCO:
            return "json"
        elif self == InputFormat.YOLO_DETECTION or self == InputFormat.YOLO_SEGMENTATION:
            return "txt"


class OutputFormat(Enum):
    MUNG = 1,
    COCO = 2,
    YOLO_DETECTION = 3,
    YOLO_SEGMENTATION = 4

    @staticmethod
    def from_string(output_format: str):
        output_format = output_format.lower()
        if output_format == "mung":
            return OutputFormat.MUNG
        elif output_format == "coco":
            return OutputFormat.COCO
        elif output_format == "yolod":
            return OutputFormat.YOLO_DETECTION
        elif output_format == "yolos":
            return OutputFormat.YOLO_SEGMENTATION
        else:
            raise ValueError(f"Invalid output format: {output_format}")

    def to_annotation_extension(self) -> str:
        input_format: InputFormat
        if self == OutputFormat.MUNG:
            return "xml"
        elif self == OutputFormat.COCO:
            return "json"
        elif self == OutputFormat.YOLO_DETECTION or self == InputFormat.YOLO_SEGMENTATION:
            return "txt"
