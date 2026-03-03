
from pathlib import Path
from typing import TYPE_CHECKING

from ..Annotation import Annotation
from ..annotation_type import AnnotationType
from ... import ConversionUtils
if TYPE_CHECKING:
    from ..FullPage import FullPage


class _YOLOHelper:
    @staticmethod
    def from_yolo_detection(
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> "FullPage":
        from ..FullPage import FullPage
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
    ) -> "FullPage":
        from ..FullPage import FullPage
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
            an_type: AnnotationType
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
            page: "FullPage",
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
