import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ...annotations import Annotation, AnnotationType
from .base import _HelperBase
from ..formats import Format

if TYPE_CHECKING:
    from ...annotations.full_page import FullPage


class DoloresCOCOHelper(_HelperBase):
    FORMAT_TYPE = Format.DOLORES_COCO
    __supports_from_file__ = True

    @staticmethod
    def from_file(
        annotation_path: Path,
        image_path: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        an_type: AnnotationType = AnnotationType.GROUND_TRUTH,
        with_confidence: bool = False,
    ) -> "FullPage":
        from ...annotations.full_page import FullPage

        with open(annotation_path, "r", encoding="utf8") as f:
            data = json.load(f)
        if len(data["images"]) != 1:
            raise ValueError(
                f"Expected exactly one image in DOLORES COCO file, got {len(data['images'])}"
            )

        # get image info
        image_info = data["images"][0]
        image_width, image_height = image_info["width"], image_info["height"]

        # collected annotations
        annots: list[list[Annotation]] = [[] for _ in range(len(class_output_names))]
        # class id to class names inside the loaded document
        class_id_to_name: dict[int, str] = {
            c["id"]: c["name"] for c in data["categories"]
        }
        for annotation in data["annotations"]:
            # convert loaded names to names wanted by the loading script
            class_name = class_id_to_name[annotation["category_id"]]
            global_class_id = class_reference_table.get(class_name)
            if global_class_id is None:
                continue
            # create annotation
            left, top, width, height = annotation["bbox"]
            if (
                left > image_width
                or left + width > image_width
                or top > image_height
                or top + height > image_height
            ):
                warnings.warn(
                    f"Bbox {class_name, left, top, width, height} out of bounds, "
                    f"file name: {image_info['file_name']}, object id: {annotation['id']}"
                )
            try:
                annots[global_class_id].append(
                    Annotation(
                        global_class_id,
                        left,
                        top,
                        width,
                        height,
                        mask=None,
                        class_name=class_name,
                        an_type=an_type,
                    )
                )
            except ValueError as e:
                warnings.warn(
                    f"{e!s}, file name: {image_info['file_name']}, object id: {annotation['id']}"
                )

        return FullPage((image_width, image_height), annots, class_output_names)
