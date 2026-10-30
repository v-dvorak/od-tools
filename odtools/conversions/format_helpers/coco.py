import json
from json import JSONEncoder
from pathlib import Path
from typing import TYPE_CHECKING

from ...annotations import Annotation, AnnotationType
from .base import _HelperBase
from ..formats import Format

if TYPE_CHECKING:
    from ...annotations.full_page import FullPage


class COCOHelper(_HelperBase):
    FORMAT_TYPE = Format.COCO
    __supports_from_file__ = True
    __supports_to_file__ = True

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

        with open(annotation_path, "r") as file:
            data = json.load(file)
        image_width, image_height = data["width"], data["height"]
        annots = [[] for _ in range(len(class_output_names))]
        for class_name, class_id in class_reference_table.items():
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
                        segm.append(
                            (
                                int(annot["segmentation"][0][i]),
                                int(annot["segmentation"][0][i + 1]),
                            )
                        )
                        i += 2

                # save parsed annotation
                annots[class_id].append(
                    Annotation(
                        class_id,
                        left,
                        top,
                        width,
                        height,
                        segm,  # type: ignore
                        annot.get("name") or class_name,
                        an_type=an_type,
                    )
                )

        return FullPage((image_width, image_height), annots, class_output_names)

    @staticmethod
    def to_file(
        page: "FullPage",
        output_path: Path,
        with_confidence: bool = False,
    ) -> None:
        with open(output_path, "w") as f:
            json.dump(page, f, indent=4, cls=COCOFullPageEncoder)


class COCOFullPageEncoder(JSONEncoder):
    def default(self, o):
        from ...annotations.full_page import FullPage

        if isinstance(o, FullPage):
            output = {
                # "source": obj.source,
                "width": o.size[0],
                "height": o.size[1],
            }
            for i in range(len(o.class_names)):
                output[o.class_names[i]] = o.annotations[i]  # type: ignore
            return output
        elif isinstance(o, Annotation):
            return COCOAnnotationEncoder().default(o)

        return super().default(o)


class COCOAnnotationEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Annotation):
            # flatten
            segm = []
            for x, y in o.mask:  # type: ignore
                segm.append(x)
                segm.append(y)

            return {
                "name": o.class_name,
                "left": o.bbox.left,
                "top": o.bbox.top,
                "width": o.bbox.width,
                "height": o.bbox.height,
                "segmentation": [segm],
            }
        return super().default(o)
