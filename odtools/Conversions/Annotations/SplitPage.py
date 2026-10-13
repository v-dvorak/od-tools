from pathlib import Path
from typing import Self, Optional
from copy import deepcopy

from .FullPage import FullPage, Annotation
from ..BoundingBox import BoundingBox
from ..Formats import OutputFormat


class SplitPage:
    def __init__(
        self,
        image_size: tuple[int, int],
        subpages: list[list[FullPage]],
        class_names: list[str],
        splits,
    ):
        self.size = image_size
        self.subpages = subpages
        self.class_names = class_names
        self.splits = splits

    def save_to_file(
        self,
        output_dir: Path,
        dato_name: Path | str,
        output_format: OutputFormat,
        add_tags: Optional[str] = None,
    ) -> None:
        for row in range(len(self.subpages)):
            for col in range(len(self.subpages[0])):
                self.subpages[row][col].save_to_file(
                    output_dir,
                    "-".join([str(dato_name), str(row), str(col)])
                    if add_tags is None
                    else "-".join([str(dato_name), str(row), str(col), add_tags]),
                    output_format,
                )

    @classmethod
    def from_full_page(
        cls,
        full_page: FullPage,
        splits: list[list[BoundingBox]],
        inside_threshold: float = 1.0,
        cut_to_fit: bool = False,
    ) -> Self:
        cutouts = []
        for row in splits:
            cutout_row = []
            for cutout in row:
                intersecting_annotations = []

                for annotation_class in full_page.annotations:
                    class_annots = []

                    for annotation in annotation_class:
                        annotation_to_add: Annotation | None = None
                        if annotation.bbox.intersects(cutout) and cut_to_fit:
                            annotation_to_add = cut_annotation_to_fit_cutout(
                                annotation, cutout
                            )

                        elif annotation.bbox.is_fully_inside(cutout):
                            annotation_to_add = annotation

                        if annotation_to_add is not None:
                            class_annots.append(
                                annotation_to_add.adjust_position_copy(
                                    -cutout.left, -cutout.top
                                )
                            )

                    intersecting_annotations.append(class_annots)

                cutout_row.append(
                    FullPage(
                        cutout.size(), intersecting_annotations, full_page.class_names
                    )
                )

            cutouts.append(cutout_row)

        return cls(full_page.size, cutouts, full_page.class_names, splits)


def cut_annotation_to_fit_cutout(
    annotation: Annotation, cutout: BoundingBox
) -> Annotation:

    assert annotation.bbox.intersects(cutout)
    if annotation.bbox.is_fully_inside(cutout):
        return annotation

    annotation = deepcopy(annotation)
    # Resize bbox
    prev_bbox = annotation.bbox

    annotation.bbox = annotation.bbox.cut_to_fit(cutout)

    if annotation.mask is not None:
        annotation.mask = annotation.mask[
            annotation.bbox.top - prev_bbox.top : annotation.bbox.bottom
            - prev_bbox.top,
            annotation.bbox.left - prev_bbox.left : annotation.bbox.right
            - prev_bbox.left,
        ]

    return annotation
