from pathlib import Path
from typing import Self, Optional

from .FullPage import FullPage
from ..BoundingBox import BoundingBox
from ..Formats import OutputFormat


class SplitPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            subpages: list[list[FullPage]],
            class_names: list[str],
            splits
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
            add_tags: Optional[str] = None
    ) -> None:
        for row in range(len(self.subpages)):
            for col in range(len(self.subpages[0])):
                self.subpages[row][col].save_to_file(
                    output_dir,
                    "-".join([str(dato_name), str(row), str(col)]) if add_tags is None
                    else "-".join([str(dato_name), str(row), str(col), add_tags]),
                    output_format
                )

    @classmethod
    def from_full_page(
            cls,
            full_page: FullPage,
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
                            class_annots.append(annotation.adjust_position_copy(- cutout.left, - cutout.top)) # type: ignore
                    intersecting_annotations.append(class_annots)

                cutout_row.append(FullPage(
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
