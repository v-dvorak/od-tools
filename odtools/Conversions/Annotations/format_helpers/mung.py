from pathlib import Path
import numpy as np
from itertools import chain
from typing import TYPE_CHECKING

from mung.io import read_nodes_from_file, write_nodes_to_file
from mung.graph import Node
from ..Annotation import Annotation
from ..annotation_type import AnnotationType
from ... import ConversionUtils
if TYPE_CHECKING:
    from ..FullPage import FullPage


class _MuNGHelper:
    @staticmethod
    def from_mung(
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> "FullPage":
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
    ) -> "FullPage":
        from ..FullPage import FullPage
        nodes = read_nodes_from_file(annot_path.__str__())

        annots = []
        # process each node
        for node in nodes:
            if node.class_name in class_reference_table:
                annots.append(Annotation.from_mung_node(class_reference_table[node.class_name], node, an_type=an_type))

        # create single page
        full_page = FullPage.from_list_of_coco_annotations(image_size, annots, class_output_names)
        return full_page

    @staticmethod
    def save_annotation(
        page: "FullPage",
        output_path: Path
    ) -> None:
        nodes = []
        id_ = 0
        for annot in chain.from_iterable(page.annotations):
            nodes.append(Node(
                id_,
                page.class_names[annot.class_id],
                top=annot.bbox.top,
                left=annot.bbox.left,
                width=annot.bbox.width,
                height=annot.bbox.height,
                mask=np.ones((annot.bbox.height, annot.bbox.width)),
                # data={"confidence": annot.confidence}
            ))
            id_ += 1
        write_nodes_to_file(nodes, str(output_path))
