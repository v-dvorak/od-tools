from pathlib import Path
import numpy as np
from itertools import chain
from typing import TYPE_CHECKING

from mung.io import read_nodes_from_file, write_nodes_to_file
from mung.graph import Node
from ...annotations import Annotation, AnnotationType
from .. import conversion_utils
from ..formats import Format
from .base import _HelperBase

if TYPE_CHECKING:
    from ...annotations import FullPage


class MuNGHelper(_HelperBase):
    FORMAT_TYPE = Format.MUNG
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
        image_size = conversion_utils.get_num_pixels(image_path)
        return MuNGHelper.from_mung_file(
            annotation_path,
            image_size,
            class_reference_table,
            class_output_names,
            an_type=an_type,
        )

    @staticmethod
    def from_mung_file(
        annot_path: Path,
        image_size: tuple[int, int],
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        an_type: AnnotationType = AnnotationType.GROUND_TRUTH,
    ) -> "FullPage":
        from ...annotations.full_page import FullPage

        nodes = read_nodes_from_file(annot_path.__str__())

        annots = []
        # process each node
        for node in nodes:
            if node.class_name in class_reference_table:
                annots.append(
                    Annotation.from_mung_node(
                        class_reference_table[node.class_name], node, an_type=an_type
                    )
                )

        # create single page
        full_page = FullPage.from_list_of_coco_annotations(
            image_size, annots, class_output_names
        )
        return full_page

    @staticmethod
    def to_file(
        page: "FullPage",
        output_path: Path,
        with_confidence: bool = False,
    ) -> None:
        nodes = []
        for id_, annot in enumerate(chain.from_iterable(page.annotations)):
            nodes.append(
                Node(
                    id_,
                    annot.class_name,
                    top=annot.bbox.top,
                    left=annot.bbox.left,
                    width=annot.bbox.width,
                    height=annot.bbox.height,
                    mask=np.ones((annot.bbox.height, annot.bbox.width)),
                    data={"confidence": annot.confidence} if with_confidence else {},
                )
            )
        write_nodes_to_file(nodes, str(output_path))
