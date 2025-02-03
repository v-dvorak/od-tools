from .Graph.Node import Node
from .Graph.Tags import NOTEHEAD_TYPE_TAG, NoteheadType
from ..Conversions.Annotations.Annotation import Annotation


def prepare_annots_for_reconstruction(
        measures_a: list[Annotation],
        grand_staffs_a: list[Annotation],
        noteheads_a: list[tuple[NoteheadType, list[Annotation]]]
) -> list[list[Node]]:
    all_noteheads: list[list[Node]] = []
    for nh_type, annots in noteheads_a:
        nhs = []
        for annot in annots:
            n = Node(annot)
            n.set_tag(NOTEHEAD_TYPE_TAG, nh_type)
            nhs.append(n)
        all_noteheads.append(nhs)

    measures: list[Node] = []
    for measure in measures_a:
        measures.append(Node(measure))

    grand_staffs: list[Node] = []
    for gs in grand_staffs_a:
        grand_staffs.append(Node(gs))

    return [measures, grand_staffs, *all_noteheads]

