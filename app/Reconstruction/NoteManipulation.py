from app.Conversions.BoundingBox import Direction
from .Node import Node, VirtualNode, sort_to_strips_with_threshold


def assign_height_to_notes(measure: Node):
    half_line_height = measure.annot.bbox.height / 8
    for note in measure.children():
        x_center, _ = note.annot.bbox.center()
        distance_from_zero = measure.annot.bbox.bottom - x_center
        note.set_tag("height", distance_from_zero / half_line_height)


def assign_gs_index_to_notes(measures: list[Node], gs_index: int):
    for measure in measures:
        for note in measure.children():
            note.set_tag("gs_index", gs_index)


def note_note_to_str(note: Node) -> str:
    if note.get_tag("gs_index") is None:
        return str(round(note.get_tag("height")))
    else:
        return f"{note.get_tag('gs_index')}:{round(note.get_tag('height'))}"


def compute_note_events(linked_measures: VirtualNode, iou_threshold: float = 0.8) -> list[VirtualNode]:
    """
    Takes linked measures and computes note events from notes included in these measures.
    Returns a list of events represented as VirtualNode whose children are given notes.

    Threshold is used to determine if the next note is in the same event as the last note
    based on their horizontal overlap.

    :param linked_measures: virtual node representing the linked measures
    :param iou_threshold: threshold for note sorting to events
    :return: list of note events
    """
    all_notes: list[Node] = [note for measure in linked_measures.children() for note in measure.children()]
    strips = sort_to_strips_with_threshold(
        all_notes,
        iou_threshold,
        direction=Direction.VERTICAL,
        check_intersections=True
    )

    events: list[VirtualNode] = []
    for strip in strips:
        events.append(VirtualNode(strip))

    return events
