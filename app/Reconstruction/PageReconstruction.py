import itertools
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from .Graph.Names import NodeName
from .Graph.Node import Node, VirtualNode, sort_to_strips_with_threshold
from .Graph.Tags import SYMBOL_PITCH_TAG, NOTEHEAD_TYPE_TAG, SYMBOL_GS_INDEX_TAG, ACCIDENTAL_TYPE_TAG
from .MeasureManipulation import SectionType, link_measures_inside_grand_staff
from .NoteManipulation import _assign_gs_index_to_notes
from .VizUtils import write_numbers_on_image, print_info
from ..Conversions.BoundingBox import Direction


def sort_page_into_sections(
        measures: list[Node],
        grand_staff: list[Node]
) -> list[tuple[SectionType, list[Node]]]:
    """
    Sorts given list of Measures into two types of sections: in and out of grand staff,
    based on the list of given grand staff.

    :param measures: list of measures to sort
    :param grand_staff: list of grand staff
    :return: list of tuples (in/out section type, list of nodes)
    """
    grand_staff = sorted(grand_staff, key=lambda g: g.annot.bbox.top)
    measures = sorted(measures, key=lambda m: m.annot.bbox.top)
    sections: list[tuple[SectionType, list[Node]]] = []

    section = []
    gs_index = 0
    in_gs = False

    for measure in measures:
        # ran out of gs to assign measures to, dump them to last section of page
        if gs_index >= len(grand_staff):
            section.append(measure)
            in_gs = False
            continue

        current_gs = grand_staff[gs_index]
        intersects = measure.annot.bbox.intersects(current_gs.annot.bbox)

        # measure is inside gs and it is the first one found
        if intersects and not in_gs:
            # edge case, when the algorithm starts inside gs, this could otherwise create an empty section
            if len(section) > 0:
                sections.append((SectionType.OUT_GS, section))
            section = [measure]
            in_gs = True
        # measure is inside gs and the gs is the same as the last one
        elif intersects and in_gs:
            section.append(measure)
        # measure is outside any gs, same as the last one
        elif not intersects and not in_gs:
            section.append(measure)
        # measure is outside gs and the last one was inside
        elif not intersects and in_gs:
            sections.append((SectionType.IN_GS, section))
            section = [measure]

            # it can intersect with the next gs
            if (gs_index + 1 < len(grand_staff)
                    and measure.annot.bbox.intersects(grand_staff[gs_index + 1].annot.bbox)):
                in_gs = True
            else:
                in_gs = False

            # switch to next gs
            gs_index += 1
        else:
            raise NotImplementedError()

    # append last section
    sections.append((SectionType.IN_GS if in_gs else SectionType.OUT_GS, section))
    return sections


def link_measures_based_on_grand_staffs(
        measures: list[Node],
        grand_staffs: list[Node],
        mriou_threshold: float,
        image_path: Path = None,
        verbose: bool = False,
        visualize: bool = False,
) -> list[VirtualNode]:
    """
    Sorts symbols from given measures into groups based on measure relationships.
    Returns a list of ``VirtualNode`` named `measure group` where all the symbols
    are played on the same instrument during a single measure.
    (Groups are either formed from symbols from simple measures or from grand staff measures.)

    MRIoU (measure reading IoU) determines if the next measure is in the same staff as the last measure
    based on their vertical overlap.

    :param measures: list of measures to sort
    :param grand_staffs: list of grand staff
    :param mriou_threshold: "measure reading" IoU, determines whether two measures belong to the same staff based on IoU
    :param image_path: path to image
    :param verbose: make script verbose
    :param visualize: show visualizations
    :return: list of symbols grouped by measures
    """
    if image_path is None and visualize:
        raise ValueError("Image path is required when visualize is set to True.")

    # SORT MEASURES INTO SECTION IN/OUT OF GRAND STAFF
    sections = sort_page_into_sections(measures, grand_staffs)

    # SORT SECTIONS INTO ROWS OF MEASURES
    sorted_sections: list[tuple[SectionType, list[list[Node]]]] = []
    for section_type, section in sections:
        # keep section type and sort measures inside by reading order
        sorted_sections.append(
            (section_type,
             sort_to_strips_with_threshold(
                 section,
                 mriou_threshold,
                 direction=Direction.HORIZONTAL
             )))

    if verbose:
        print_info(
            "Sections sorted by reading order",
            "in/out: number of measures in each row",
            [f"{section_type}: {', '.join([str(len(subsection)) for subsection in s_section])}"
             for section_type, s_section in sorted_sections]
        )

    if visualize:
        print("Showing measures reading order...")
        dumped_measures = list(itertools.chain.from_iterable([m for ms in sorted_sections for m in ms[1]]))
        write_numbers_on_image(image_path, dumped_measures)
        input("Press Enter to continue")

    # PREPARE MEASURES FOR EVENT DETECTION
    linked_measures: list[VirtualNode] = []

    for section_type, s_section in sorted_sections:
        # this is the true grand staff
        if section_type == SectionType.IN_GS and len(s_section) == 2:
            # tag staff that belong to it
            _assign_gs_index_to_notes(s_section[0], 1)
            _assign_gs_index_to_notes(s_section[1], 0)
            # and link individual measures together
            linked_measures.extend(link_measures_inside_grand_staff(s_section[0], s_section[1]))

        # this is a section of many single staffs (or something undefined)
        else:
            for staff in s_section:
                for measure in staff:
                    # list of mesures makes it easier to adapt the following algorithms
                    linked_measures.append(VirtualNode([measure]))

    if verbose:
        print("Linked measures sorted by reading order")
        print(", ".join([str(len(e.children())) for e in linked_measures]))
        print()

    if verbose:
        print_info(
            "Detected sections",
            "in/out: number of measures",
            [f"{section_type}: {len(section)}" for section_type, section in sections]
        )

    measure_grouping: list[VirtualNode] = []
    for lm in linked_measures:
        # dump all group symbols into one
        measure_grouping.append(
            VirtualNode([c for measure in lm.children() for c in measure.children()], name=NodeName.SYMBOL_GROUP))

    return measure_grouping


def compute_note_events(note_group: list[Node], neiou_threshold: float = 0.8) -> list[VirtualNode]:
    """
    Takes group of notes (noteheads) and computes note events from notes included in these measures.
    Returns a list of events represented as VirtualNode whose children are given notes.

    Threshold is used to determine if the next note is in the same event as the last note
    based on their horizontal overlap.

    :param note_group: virtual node representing the linked measures
    :param neiou_threshold: threshold for note sorting to events
    :return: list of note events
    """
    strips = sort_to_strips_with_threshold(
        note_group,
        neiou_threshold,
        direction=Direction.VERTICAL,
        check_intersections=True
    )

    events: list[VirtualNode] = []
    for strip in strips:
        events.append(VirtualNode(strip, name=NodeName.NOTE_EVENT))

    return events


def compute_note_events_for_page(
        linked_measures: list[VirtualNode],
        neiou_treshold: float,
        image_path: Path = None,
        verbose: bool = False,
        visualize: bool = False
) -> list[VirtualNode]:
    if image_path is None and visualize:
        raise ValueError("Image path is required when visualize is set to True.")

    measure_groups: list[VirtualNode] = []
    for mes in linked_measures:
        events_in_measure = compute_note_events(
            # filter out notes
            [note for note in mes.children() if note.name == NodeName.NOTEHEAD],
            neiou_threshold=neiou_treshold
        )
        group_symbols = sorted(
            events_in_measure + [symbol for symbol in mes.children() if symbol.name != NodeName.NOTEHEAD],
            key=lambda x: x.total_bbox.left
        )

        measure_groups.append(
            VirtualNode(group_symbols,
                        name=NodeName._MEASURER_GROUP))

    if visualize:
        flat_list: list[Node] = [note for group in measure_groups
                                 for event in group.children()
                                 for note in event.children()
                                 if event.name == NodeName.NOTE_EVENT]
        print("Showing note reading order...")
        write_numbers_on_image(image_path, flat_list)
        input("Press enter to continue")

    return measure_groups


def _symbol_pitch_to_str(note: Node) -> int:
    # skip python default rounding (0.5 should be rounded to 1)
    return int(Decimal(note.get_tag(SYMBOL_PITCH_TAG)).to_integral(ROUND_HALF_UP))


def _notehead_to_string(note: Node) -> str:
    gs_index = note.get_tag(SYMBOL_GS_INDEX_TAG)
    pitch = _symbol_pitch_to_str(note)
    if gs_index is not None:
        return f"{gs_index}{note.get_tag(NOTEHEAD_TYPE_TAG)}{pitch}"
    else:
        return f"{note.get_tag(NOTEHEAD_TYPE_TAG)}{pitch}"


def _accident_to_string(note: Node) -> str:
    gs_index = note.get_tag(SYMBOL_GS_INDEX_TAG)
    pitch = _symbol_pitch_to_str(note)
    if gs_index is not None:
        return f"{gs_index}{note.get_tag(ACCIDENTAL_TYPE_TAG)}{pitch}"
    else:
        return f"{note.get_tag(ACCIDENTAL_TYPE_TAG)}{pitch}"


def symbol_to_str(note: Node) -> str:
    match note.name:
        case NodeName.NOTEHEAD:
            return _notehead_to_string(note)
        case NodeName.ACCIDENTAL:
            return _accident_to_string(note)
        case _:
            raise ValueError(f"Unknown symbol type {note.name}")


def linearize_note_events(measure_groups: list[VirtualNode]) -> str:
    measure_presentation: list[str] = []

    for measure in measure_groups:
        mr = []
        event_with_acc: list[str] = []
        for child in measure.children():
            # process note event
            if child.name == NodeName.NOTE_EVENT:
                event_with_acc.extend([symbol_to_str(note) for note in child.children()])
                mr.append(" ".join(event_with_acc))
                event_with_acc = []
            # process accidental
            elif child.name == NodeName.ACCIDENTAL:
                child: Node
                event_with_acc.append(symbol_to_str(child))
            else:
                raise ValueError(f"Unknown node type \"{child.name}\"")

        measure_presentation.append(" | ".join(mr))

    return " || ".join(measure_presentation)


from .NoteManipulation import assign_notes_to_measures_and_compute_pitch
from .VizUtils import visualize_result


def reconstruct_note_events(
        measures: list[Node],
        grand_staffs: list[Node],
        symbols_with_pitch: list[Node],
        ual_factor: float = 1.5,
        mriou_threshold: float = 0.5,
        neiou_threshold: float = 0.4,
        image_path: Path = None,
        verbose: bool = False,
        visualize: bool = False
) -> list[VirtualNode]:
    if image_path is None and visualize:
        raise ValueError("Image path is required when visualize is set to True.")

    assign_notes_to_measures_and_compute_pitch(
        measures,
        symbols_with_pitch,
        ual_factor=ual_factor,
        image_path=image_path,
        verbose=verbose,
        visualize=visualize
    )

    linked_measures = link_measures_based_on_grand_staffs(
        measures,
        grand_staffs,
        mriou_threshold,
        image_path=image_path,
        verbose=verbose,
        visualize=visualize
    )

    measure_groups = compute_note_events_for_page(
        linked_measures,
        neiou_threshold,
        image_path=image_path,
        verbose=verbose,
        visualize=visualize
    )

    if visualize:
        print("Showing end result...")
        visualize_result(
            image_path,
            measures,
            [ob for group in measure_groups for ob in group.children()],
            grand_staffs
        )
        input("Press enter to continue")

    if verbose:
        print(linearize_note_events(measure_groups))

    return measure_groups
