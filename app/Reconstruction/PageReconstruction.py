import itertools
from pathlib import Path

from .Graph.Node import Node, VirtualNode, sort_to_strips_with_threshold
from .MeasureManipulation import SectionType, link_measures_inside_grand_staff
from .NoteManipulation import _assign_gs_index_to_notes, note_node_to_str
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

    return linked_measures


def compute_note_events(linked_measures: VirtualNode, neiou_threshold: float = 0.8) -> list[VirtualNode]:
    """
    Takes linked measures and computes note events from notes included in these measures.
    Returns a list of events represented as VirtualNode whose children are given notes.

    Threshold is used to determine if the next note is in the same event as the last note
    based on their horizontal overlap.

    :param linked_measures: virtual node representing the linked measures
    :param neiou_threshold: threshold for note sorting to events
    :return: list of note events
    """
    all_notes: list[Node] = [note for measure in linked_measures.children() for note in measure.children()]
    strips = sort_to_strips_with_threshold(
        all_notes,
        neiou_threshold,
        direction=Direction.VERTICAL,
        check_intersections=True
    )

    events: list[VirtualNode] = []
    for strip in strips:
        events.append(VirtualNode(strip))

    return events


def compute_note_events_for_page(
        linked_measures: list[VirtualNode],
        neiou_treshold: float,
        image_path: Path = None,
        verbose: bool = False,
        visualize: bool = False
) -> list[list[VirtualNode]]:
    if image_path is None and visualize:
        raise ValueError("Image path is required when visualize is set to True.")
    
    events_by_measure: list[list[VirtualNode]] = []
    for mes in linked_measures:
        events_by_measure.append(compute_note_events(mes, neiou_threshold=neiou_treshold))

    if visualize:
        flat_list: list[Node] = [item for sublist1 in events_by_measure
                                 for sublist2 in sublist1
                                 for item in sublist2.children()]
        print("Showing note reading order...")
        write_numbers_on_image(image_path, flat_list)
        input("Press enter to continue")

    return events_by_measure


def linearize_note_events(events_by_measure: list[list[VirtualNode]]) -> str:
    # measure sep
    representation = " || ".join(
        # event sep
        [" | ".join(
            # note sep
            [" ".join(
                [note_node_to_str(n) for n in event.children()]
            )
                for event in measure]
        )
            for measure in events_by_measure]
    )

    if len(representation) != 0:
        representation = "|| " + representation + " ||"

    return representation


from .NoteManipulation import assign_notes_to_measures_and_compute_pitch
from .VizUtils import visualize_result


def reconstruct_note_events(
        measures: list[Node],
        grand_staffs: list[Node],
        notes: list[Node],
        ual_factor: float = 1.5,
        mriou_threshold: float = 0.5,
        neiou_threshold: float = 0.4,
        image_path: Path = None,
        verbose: bool = False,
        visualize: bool = False
) -> list[list[VirtualNode]]:
    if image_path is None and visualize:
        raise ValueError("Image path is required when visualize is set to True.")

    assign_notes_to_measures_and_compute_pitch(
        measures,
        notes,
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

    events_by_measure = compute_note_events_for_page(
        linked_measures,
        neiou_threshold,
        image_path=image_path,
        verbose=verbose,
        visualize=visualize
    )

    if visualize:
        print("Showing end result...")
        visualize_result(image_path, measures, [e for es in events_by_measure for e in es], grand_staffs)
        input("Press enter to continue")

    if verbose:
        print(linearize_note_events(events_by_measure))

    return events_by_measure
