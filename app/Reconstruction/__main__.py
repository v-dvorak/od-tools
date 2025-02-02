import itertools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .DebugViz import write_note_heights_to_image, write_numbers_on_image, visualize_result
from .MeasureManipulation import SectionType, sort_page_into_sections
from .MeasureManipulation import link_measures_inside_grand_stave
from .Node import Node, VirtualNode, assign_to_closest, sort_to_strips_with_threshold
from .NoteManipulation import assign_height_to_notes, compute_note_events, note_note_to_str, \
    assign_gs_index_to_notes
from ..Conversions.Annotations.FullPage import FullPage
from ..Conversions.BoundingBox import Direction
from ..Conversions.ConversionUtils import get_num_pixels
from ..Splitting.SplitUtils import create_split_box_matrix, draw_rectangles_on_image
from ..Val.Utils import create_split_images

image_path = "demo/30d6c780-c8fe-11e7-9c14-005056827e51_375c6850-f593-11e7-b30f-5ef3fc9ae867.jpg"
image_path = "sheet-test.jpg"
# not ideal
image_path = "F:/Studium/4/rocnikac/od-tools/datasets/2024-10-07_proto-dataset/images/81c9f683-28d1-4e73-8e25-e37333408f5a_ac45624e-0846-4c6d-a079-a1f1877e1aea.jpg"
# ideal
# image_path = "F:/Studium/4/rocnikac/od-tools/datasets/2024-10-07_proto-dataset/images/bf061840-2322-11eb-979b-005056827e52_2e117f2e-4c19-4bc3-ba6b-5531ca623e22.jpg"
# ideal, more sophisticated
# image_path = "F:/Studium/4/rocnikac/od-tools/datasets/2024-10-07_proto-dataset/images/81c9f683-28d1-4e73-8e25-e37333408f5a_5b6164cc-5653-494b-b43f-946fbb64d440.jpg"
# image = "demo/3bb9e322-bc61-4307-856b-6f8fb1a640df_2d5f652c-1df0-474c-ae23-3fb699afe808.jpg"

visualize = False

notehead_detector = YOLO("notehead-detector.pt")
staff_detector = YOLO("ola-layout-analysis-1.0-2024-10-10.pt")
image_path = Path(image_path)


class YOLO_Module:
    @staticmethod
    def single_image_detection(
            image_path: Path,
            model: YOLO,
            bw: bool = False
    ) -> FullPage:
        image = Image.open(image_path)
        if bw:
            image = image.convert("L")
        result = model.predict(image, save=False, save_txt=False, verbose=False)[0]
        return FullPage.from_yolo_result(result)

    @staticmethod
    def split_image_detection(
            image_path: Path,
            model: YOLO,
            window_size: tuple[int, int] = (640, 640),
            overlap_ratio: float = 0.25,
            iou_threshold: float = 0.25,
    ) -> FullPage:
        splits = create_split_box_matrix(
            get_num_pixels(image_path),
            window_size=window_size,
            overlap_ratio=overlap_ratio
        )
        tiles = create_split_images(image_path, splits)
        results = model.predict(tiles, save=False, save_txt=False, verbose=False)

        subpages: list[FullPage] = []
        for result in results:
            subpages.append(FullPage.from_yolo_result(result))

        return FullPage.combine_multiple_pages_and_resolve(subpages, splits, iou_threshold=iou_threshold)


# predict staff
staff_page = YOLO_Module.single_image_detection(image_path, staff_detector, bw=True)

# predict noteheads
notehead_page = YOLO_Module.split_image_detection(image_path, notehead_detector, overlap_ratio=0.1)

combined = FullPage(
    staff_page.size,
    staff_page.annotations + notehead_page.annotations,
    staff_page.class_names + notehead_page.class_names
)

# ['system_measures', 'stave_measures', 'staves', 'systems', 'grand_staff', 'noteheadFull', 'noteheadHalf']

if visualize:
    viz_data = [
        ((0, 0, 255), [x.bbox for xs in staff_page.annotations for x in xs]),  # staff
        ((0, 255, 0), [x.bbox for x in notehead_page.annotations[0]]),
        ((255, 0, 0), [x.bbox for x in notehead_page.annotations[1]]),
    ]

    temp = cv2.imread(image_path)
    for (i, (color, data)) in enumerate(viz_data):
        temp = draw_rectangles_on_image(
            temp,
            data,
            color=color,
            thickness=2,
            show=(i == len(viz_data) - 1)
        )

# INITIALIZE GRAPH
notehead_full = []
for note in combined.annotations[5]:
    notehead_full.append(Node(note))

notehead_half = []
for note in combined.annotations[6]:
    notehead_half.append(Node(note))

measures = []
for measure in combined.annotations[1]:
    measures.append(Node(measure))

grand_staff = []
for gs in combined.annotations[4]:
    grand_staff.append(Node(gs))


def _print_info(name: str, header: str, content: list[str], separator: str = "-"):
    print()
    print(len(header) * separator)
    print(name)
    if header is not None:
        print(header)
    print(len(header) * separator)
    print("\n".join(content))
    print(len(header) * separator)


def reconstruct_note_events(
        measures: list[Node],
        grand_staff: list[Node],
        notes: list[Node],
        verbose: bool = False,
        visualize: bool = False,
        ual_factor: float = 3 / 2,
        mriou_threshold: float = 0.5,
) -> list[list[VirtualNode]]:
    # ASSIGN NOTES TO MEASURES AND CALCULATE NOTE HEIGHTS
    # the method assigns notes to measures based on center of the detection
    # "* (3 / 2)" means +- one whole average measure from top and bottom from chosen measure
    upper_assignment_limit = np.mean([m.annot.bbox.height for m in measures]) * ual_factor
    assign_to_closest(measures, notes, upper_limit=upper_assignment_limit)

    # ASSIGN HEIGHT TO EACH NOTE
    for measure in measures:
        assign_height_to_notes(measure)

    if visualize:
        print("Showing note heights...")
        measure_height_viz = draw_rectangles_on_image(
            image_path,
            [m.total_bbox for m in measures]
        )
        write_note_heights_to_image(
            measure_height_viz,
            [n for m in measures for n in m.children()]
        )
        input("Press Enter to continue")

    # SORT MEASURES INTO SECTION IN/OUT OF GRAND STAFF
    sections = sort_page_into_sections(measures, grand_staff)

    if verbose:
        _print_info(
            "Detected sections",
            "in/out: number of measures",
            [f"{section_type}: {len(section)}" for section_type, section in sections]
        )

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
        _print_info(
            "Sections sorted by reading order",
            "in/out: number of measures in each row",
            [f"{section_type}: {', '.join([str(len(subsection)) for subsection in s_section])}"
             for section_type, s_section in sorted_sections]
        )

    if visualize:
        print("Showing sorted measures...")
        dumped_measures = list(itertools.chain.from_iterable([m for ms in sorted_sections for m in ms[1]]))
        write_numbers_on_image(image_path, dumped_measures)
        input("Press Enter to continue")

    # PREPARE MEASURES FOR EVENT DETECTION
    linked_measures: list[VirtualNode] = []

    for section_type, s_section in sorted_sections:
        # this is the true grand stave
        if section_type == SectionType.IN_GS and len(s_section) == 2:
            # tag staff that belong to it
            assign_gs_index_to_notes(s_section[0], 1)
            assign_gs_index_to_notes(s_section[1], 0)
            # and link individual measures together
            linked_measures.extend(link_measures_inside_grand_stave(s_section[0], s_section[1]))

        # this is a section of many single staves (or something undefined)
        else:
            for stave in s_section:
                for measure in stave:
                    # list of mesures makes it easier to adapt the following algorithms
                    linked_measures.append(VirtualNode([measure]))

    if verbose:
        print("Linked measures sorted by reading order")
        print(", ".join([str(len(e.children())) for e in linked_measures]))
        print()

    events: list[VirtualNode] = []
    events_by_measure: list[list[VirtualNode]] = []
    for mes in linked_measures:
        temp = compute_note_events(mes, iou_threshold=0.4)
        events.extend(temp)
        events_by_measure.append(temp)

    if visualize:
        flat_list: list[Node] = [item for sublist1 in events_by_measure for sublist2 in sublist1 for item in
                                 sublist2.children()]
        print("Showing note reading order...")
        write_numbers_on_image(image_path, flat_list)
        input("Press enter to continue")

        print("Showing end result...")
        visualize_result(image_path, measures, events, grand_staff)
        input("Press enter to continue")

    repre = " || ".join(
        [" | ".join(
            [" ".join([note_note_to_str(n) for n in event.children()])
             for event in measure])
            for measure in events_by_measure])

    if len(repre) != 0:
        repre = "|| " + repre + " ||"

    print(repre)

    return events_by_measure


def linearize_note_events(events_by_measure: list[list[VirtualNode]]) -> str:
    # measure sep
    repre = " || ".join(
        # event sep
        [" | ".join(
            # note sep
            [" ".join(
                [note_note_to_str(n) for n in event.children()]
            )
                for event in measure]
        )
            for measure in events_by_measure]
    )

    if len(repre) != 0:
        repre = "|| " + repre + " ||"

    return repre


reconstruct_note_events(
    measures,
    grand_staff,
    notehead_full + notehead_half,
    verbose=True,
    visualize=True
)
