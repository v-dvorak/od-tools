from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.Conversions.Annotations.Annotation import Annotation
from app.Conversions.Annotations.FullPage import FullPage
from app.Conversions.BoundingBox import BoundingBox, Direction
from app.Conversions.ConversionUtils import get_num_pixels
from app.Splitting.SplitUtils import create_split_box_matrix, draw_rectangles_on_image
from app.Val.Utils import create_split_images

image_path = "demo/30d6c780-c8fe-11e7-9c14-005056827e51_375c6850-f593-11e7-b30f-5ef3fc9ae867.jpg"
image_path = "sheet-test.jpg"
# not ideal
image_path = "F:/Studium/4/rocnikac/od-tools/datasets/2024-10-07_proto-dataset/images/81c9f683-28d1-4e73-8e25-e37333408f5a_ac45624e-0846-4c6d-a079-a1f1877e1aea.jpg"
# ideal
image_path = "F:/Studium/4/rocnikac/od-tools/datasets/2024-10-07_proto-dataset/images/bf061840-2322-11eb-979b-005056827e52_2e117f2e-4c19-4bc3-ba6b-5531ca623e22.jpg"
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

from enum import Enum


class NoteType(Enum):
    HALF = 0
    FULL = 1


class Note:
    annot: Annotation
    height: float = None
    gs_position: int = -1
    note_type: NoteType = None

    def __init__(self, note: Annotation, note_type: NoteType):
        self.annot = note
        self.note_type = note_type

    def set_height(self, height: float):
        self.height = height

    def get_height(self):
        return self.height

    def __str__(self) -> str:
        if self.gs_position == -1:
            return str(round(self.height))
        else:
            return f"{self.gs_position}:{round(self.height)}"


class Measure:
    annot: Annotation
    total_box: BoundingBox
    notes: list[Note]

    def __init__(self, measure: Annotation):
        self.annot = measure
        self.total_bbox = measure.bbox
        self.notes = []

    def update_total_bbox(self):
        if len(self.notes) > 0:
            temp_box = BoundingBox(
                min(self.notes, key=lambda b: b.annot.bbox.left).annot.bbox.left,
                min(self.notes, key=lambda b: b.annot.bbox.top).annot.bbox.top,
                max(self.notes, key=lambda b: b.annot.bbox.right).annot.bbox.right,
                max(self.notes, key=lambda b: b.annot.bbox.bottom).annot.bbox.bottom
            )
            self.total_bbox = BoundingBox(
                temp_box.left if temp_box.left < self.total_bbox.left else self.total_bbox.left,
                temp_box.top if temp_box.top < self.total_bbox.top else self.total_bbox.top,
                temp_box.right if temp_box.right > self.total_bbox.right else self.total_bbox.right,
                temp_box.bottom if temp_box.bottom > self.total_bbox.bottom else self.total_bbox.bottom
            )

from typing import Self

class Node:
    annot: Annotation
    total_box: BoundingBox
    _children: list[Self]

    def __init__(self, base: Annotation):
        self.annot = base
        self.total_bbox = base.bbox
        self._children = []

    def add_child(self, child: Self):
        self._children.append(child)

    def children(self):
        return self._children

    def update_total_bbox(self):
        if len(self._children) > 0:
            temp_box = BoundingBox(
                min(self._children, key=lambda b: b.annot.bbox.left).annot.bbox.left,
                min(self._children, key=lambda b: b.annot.bbox.top).annot.bbox.top,
                max(self._children, key=lambda b: b.annot.bbox.right).annot.bbox.right,
                max(self._children, key=lambda b: b.annot.bbox.bottom).annot.bbox.bottom
            )
            self.total_bbox = BoundingBox(
                temp_box.left if temp_box.left < self.total_bbox.left else self.total_bbox.left,
                temp_box.top if temp_box.top < self.total_bbox.top else self.total_bbox.top,
                temp_box.right if temp_box.right > self.total_bbox.right else self.total_bbox.right,
                temp_box.bottom if temp_box.bottom > self.total_bbox.bottom else self.total_bbox.bottom
            )


def bbox_center_distance(bbox1: BoundingBox, bbox2: BoundingBox, direction: Direction = None) -> float:
    if direction is None:
        raise NotImplementedError("TODO: euclidian distance")
    elif direction == Direction.HORIZONTAL:
        return abs((bbox1.left + bbox1.width / 2) - (bbox2.left + bbox2.width / 2))
    elif direction == Direction.VERTICAL:
        return abs((bbox1.top + bbox1.height / 2) - (bbox2.top + bbox2.height / 2))


def assign_to_closest(target: list[Measure], source: list[Note]):
    """
    Assigns object from source to targets based on their distance from them.
    Modifies the target list in place.
    """
    for note in source:
        best_distance = np.inf
        best_measure: Measure = None

        for measure in target:
            if note.annot.bbox.is_fully_inside(measure.annot.bbox, direction=Direction.HORIZONTAL):
                current_distance = bbox_center_distance(measure.annot.bbox, note.annot.bbox,
                                                        direction=Direction.VERTICAL)
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_measure = measure

        if best_measure is None:
            print(f"Warning: No suitable target found for source: {note.annot.bbox}, id {note.annot.class_id}")
        else:
            best_measure.notes.append(note)

    for measure in target:
        measure.update_total_bbox()


class GrandStaff:
    annot: Annotation

    def __init__(self, grand_staff: Annotation):
        self.annot = grand_staff


# INITIALIZE GRAPH
notehead_full = []
for note in combined.annotations[5]:
    notehead_full.append(Note(note, NoteType.FULL))

notehead_half = []
for note in combined.annotations[6]:
    notehead_half.append(Note(note, NoteType.HALF))

measures = []
for measure in combined.annotations[1]:
    measures.append(Measure(measure))

grand_staff = []
for gs in combined.annotations[4]:
    grand_staff.append(GrandStaff(gs))

# ASSIGN NOTES TO MEASURES
assign_to_closest(measures, notehead_full)
assign_to_closest(measures, notehead_half)

if visualize:
    temp = draw_rectangles_on_image(
        image_path,
        [m.total_bbox for m in measures],
        show=True
    )


def assign_height_to_notes_inside_measure(measure: Measure):
    half_line_height = measure.annot.bbox.height / 8
    for note in measure.notes:
        x_center, _ = note.annot.bbox.center()
        distance_from_zero = measure.annot.bbox.bottom - x_center
        note.height = distance_from_zero / half_line_height


# ASSIGN HEIGHT TO EACH NOTE
for measure in measures:
    assign_height_to_notes_inside_measure(measure)


def sort_measures_by_grand_staff(measures: list[Measure], grand_staff: list[GrandStaff]):
    grand_staff = sorted(grand_staff, key=lambda g: g.annot.bbox.top)
    measures = sorted(measures, key=lambda m: m.annot.bbox.top)
    sorted_by_gs: list[tuple[str, list[Measure]]] = []

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
                sorted_by_gs.append(("out", section))
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
            sorted_by_gs.append(("in", section))
            section = [measure]

            # it can intersect with the next gs
            if gs_index + 1 < len(grand_staff) and measure.annot.bbox.intersects(
                    grand_staff[gs_index + 1].annot.bbox):
                in_gs = True
            else:
                in_gs = False

            # switch to next gs
            gs_index += 1
        else:
            raise NotImplementedError("wut")

    # append last section
    sorted_by_gs.append(("in" if in_gs else "out", section))
    return sorted_by_gs


# SORT MEASURES INTO SECTION IN/OUT OF GRAND STAFF
sorted_by_gs = sort_measures_by_grand_staff(measures, grand_staff)
for state, measures in sorted_by_gs:
    print(f"{state}: {len(measures)}", end=", ")


def sort_to_strips_with_threshold(nodes: list[Measure], threshold: float,
                                  direction: Direction = Direction.HORIZONTAL) -> list[list[Measure]]:
    """
    Sort objects according to the given threshold into strips (rows/columns).
    HORIZONTAL corresponds to the reading order: left to right, top to bottom.
    VERTICAL corresponds to the reading order: bottom to top, left to right.

    Sorts always from lowest to highest based on the top or left coordinate.

    :param nodes: list of objects to sort
    :param threshold: threshold for sorting, how big there could be between two objects in the same strip
    :param direction: the direction of sorting
    :return: list of sorted strips
    """
    if direction == Direction.HORIZONTAL:
        top_sorted = sorted(nodes, key=lambda n: n.annot.bbox.top)
        sorted_rows = []
        row: list[Measure] = []
        for node in top_sorted:
            if len(row) == 0:
                row.append(node)
            elif abs(node.annot.bbox.top - row[-1].annot.bbox.top) < threshold:
                row.append(node)
            else:
                sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.left))
                row = [node]

        sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.left))

        return sorted_rows

    elif direction == Direction.VERTICAL:
        left_sorted = sorted(nodes, key=lambda n: n.annot.bbox.left)
        sorted_rows: list[list[Measure]] = []
        row: list[Measure] = []
        for node in left_sorted:
            if len(row) == 0:
                row.append(node)
            elif abs(node.annot.bbox.left - row[-1].annot.bbox.left) < threshold:
                row.append(node)
            else:
                sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.top, reverse=True))
                row = [node]

        sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.top, reverse=True))

        return sorted_rows

    else:
        raise NotImplementedError(f"Not implemented for direction {direction}")


# SORT SECTIONS INTO ROWS OF MEASURES
measure_row_threshold = np.mean([m.annot.bbox.bottom - m.annot.bbox.top for m in measures]) * (2 / 5)

sorted_sections: list[tuple[str, list[list[Measure]]]] = []
for state, data in sorted_by_gs:
    sorted_sections.append((state, sort_to_strips_with_threshold(data, measure_row_threshold)))

print(len(sorted_sections))
for state, section in sorted_sections:
    print(f"{state}: {len(section)}")
    for subsection in section:
        print(len(subsection), end=", ")
    print()

from PIL import ImageDraw, ImageFont


def write_numbers_on_image(image_path, measures: list[Measure]):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        font = ImageFont.load_default()

    for i, measure in enumerate(measures, start=1):
        draw.text((measure.annot.bbox.left, measure.annot.bbox.top), str(i), font=font, fill=(255, 0, 0))

    image.show()


import itertools

dumped_measures = list(itertools.chain.from_iterable([m for ms in sorted_sections for m in ms[1]]))
if visualize:
    write_numbers_on_image(image_path, dumped_measures)


def horizontal_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    overlap_start = max(bbox1.left, bbox2.left)
    overlap_end = min(bbox1.right, bbox2.right)

    union_start = min(bbox1.left, bbox2.left)
    union_end = max(bbox1.right, bbox2.right)

    overlap = overlap_end - overlap_start
    union = union_end - union_start
    return overlap / union


def link_measures_inside_grand_stave(
        top_row: list[Measure],
        bottom_row: list[Measure],
        linkage_threshold: float = 0.9
) -> list[list[Measure]]:
    top_index = 0
    bottom_index = 0

    linked_measures: list[list[Measure]] = []

    # going from left to right
    while top_index < len(top_row) and bottom_index < len(bottom_row):
        top_measure = top_row[top_index]
        bottom_measure = bottom_row[bottom_index]

        iou = horizontal_iou(top_measure.annot.bbox, bottom_measure.annot.bbox)
        print(f"iou: {iou}")
        # linkage found
        if iou > linkage_threshold:
            linked_measures.append([top_measure, bottom_measure])
            top_index += 1
            bottom_index += 1
        # throw out one measure and advance in its row
        else:
            # drop the leftmost measure
            if top_measure.annot.bbox.left < bottom_measure.annot.bbox.left:
                linked_measures.append([top_measure])
                top_index += 1
            else:
                linked_measures.append([bottom_measure])
                bottom_index += 1

    if top_index == len(top_row) and bottom_index == len(bottom_row):
        return linked_measures

    # dump the rest of bottom row
    elif top_index == len(top_row):
        while bottom_index < len(bottom_row):
            linked_measures.append([bottom_row[bottom_index]])
    # dump the rest of top row
    else:
        while top_index < len(top_row):
            linked_measures.append([top_row[top_index]])

    return linked_measures


def tag_notes_with_gs_index(measures: list[Measure], gs_index: int):
    for measure in measures:
        for note in measure.notes:
            note.gs_position = gs_index


# PREPARE MEASURES FOR EVENT DETECTION
events_in_measures: list[list[Measure]] = []

for state, section in sorted_sections:
    # this is true grand stave
    if state == "in" and len(section) == 2:
        tag_notes_with_gs_index(section[0], 1)
        tag_notes_with_gs_index(section[1], 0)
        events_in_measures.extend(link_measures_inside_grand_stave(section[0], section[1]))
    # this is a false grand stave of a section of many single staves (or something undefined)
    else:
        for stave in section:
            for measure in stave:
                # list of mesures make it easier to adapt following algorithms
                # for multiple measures playing at once
                events_in_measures.append([measure])


def compute_note_events(measures: list[Measure]) -> list[list[Note]]:
    all_notes: list[Note] = [n for m in measures for n in m.notes]
    note_event_threshold = np.mean([n.annot.bbox.width for n in all_notes]) * (1 / 5)
    return sort_to_strips_with_threshold(all_notes, note_event_threshold, direction=Direction.VERTICAL)


events: list[list[Note]] = []
events_by_measure: list[list[list[Note]]] = []
for mes in events_in_measures:
    temp = compute_note_events(mes)
    events.extend(temp)
    events_by_measure.append(temp)

event_bbox: list[Node] = []
for event in events:
    node = Node(event[0].annot)
    node._children = event
    node.update_total_bbox()
    event_bbox.append(node)

write_numbers_on_image(image_path, [n for e in events for n in e])

temp = draw_rectangles_on_image(
    image_path,
    [n.annot.bbox for m in sorted_by_gs for n in m[1]],
    color=(0, 0, 255),
    thickness=2,
)

temp = draw_rectangles_on_image(
    temp,
    [n.total_bbox for n in event_bbox],
    color=(255, 0, 0),
    thickness=2,
    show=True
)


def write_note_heights_to_image(image, measures: list[Note]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        font = ImageFont.load_default()

    for note in measures:
        draw.text((note.annot.bbox.left, note.annot.bbox.top), str(round(note.height)), font=font, fill=(0, 255, 0))

    image.show()


write_note_heights_to_image(temp, [n for c in events for n in c])

repre = " || ".join(
    [" | ".join(
        [" ".join([n.__str__() for n in event])
         for event in measure])
        for measure in events_by_measure])

if len(repre) != 0:
    repre = "|| " + repre + " ||"

print(repre)
