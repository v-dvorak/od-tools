from enum import Enum
from pathlib import Path
from typing import Self, Any

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


class BaseNode:
    total_box: BoundingBox
    _children: list[Self]
    _tags: dict[str, Any]

    def __init__(self, tags: dict[str, Any] = None):
        self.total_bbox = None
        self._children = []
        self._tags: dict[str, Any] = tags if tags is not None else {}

    def add_child(self, child: Self):
        self._children.append(child)

    def children(self) -> list[Self]:
        return self._children

    def get_tag(self, key: str) -> Any:
        return self._tags.get(key, None)

    def set_tag(self, key: str, value: Any):
        self._tags[key] = value

    def update_total_bbox(self):
        raise NotImplementedError()


class Node(BaseNode):
    """
    Object with a bounding box in a scene.
    """
    annot: Annotation

    def __init__(self, base: Annotation, tags: dict[str, Any] = None):
        super().__init__(tags)
        self.annot = base
        self.total_bbox = base.bbox

    def add_child(self, child: Self, update_t_bbox: bool = False):
        self._children.append(child)
        if update_t_bbox:
            self.update_total_bbox()

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


class VirtualNode(BaseNode):
    """
    Virtual object without a bounding box in a scene.
    """
    _children: list[Node]

    def __init__(self, children: list[Node], tags: dict[str, Any] = None):
        super().__init__(tags)
        self._children: list[Node] = children
        self.update_total_bbox()

    def update_total_bbox(self):
        if len(self._children) > 0:
            self.total_bbox = BoundingBox(
                min(self._children, key=lambda b: b.annot.bbox.left).annot.bbox.left,
                min(self._children, key=lambda b: b.annot.bbox.top).annot.bbox.top,
                max(self._children, key=lambda b: b.annot.bbox.right).annot.bbox.right,
                max(self._children, key=lambda b: b.annot.bbox.bottom).annot.bbox.bottom
            )

    def children(self) -> list[Node]:
        return self._children


def bbox_center_distance(bbox1: BoundingBox, bbox2: BoundingBox, direction: Direction = None) -> float:
    c_v1, c_h1 = bbox1.center()
    c_v2, c_h2 = bbox2.center()
    if direction is None:
        raise NotImplementedError("TODO: euclidian distance")
    elif direction == Direction.HORIZONTAL:
        return abs((bbox1.left + bbox1.width / 2) - (bbox2.left + bbox2.width / 2))
    elif direction == Direction.VERTICAL:
        return abs((bbox1.top + bbox1.height / 2) - (bbox2.top + bbox2.height / 2))


def assign_to_closest(target: list[Node], source: list[Node], upper_limit: float = None):
    """
    Assigns object from source to targets based on their distance from them.
    Modifies the target list in place.

    :param target: list of targets to assign sources to
    :param source: list of sources to assign to targets
    :param upper_limit: maximum distance to assign source to target
    """
    for current_source in source:
        best_distance = np.inf
        best_target: Node = None

        for current_target in target:
            if current_source.annot.bbox.is_fully_inside(current_target.annot.bbox, direction=Direction.HORIZONTAL):
                current_distance = bbox_center_distance(
                    current_target.annot.bbox,
                    current_source.annot.bbox,
                    direction=Direction.VERTICAL
                )
                if (upper_limit is None or current_distance < upper_limit) and current_distance < best_distance:
                    best_distance = current_distance
                    best_target = current_target

        if best_target is None:
            print(
                f"Warning: No suitable target found for source: {current_source.annot.bbox}, id {current_source.annot.class_id}")
        else:
            best_target.add_child(current_source)

    for current_target in target:
        current_target.update_total_bbox()


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

# ASSIGN NOTES TO MEASURES
# the method assigns notes to measures based on center points of detection
#   * (3 / 2) means +- one whole average measure from top and bottom from chosen measure
upper_assignment_limit = np.mean([m.annot.bbox.height for m in measures]) * (3 / 2)
assign_to_closest(measures, notehead_full, upper_limit=upper_assignment_limit)
assign_to_closest(measures, notehead_half, upper_limit=upper_assignment_limit)

if visualize:
    temp = draw_rectangles_on_image(
        image_path,
        [m.total_bbox for m in measures],
        show=True
    )


def assign_height_to_notes_inside_measure(measure: Node):
    half_line_height = measure.annot.bbox.height / 8
    for note in measure.children():
        x_center, _ = note.annot.bbox.center()
        distance_from_zero = measure.annot.bbox.bottom - x_center
        note.set_tag("height", distance_from_zero / half_line_height)


# ASSIGN HEIGHT TO EACH NOTE
for measure in measures:
    assign_height_to_notes_inside_measure(measure)


class SectionType(Enum):
    IN_GS = 0
    OUT_GS = 1


def sort_measures_by_grand_staff(measures: list[Node], grand_staff: list[Node]):
    grand_staff = sorted(grand_staff, key=lambda g: g.annot.bbox.top)
    measures = sorted(measures, key=lambda m: m.annot.bbox.top)
    sorted_by_gs: list[tuple[SectionType, list[Node]]] = []

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
                sorted_by_gs.append((SectionType.OUT_GS, section))
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
            sorted_by_gs.append((SectionType.IN_GS, section))
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
    sorted_by_gs.append((SectionType.IN_GS if in_gs else SectionType.OUT_GS, section))
    return sorted_by_gs


# SORT MEASURES INTO SECTION IN/OUT OF GRAND STAFF
sorted_by_gs = sort_measures_by_grand_staff(measures, grand_staff)
for section_type, measures in sorted_by_gs:
    print(f"{section_type}: {len(measures)}", end=", ")


def sort_to_strips_with_threshold(
        nodes: list[Node],
        iou_threshold: float,
        direction: Direction = Direction.HORIZONTAL,
        check_intersections: bool = False
) -> list[list[Node]]:
    """
    Sort objects according to the given threshold into strips (rows/columns).
    HORIZONTAL corresponds to the reading order: left to right, top to bottom.
    VERTICAL corresponds to the reading order: bottom to top, left to right.

    Sorts always from lowest to highest based on the top or left coordinate.

    :param nodes: list of objects to sort
    :param iou_threshold: threshold for sorting, how big there could be between two objects in the same strip
    :param direction: the direction of sorting
    :return: list of sorted strips
    """

    def _intersects_any(row: list[Node], node: Node):
        for n in row:
            if node.annot.bbox.intersects(n.annot.bbox):
                return True
        return False

    if len(nodes) == 0:
        return []

    if direction == Direction.HORIZONTAL:
        top_sorted = sorted(nodes, key=lambda n: n.annot.bbox.top)
        sorted_rows: list[list[Node]] = []
        row: list[Node] = []
        for node in top_sorted:
            if len(row) == 0:
                row.append(node)
                continue

            computed_iou = node.annot.bbox.intersection_over_union(
                row[-1].annot.bbox,
                direction=Direction.VERTICAL
            )
            inter_any = (check_intersections and _intersects_any(row, node))

            if computed_iou > iou_threshold or inter_any:
                row.append(node)
            else:
                sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.left))
                row = [node]

        sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.left))

        return sorted_rows

    elif direction == Direction.VERTICAL:
        left_sorted = sorted(nodes, key=lambda n: n.annot.bbox.left)
        sorted_rows: list[list[Node]] = []
        row: list[Node] = []
        for node in left_sorted:
            if len(row) == 0:
                row.append(node)
                continue

            computed_iou = node.annot.bbox.intersection_over_union(
                row[-1].annot.bbox,
                direction=Direction.HORIZONTAL
            )
            inter_any = (check_intersections and _intersects_any(row, node))

            if computed_iou > iou_threshold or inter_any:
                row.append(node)
            else:
                sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.top, reverse=True))
                row = [node]

        sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.top, reverse=True))

        return sorted_rows

    else:
        raise NotImplementedError(f"Not implemented for direction {direction}")


# SORT SECTIONS INTO ROWS OF MEASURES
measure_row_threshold = 0.5

sorted_sections: list[tuple[SectionType, list[list[Node]]]] = []
for section_type, data in sorted_by_gs:
    sorted_sections.append(
        (section_type, sort_to_strips_with_threshold(data, measure_row_threshold, direction=Direction.HORIZONTAL)))

print(len(sorted_sections))
for section_type, section in sorted_sections:
    print(f"{section_type}: {len(section)}")
    for subsection in section:
        print(len(subsection), end=", ")
    print()

from PIL import ImageDraw, ImageFont


def write_numbers_on_image(image_path, measures: list[Node]):
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


def link_measures_inside_grand_stave(
        top_row: list[Node],
        bottom_row: list[Node],
        linkage_iou_threshold: float = 0.5
) -> list[VirtualNode]:
    """
    Takes measures in top and bottom stave of a single grand stave
    and returns a list of linked pairs connected by VirtualNode.
    If a measure does not link with any other measure, it is returned as a single child of VirtualNode.

    Linkage is made if the computed IoU (intersection over union of pairs horizontal coordinates)
    is less than linkage_iou_threshold.

    :param top_row: list of nodes representing the top stave
    :param bottom_row: list of nodes representing the bottom stave
    :param linkage_iou_threshold: threshold for linkage to be made
    """
    top_index = 0
    bottom_index = 0
    print(len(top_row), len(bottom_row))

    linked_measures: list[VirtualNode] = []

    # going from left to right
    while top_index < len(top_row) and bottom_index < len(bottom_row):
        top_measure = top_row[top_index]
        bottom_measure = bottom_row[bottom_index]

        iou = top_measure.annot.bbox.intersection_over_union(bottom_measure.annot.bbox, direction=Direction.HORIZONTAL)
        print(f"iou: {iou}")

        # linkage found
        if iou > linkage_iou_threshold:
            linked_measures.append(VirtualNode([top_measure, bottom_measure]))
            top_index += 1
            bottom_index += 1

        # throw out one measure and advance in its row
        else:
            # drop the leftmost measure
            if top_measure.annot.bbox.left < bottom_measure.annot.bbox.left:
                linked_measures.append(VirtualNode([top_measure]))
                top_index += 1
            else:
                linked_measures.append(VirtualNode([bottom_measure]))
                bottom_index += 1

    if top_index == len(top_row) and bottom_index == len(bottom_row):
        return linked_measures

    # dump the rest of bottom row
    elif top_index == len(top_row):
        while bottom_index < len(bottom_row):
            linked_measures.append(VirtualNode([bottom_row[bottom_index]]))
            bottom_index += 1
    # dump the rest of top row
    else:
        while top_index < len(top_row):
            linked_measures.append(VirtualNode([top_row[top_index]]))
            top_index += 1

    return linked_measures


def tag_notes_with_gs_index(measures: list[Node], gs_index: int):
    for measure in measures:
        for note in measure.children():
            note.set_tag("gs_index", gs_index)


# PREPARE MEASURES FOR EVENT DETECTION
events_in_measures: list[VirtualNode] = []

for section_type, section in sorted_sections:
    # this is true grand stave
    if section_type == SectionType.IN_GS and len(section) == 2:
        tag_notes_with_gs_index(section[0], 1)
        tag_notes_with_gs_index(section[1], 0)
        events_in_measures.extend(link_measures_inside_grand_stave(section[0], section[1]))
    # this is a false grand stave of a section of many single staves (or something undefined)
    else:
        for stave in section:
            for measure in stave:
                # list of mesures make it easier to adapt following algorithms
                # for multiple measures playing at once
                events_in_measures.append(VirtualNode([measure]))

for e in events_in_measures:
    print(len(e.children()), end=", ")
print()


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


events: list[VirtualNode] = []
events_by_measure: list[list[VirtualNode]] = []
for mes in events_in_measures:
    temp = compute_note_events(mes, iou_threshold=0.4)
    events.extend(temp)
    events_by_measure.append(temp)

write_numbers_on_image(image_path, [n for e in events for n in e.children()])

temp = draw_rectangles_on_image(
    image_path,
    [gs.annot.bbox for gs in grand_staff],
    color=(0, 255, 0),
    thickness=2,
)

temp = draw_rectangles_on_image(
    temp,
    [n.annot.bbox for m in sorted_by_gs for n in m[1]],
    color=(0, 0, 255),
    thickness=2,
)

temp = draw_rectangles_on_image(
    temp,
    [e.total_bbox for e in events],
    color=(255, 0, 0),
    thickness=2,
    show=True
)


def write_note_heights_to_image(image, measures: list[Node]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        font = ImageFont.load_default()

    for note in measures:
        draw.text(
            (note.annot.bbox.left, note.annot.bbox.top),
            str(round(note.get_tag("height"))),
            font=font,
            fill=(0, 255, 0)
        )

    image.show()


write_note_heights_to_image(temp, [n for c in events for n in c.children()])


def note_node_to_str(note: Node) -> str:
    if note.get_tag("gs_index") is None:
        return str(round(note.get_tag("height")))
    else:
        return f"{note.get_tag('gs_index')}:{round(note.get_tag('height'))}"


repre = " || ".join(
    [" | ".join(
        [" ".join([note_node_to_str(n) for n in event.children()])
         for event in measure])
        for measure in events_by_measure])

if len(repre) != 0:
    repre = "|| " + repre + " ||"

print(repre)
