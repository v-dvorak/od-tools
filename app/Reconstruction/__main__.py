from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

from .Node import Node, VirtualNode
from .NoteManipulation import assign_notes_to_measures_and_compute_pitch
from .PageReconstruction import compute_note_events_for_page, linearize_note_events
from .VizUtils import visualize_result
from ..Conversions.Annotations.FullPage import FullPage
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

if True:
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

from .PageReconstruction import link_measures_based_on_grand_staffs


def reconstruct_note_events(
        measures: list[Node],
        grand_staffs: list[Node],
        notes: list[Node],
        ual_factor: float = 1.5,
        mriou_threshold: float = 0.5,
        neiou_threshold: float = 0.4,
        verbose: bool = False,
        visualize: bool = False
) -> list[list[VirtualNode]]:
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

from timeit import default_timer as timer
start = timer()
reconstruct_note_events(
    measures,
    grand_staff,
    notehead_full + notehead_half,
)
print(timer() - start)
