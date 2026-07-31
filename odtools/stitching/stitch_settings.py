from dataclasses import dataclass, field
import numpy as np

from ..Conversions import FullPage, BoundingBox, Annotation
from ..Inference import SplitSettings

DEFAULT_NMS_IOU_THRESHOLD: float = 0.5


@dataclass
class StitchSettings:
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD
    special_cases: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in self.special_cases.values():
            assert 0 <= value <= 1

    def get_nms_iou_threshold_for_class(self, class_id: int) -> float:
        threshold = self.special_cases.get(class_id)

        if threshold is None:
            return self.nms_iou_threshold
        else:
            return threshold

    @classmethod
    def from_json(cls, data: dict) -> "StitchSettings":
        iou = data.get("nms_iou")
        if iou is None:
            return cls(DEFAULT_NMS_IOU_THRESHOLD, {})
        else:
            return cls(iou, {})


def _apply_nms_single_class(
    annotations: list[Annotation], iou_threshold: float
) -> list[Annotation]:
    annotations = sorted(annotations, key=lambda a: a.confidence, reverse=True)
    kept = []

    while annotations:
        current = annotations.pop(0)
        kept.append(current)

        annotations = [
            a for a in annotations if current.intersection_over_union(a) < iou_threshold
        ]

    return kept

def _apply_nms_cross_part_single_class(
    annotations_a: list[Annotation],
    annotations_b: list[Annotation],
    iou_threshold: float,
) -> tuple[list[Annotation], list[Annotation]]:
    """
    NMS across two parts where annotations only interact between parts,
    not within the same part.

    Returns filtered (annotations_a, annotations_b).
    """
    A = 0
    B = 1
    # tag each annotation with its source part
    tagged = (
        [(a, A) for a in annotations_a] +
        [(b, B) for b in annotations_b]
    )
    tagged = sorted(tagged, key=lambda x: x[0].confidence, reverse=True)

    kept_a, kept_b = [], []

    while tagged:
        current, current_part = tagged.pop(0)

        if current_part == A:
            kept_a.append(current)
        else:
            kept_b.append(current)

        tagged = [
            (a, part)
            for a, part in tagged
            if part == current_part  # same part -> never suppress
            or current.intersection_over_union(a) < iou_threshold
        ]

    return kept_a, kept_b


def apply_nms_two_page(
        page_a: FullPage,
        page_b: FullPage,
        iou_threshold: float
) -> None:
    """
    Modifies pages in place.
    """
    assert len(page_a.annotations) == len(page_b.annotations)

    for class_id in range(len(page_a.annotations)):
        an_a = page_a.annotations[class_id]
        an_b = page_b.annotations[class_id]

        an_a, an_b = _apply_nms_cross_part_single_class(
            an_a,
            an_b,
            iou_threshold
        )

        page_a.annotations[class_id] = an_a
        page_b.annotations[class_id] = an_b



def combine_multiple_pages_and_resolve(
    subpages: list[FullPage],
    splits: list[list[BoundingBox]],
    split_settings: SplitSettings,
    stitch_settings: StitchSettings,
    verbose: bool = False,
) -> "FullPage":
    """
    Combines multiple pages into a single page.

    :param subpages: list of subpages
    :param splits: matrix of splits
    :param iou_threshold: how big IoU has to be to trigger resolving
    :param edge_offset: offset from edges, anything outside the edge will be dropped from final page
    :param verbose: make script verbose
    :return: FullPage
    """

    for i, (subpage, split) in enumerate(
        zip(subpages, [x for xs in splits for x in xs])
    ):
        # subpage: FullPage
        # split: BoundingBox

        # cut predictions on edges
        if split_settings.edge_offset > 0:
            x, y = i % len(splits[0]), i // len(splits[0])
            subpage.cut_off_predictions_too_close_to_edge(
                edge_offset=split_settings.edge_offset,
                edge_tile=(
                    x != 0,
                    y != 0,
                    x != len(splits[0]) - 1,
                    y != len(splits) - 1,
                ),
                verbose=verbose,
            )
        # shift annotations based in their absolute position in image
        subpage.adjust_position_for_all_annotations(split.left, split.top)

    # retrieve important values without actually passing them as arguments
    # class names from first class
    class_names = subpages[0].class_names
    # the last split is also de facto the bottom right corner of the image,
    # we can retrieve image image_size from here,
    last_split: BoundingBox = splits[-1][-1]

    # matrix = list(np.reshape(subpages, (len(splits), len(splits[0]))))
    # print(splits)
    # exit()

    # resolve_matrix_of_pages(matrix, stitch_settings.nms_iou_threshold)

    completed_annotations = [[] for _ in range(len(class_names))]
    for subpage in subpages:
        for annotation in subpage.all_annotations():
            completed_annotations[annotation.class_id].append(annotation)

    complete_page = FullPage(
        (last_split.right, last_split.bottom), completed_annotations, class_names
    )

    complete_page.annotations = [
        _apply_nms_single_class(annots, stitch_settings.nms_iou_threshold)
        for annots in complete_page.annotations
    ]

    return complete_page


def resolve_matrix_of_pages(
    subpage_matrix: list[list[FullPage]],
    iou_threshold: float = 0.25,
) -> None:

    # vectors = [(1, 0), (1, 1), (0, 1)]
    vectors = [(1, 0), (2, 0),
       (0, 1), (1, 1), (2, 1),
       (0, 2), (1, 2), (2, 2)]
    for row in range(len(subpage_matrix)):
        for col in range(len(subpage_matrix[0])):
            for dx, dy in vectors:
                if row + dx < len(subpage_matrix) and col + dy < len(subpage_matrix[0]):
                    print(f"processing {(row, col)} vs {(row + dx, col + dy)}")
                    print(subpage_matrix[row][col].annotation_count() +
                        subpage_matrix[row + dx][col + dy].annotation_count())
                    apply_nms_two_page(
                        subpage_matrix[row][col],
                        subpage_matrix[row + dx][col + dy],
                        iou_threshold
                    )
                    print(subpage_matrix[row][col].annotation_count() +
                        subpage_matrix[row + dx][col + dy].annotation_count())
                    
