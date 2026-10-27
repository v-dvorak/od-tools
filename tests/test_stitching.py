from odtools.conversions.annotations import FullPage
from odtools.conversions.bounding_box import BoundingBox
from odtools.inference import SplitSettings
from odtools.stitching import StitchSettings, stitch_settings


def _annot(class_id, left, top, width, height, confidence=1.0):
    from odtools.conversions.annotations import Annotation

    return Annotation(class_id, left, top, width, height, None, confidence=confidence)


def _subpage(annotations, class_names, size=(60, 60)) -> FullPage:
    rows = [[] for _ in class_names]
    for a in annotations:
        rows[a.class_id].append(a)
    return FullPage(size, rows, class_names)


class TestNMS:
    def test_duplicates_are_suppressed_keeping_best_confidence(self):
        annotations = [
            _annot(0, 0, 0, 10, 10, confidence=0.9),
            _annot(0, 1, 1, 10, 10, confidence=0.5),
            _annot(0, 2, 2, 10, 10, confidence=0.7),
            _annot(0, 40, 40, 10, 10, confidence=0.1),  # distinct, must survive
        ]
        kept = stitch_settings._apply_nms_single_class(annotations, iou_threshold=0.25)

        assert len(kept) == 2
        kept_confidences = sorted(a.confidence for a in kept)
        assert kept_confidences == [0.1, 0.9]


class TestCombineMultiplePagesAndResolve:
    def _setup(self):
        # 1x2 grid of overlapping 60px-tall tiles covering a 100x60 image;
        # the tiles overlap in the 50..60 column, so an object straddling
        # x=55 is seen by both
        splits = [[BoundingBox(0, 0, 60, 60), BoundingBox(50, 0, 100, 60)]]
        class_names = ["class_a"]

        subpages = [
            _subpage(
                [
                    # unique object, only visible to the left tile
                    _annot(0, 10, 10, 10, 10, confidence=0.9),
                    # seam object as seen by the left tile
                    _annot(0, 45, 20, 15, 20, confidence=0.8),
                ],
                class_names,
            ),
            _subpage(
                [
                    # the same seam object as seen by the right tile
                    _annot(0, 0, 21, 15, 20, confidence=0.6),
                ],
                class_names,
            ),
        ]
        return subpages, splits

    def test_duplicates_across_seam_are_resolved(self):
        subpages, splits = self._setup()

        resolved = stitch_settings.combine_multiple_pages_and_resolve(
            subpages,
            splits,
            SplitSettings(
                width=60, height=60, overlap_ratio=0.1, edge_offset_ratio=0.0
            ),
            StitchSettings(nms_iou_threshold=0.25),
        )

        # seam duplicate merged away, unique object kept
        assert resolved.annotation_count() == 2
        # coordinates are shifted back to full-image space
        lefts = sorted(a.bbox.left for a in resolved.all_annotations())
        assert lefts == [10, 45]

    def test_final_page_size_from_last_split(self):
        subpages, splits = self._setup()

        resolved = stitch_settings.combine_multiple_pages_and_resolve(
            subpages,
            splits,
            SplitSettings(
                width=60, height=60, overlap_ratio=0.1, edge_offset_ratio=0.0
            ),
            StitchSettings(nms_iou_threshold=0.25),
        )

        assert resolved.size == (100, 60)

    def test_edge_predictions_are_cut(self):
        # two adjacent tiles; the right edge of the left tile is an
        # interior edge, predictions hugging it are dropped
        splits = [[BoundingBox(0, 0, 50, 50), BoundingBox(50, 0, 100, 50)]]
        class_names = ["class_a"]
        subpages = [
            _subpage(
                [
                    # hugging the interior (right) edge of the left tile -> dropped
                    _annot(0, 41, 10, 8, 10, confidence=0.9),
                    # comfortably inside -> kept
                    _annot(0, 10, 20, 10, 10, confidence=0.9),
                ],
                class_names,
                size=(50, 50),
            ),
            _subpage([], class_names, size=(50, 50)),
        ]

        resolved = stitch_settings.combine_multiple_pages_and_resolve(
            subpages,
            splits,
            SplitSettings(
                width=50, height=50, overlap_ratio=0.1, edge_offset_ratio=0.1
            ),
            StitchSettings(apply_nms=False),
        )

        assert resolved.annotation_count() == 1
        kept = next(iter(resolved.all_annotations()))
        assert kept.bbox.top == 20
        assert kept.bbox.left == 10

    def test_per_class_nms_thresholds_are_respected(self):
        # class 0 has a per-class threshold of 0.95 -> near-duplicates survive;
        # class 1 uses the default 0.25 -> duplicates are merged
        splits = [[BoundingBox(0, 0, 50, 50)]]
        class_names = ["loose", "strict"]
        subpages = [
            _subpage(
                [
                    _annot(0, 5, 5, 10, 10, confidence=0.9),
                    _annot(0, 7, 7, 10, 10, confidence=0.8),
                    _annot(1, 30, 5, 10, 10, confidence=0.9),
                    _annot(1, 32, 7, 10, 10, confidence=0.8),
                ],
                class_names,
                size=(50, 50),
            ),
        ]

        resolved = stitch_settings.combine_multiple_pages_and_resolve(
            subpages,
            splits,
            SplitSettings(width=50, height=50, overlap_ratio=0.1),
            StitchSettings(
                nms_iou_threshold=0.25,
                special_cases={0: 0.95},
            ),
        )

        assert len(resolved.annotations[0]) == 2
        assert len(resolved.annotations[1]) == 1
