from odtools.conversions.annotations import Annotation
from odtools.conversions.bounding_box import BoundingBox, Direction


class TestConstruction:
    def test_ltwh_to_xyxy(self):
        annot = Annotation(0, 10, 20, 30, 40, None, "a")
        assert annot.bbox.coordinates() == (10, 20, 40, 60)

    def test_from_bbox(self):
        bbox = BoundingBox(1, 2, 3, 4)
        annot = Annotation.from_bbox(5, bbox, "a")
        assert annot.class_id == 5
        assert annot.bbox.coordinates() == bbox.coordinates()

    def test_equality_is_value_based(self):
        a1 = Annotation(0, 0, 0, 10, 10, None, "a")
        a2 = Annotation(0, 0, 0, 10, 10, None, "a")
        a3 = Annotation(1, 0, 0, 10, 10, None, "a")
        assert a1 == a2
        assert a1 != a3
        assert len({a1, a2}) == 1  # hashable and consistent with __eq__


class TestSegmentation:
    def test_bounding_box_from_segmentation_returns_ltwh(self):
        segm = [(10, 20), (50, 20), (50, 60), (10, 60)]
        left, top, width, height = Annotation.bounding_box_from_segmentation(segm)
        assert (left, top, width, height) == (10, 20, 40, 40)

    def test_bounding_box_from_unordered_segmentation(self):
        segm = [(50, 60), (10, 20), (50, 20), (10, 60)]
        assert Annotation.bounding_box_from_segmentation(segm) == (10, 20, 40, 40)

    def test_segmentation_from_bounding_box(self):
        segm = Annotation.segmentation_from_bounding_box(BoundingBox(10, 20, 50, 60))
        assert segm == [(10, 20), (50, 20), (50, 60), (10, 60)]

    def test_segm_bbox_round_trip(self):
        segm = [(5, 5), (100, 5), (100, 70), (5, 70)]
        left, top, width, height = Annotation.bounding_box_from_segmentation(segm)
        bbox = BoundingBox.from_ltwh(left, top, width, height)
        assert bbox.coordinates() == (5, 5, 100, 70)


class TestPosition:
    def test_adjust_position_in_place(self):
        annot = Annotation(0, 10, 20, 30, 40, None, "a")
        annot.adjust_position(5, -5)
        assert annot.bbox.coordinates() == (15, 15, 45, 55)

    def test_adjust_position_copy(self):
        annot = Annotation(0, 10, 20, 30, 40, None, "a")
        copy = annot.adjust_position_copy(5, -5)
        assert annot.bbox.coordinates() == (10, 20, 40, 60)
        assert copy.bbox.coordinates() == (15, 15, 45, 55)
        assert copy.class_id == annot.class_id


class TestGeometryDelegation:
    def test_intersects(self):
        a1 = Annotation(0, 0, 0, 10, 10, None, "a")
        a2 = Annotation(0, 5, 5, 10, 10, None, "b")
        assert a1.intersects(a2)
        assert a1.intersection_over_union(a2) > 0

    def test_directional_intersects(self):
        a1 = Annotation(0, 0, 0, 10, 10, None, "a")
        a2 = Annotation(0, 100, 5, 10, 10, None, "b")
        # vertically overlapping, horizontally disjoint
        assert a1.intersects(a2, direction=Direction.VERTICAL)
        assert not a1.intersects(a2, direction=Direction.HORIZONTAL)
