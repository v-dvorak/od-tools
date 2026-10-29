import pytest

from odtools import BoundingBox, Direction


class TestConstruction:
    def test_valid_box(self):
        bb = BoundingBox(10, 20, 30, 40)
        assert bb.left == 10
        assert bb.top == 20
        assert bb.right == 30
        assert bb.bottom == 40

    def test_negative_left_rejected(self):
        with pytest.raises(ValueError):
            BoundingBox(-1, 0, 10, 10)

    def test_negative_top_rejected(self):
        with pytest.raises(ValueError):
            BoundingBox(0, -1, 10, 10)

    def test_zero_width_rejected(self):
        with pytest.raises(ValueError):
            BoundingBox(0, 0, 0, 10)

    def test_zero_height_rejected(self):
        with pytest.raises(ValueError):
            BoundingBox(0, 0, 10, 0)

    def test_inverted_coordinates_rejected(self):
        with pytest.raises(ValueError):
            BoundingBox(30, 40, 10, 20)

    def test_from_ltwh(self):
        bb = BoundingBox.from_ltwh(5, 6, 10, 20)
        assert bb.coordinates() == (5, 6, 15, 26)

    def test_from_list_of_boxes_encapsulates(self):
        boxes = [
            BoundingBox(0, 0, 10, 10),
            BoundingBox(5, 5, 30, 25),
            BoundingBox(-0, 3, 7, 8),
        ]
        bb = BoundingBox.from_list_of_boxes(boxes)
        assert bb.coordinates() == (0, 0, 30, 25)

    def test_from_list_of_boxes_empty_raises(self):
        with pytest.raises(ValueError):
            BoundingBox.from_list_of_boxes([])


class TestProperties:
    def test_dimensions(self):
        bb = BoundingBox(10, 20, 30, 60)
        assert bb.width == 20
        assert bb.height == 40
        assert bb.area == 800
        assert bb.size() == (20, 40)

    def test_xyxy(self):
        bb = BoundingBox(1, 2, 3, 4)
        assert bb.xyxy() == (1, 2, 3, 4)

    def test_xcycwh(self):
        bb = BoundingBox(0, 0, 10, 20)
        assert bb.xcycwh() == (5, 10, 10, 20)

    def test_center(self):
        bb = BoundingBox(0, 0, 10, 20)
        assert bb.center() == (10, 5)


class TestIntersection:
    def test_intersects_true(self):
        assert BoundingBox(0, 0, 10, 10).intersects(BoundingBox(5, 5, 15, 15))

    def test_intersects_false(self):
        assert not BoundingBox(0, 0, 10, 10).intersects(BoundingBox(20, 20, 30, 30))

    def test_touching_edges_do_not_intersect(self):
        assert not BoundingBox(0, 0, 10, 10).intersects(BoundingBox(10, 0, 20, 10))

    def test_intersects_horizontal(self):
        assert BoundingBox(0, 0, 10, 10).intersects(
            BoundingBox(5, 100, 15, 110), direction=Direction.HORIZONTAL
        )
        assert not BoundingBox(0, 0, 10, 10).intersects(
            BoundingBox(20, 100, 30, 110), direction=Direction.HORIZONTAL
        )

    def test_intersects_vertical(self):
        assert BoundingBox(0, 0, 10, 10).intersects(
            BoundingBox(100, 5, 110, 15), direction=Direction.VERTICAL
        )
        assert not BoundingBox(0, 0, 10, 10).intersects(
            BoundingBox(100, 20, 110, 30), direction=Direction.VERTICAL
        )

    def test_intersection_area(self):
        assert (
            BoundingBox(0, 0, 10, 10).intersection_area(BoundingBox(5, 5, 15, 15)) == 25
        )
        assert (
            BoundingBox(0, 0, 10, 10).intersection_area(BoundingBox(20, 20, 30, 30))
            == 0
        )


class TestIoU:
    def test_identical_boxes(self):
        assert (
            BoundingBox(0, 0, 10, 10).intersection_over_union(BoundingBox(0, 0, 10, 10))
            == 1.0
        )

    def test_no_overlap(self):
        assert (
            BoundingBox(0, 0, 10, 10).intersection_over_union(
                BoundingBox(20, 20, 30, 30)
            )
            == 0.0
        )

    def test_touching_edges(self):
        assert (
            BoundingBox(0, 0, 10, 10).intersection_over_union(
                BoundingBox(10, 0, 20, 10)
            )
            == 0.0
        )

    def test_partial_overlap(self):
        # Intersection = 25, union = 100 + 100 - 25 = 175
        assert BoundingBox(0, 0, 10, 10).intersection_over_union(
            BoundingBox(5, 5, 15, 15)
        ) == pytest.approx(25 / 175)

    def test_one_inside_another(self):
        # Intersection = 36, union = 100
        assert BoundingBox(0, 0, 10, 10).intersection_over_union(
            BoundingBox(2, 2, 8, 8)
        ) == pytest.approx(36 / 100)

    def test_symmetry(self):
        b1, b2 = BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15)
        assert b1.intersection_over_union(b2) == pytest.approx(
            b2.intersection_over_union(b1)
        )


class TestContainment:
    def test_is_fully_inside(self):
        assert BoundingBox(2, 2, 8, 8).is_fully_inside(BoundingBox(0, 0, 10, 10))
        assert not BoundingBox(0, 0, 10, 10).is_fully_inside(BoundingBox(2, 2, 8, 8))

    def test_is_fully_inside_horizontal(self):
        assert BoundingBox(2, 100, 8, 110).is_fully_inside(
            BoundingBox(0, 0, 10, 200), direction=Direction.HORIZONTAL
        )

    def test_is_fully_inside_vertical(self):
        assert BoundingBox(100, 2, 110, 8).is_fully_inside(
            BoundingBox(0, 0, 200, 10), direction=Direction.VERTICAL
        )

    def test_cut_to_fit(self):
        cut = BoundingBox(0, 0, 10, 10).cut_to_fit(BoundingBox(5, 5, 50, 50))
        assert cut.coordinates() == (5, 5, 10, 10)

    def test_cut_to_fit_no_intersection_raises(self):
        with pytest.raises(ValueError):
            BoundingBox(0, 0, 10, 10).cut_to_fit(BoundingBox(20, 20, 30, 30))


class TestShift:
    def test_shift_in_place(self):
        bb = BoundingBox(0, 0, 10, 10)
        bb.shift(5, 7)
        assert bb.coordinates() == (5, 7, 15, 17)

    def test_shift_copy_does_not_mutate(self):
        bb = BoundingBox(0, 0, 10, 10)
        copy = bb.shift_copy(5, 7)
        assert bb.coordinates() == (0, 0, 10, 10)
        assert copy.coordinates() == (5, 7, 15, 17)

    def test_negative_shift(self):
        bb = BoundingBox(10, 10, 20, 20)
        bb.shift(-5, -5)
        assert bb.coordinates() == (5, 5, 15, 15)
