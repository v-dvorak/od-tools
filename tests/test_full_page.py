import pytest

from odtools.conversions.annotations import FullPage


class TestConstruction:
    def test_from_list_of_coco_annotations_sorts_by_class(self, make_annotation):
        annots = [
            make_annotation(1, 0, 0, 10, 10),
            make_annotation(0, 0, 0, 10, 10),
            make_annotation(1, 20, 20, 10, 10),
        ]
        page = FullPage.from_list_of_coco_annotations(
            (100, 100), annots, ["class_a", "class_b"]
        )
        assert page.class_names == ["class_a", "class_b"]
        assert len(page.annotations) == 2
        assert len(page.annotations[0]) == 1
        assert len(page.annotations[1]) == 2
        assert page.annotation_count() == 3

    def test_invalid_size_raises(self, make_annotation):
        with pytest.raises(ValueError):
            FullPage((0, 100), [[]], ["class_a"])

    def test_out_of_bounds_annotation_warns(self, make_annotation):
        with pytest.warns(UserWarning):
            page = FullPage(
                (100, 100),
                [[make_annotation(0, 90, 90, 50, 50)]],
                ["class_a"],
            )
        # annotation is kept, only a warning is emitted
        assert page.annotation_count() == 1


class TestAllAnnotations:
    def test_all_annotations_flattens_rows(self, make_annotation):
        page = FullPage(
            (100, 100),
            [
                [make_annotation(0, 0, 0, 5, 5), make_annotation(0, 10, 10, 5, 5)],
                [make_annotation(1, 20, 20, 5, 5)],
            ],
            ["class_a", "class_b"],
        )
        assert len(list(page.all_annotations())) == 3


class TestExtendPage:
    def test_same_classes_merge_per_class(self, make_annotation):
        page = FullPage(
            (100, 100),
            [[make_annotation(0, 0, 0, 5, 5)], [make_annotation(1, 10, 10, 5, 5)]],
            ["class_a", "class_b"],
        )
        other = FullPage(
            (100, 100),
            [
                [make_annotation(0, 20, 20, 5, 5)],
                [make_annotation(1, 30, 30, 5, 5), make_annotation(1, 40, 40, 5, 5)],
            ],
            ["class_a", "class_b"],
        )

        page.extend_page(other)

        # rows must stay per-class, not be appended as new rows
        assert page.class_names == ["class_a", "class_b"]
        assert len(page.annotations) == 2
        assert len(page.annotations[0]) == 2
        assert len(page.annotations[1]) == 3
        assert page.annotation_count() == 5

    def test_disjoint_classes_are_appended_with_remapped_ids(self, make_annotation):
        # e.g. merging predictions of two models with different wanted_ids
        page = FullPage(
            (100, 100),
            [
                [make_annotation(0, 0, 0, 5, 5)],
                [make_annotation(1, 10, 10, 5, 5)],
            ],
            ["layout", "note"],
        )
        other = FullPage(
            (100, 100),
            [[make_annotation(0, 50, 50, 5, 5)]],
            ["note"],
        )

        page.extend_page(other)

        assert page.class_names == ["layout", "note"]
        assert len(page.annotations) == 2
        # the new annotation landed in the "note" row ...
        assert len(page.annotations[1]) == 2
        # ... and its class_id was remapped to the row index
        assert all(a.class_id == 1 for a in page.annotations[1])

    def test_new_classes_are_appended(self, make_annotation):
        page = FullPage((100, 100), [[make_annotation(0, 0, 0, 5, 5)]], ["class_a"])
        other = FullPage((100, 100), [[make_annotation(0, 50, 50, 5, 5)]], ["class_b"])

        page.extend_page(other)

        assert page.class_names == ["class_a", "class_b"]
        assert page.annotation_count() == 2
        assert page.annotations[1][0].class_id == 1

    def test_size_mismatch_raises(self, make_annotation):
        page = FullPage((100, 100), [[]], ["class_a"])
        other = FullPage((200, 100), [[]], ["class_a"])
        with pytest.raises(ValueError):
            page.extend_page(other)


class TestCutOffPredictions:
    def test_interior_kept_edge_dropped(self, make_annotation):
        page = FullPage(
            (100, 100),
            [
                [
                    make_annotation(0, 50, 50, 10, 10),  # center, kept
                    make_annotation(0, 0, 0, 10, 10),  # top-left corner, dropped
                    make_annotation(0, 85, 85, 10, 10),  # right of border, dropped
                    make_annotation(
                        0, 65, 65, 10, 10
                    ),  # near bottom-right, inside border, kept
                ]
            ],
            ["class_a"],
        )

        page.cut_off_predictions_too_close_to_edge(edge_offset=20)

        kept = page.annotations[0]
        assert len(kept) == 2
        assert all(a.bbox.left >= 20 and a.bbox.top >= 20 for a in kept)

    def test_edge_tile_flags(self, make_annotation):
        # same page, but the left edge is a "real" image edge -> nothing dropped there
        page = FullPage(
            (100, 100),
            [
                [
                    make_annotation(0, 50, 50, 10, 10),
                    make_annotation(0, 0, 40, 10, 10),  # close to left edge only
                ]
            ],
            ["class_a"],
        )

        page.cut_off_predictions_too_close_to_edge(
            edge_offset=20,
            edge_tile=(False, True, True, True),
        )

        kept = {a.bbox.left for a in page.annotations[0]}
        assert kept == {50, 0}
