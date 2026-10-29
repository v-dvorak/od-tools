from odtools.conversions.annotations import FullPage, SplitPage
from odtools.conversions.bounding_box import BoundingBox


def _page() -> FullPage:
    return FullPage(
        (100, 100),
        [
            [
                # fully inside the top-left quadrant
                _annot(0, 10, 10, 30, 30),
                # fully inside the bottom-right quadrant
                _annot(0, 60, 60, 30, 30),
                # straddles the vertical middle -> intersects both columns
                _annot(0, 40, 10, 20, 20),
                # outside of every split (beyond the grid)
                _annot(0, 10, 60, 5, 5),
            ]
        ],
        ["class_a"],
    )


def _annot(class_id, left, top, width, height):
    from odtools.conversions.annotations import Annotation

    return Annotation(class_id, left, top, width, height, None, f"class_{class_id}")


def _quadrant_splits() -> list[list[BoundingBox]]:
    return [
        [BoundingBox(0, 0, 50, 50), BoundingBox(50, 0, 100, 50)],
        [BoundingBox(0, 50, 50, 100), BoundingBox(50, 50, 100, 100)],
    ]


class TestFromFullPage:
    def test_subpage_structure(self):
        splits = _quadrant_splits()
        split_page = SplitPage.from_full_page(_page(), splits)

        assert split_page.size == (100, 100)
        assert len(split_page.subpages) == 2
        assert len(split_page.subpages[0]) == 2
        # each subpage sees the full class list
        for row in split_page.subpages:
            for sub in row:
                assert sub.class_names == ["class_a"]

    def test_coordinates_shifted_relative_to_cutout(self):
        split_page = SplitPage.from_full_page(_page(), _quadrant_splits())

        # the annotation at (60, 60) lives in the bottom-right quadrant,
        # its coordinates are relative to that quadrant
        br = split_page.subpages[1][1]
        boxes = [a.bbox.coordinates() for a in br.all_annotations()]
        assert (10, 10, 40, 40) in boxes

    def test_straddling_annotation_dropped_without_cut_to_fit(self):
        split_page = SplitPage.from_full_page(
            _page(), _quadrant_splits(), cut_to_fit=False
        )

        # the straddling annotation (40,10,60,30) is not fully inside any quadrant
        total = sum(
            sub.annotation_count() for row in split_page.subpages for sub in row
        )
        # (10,10) in TL, (60,60) in BR, (10,60) is inside BL quadrant actually
        # (10,60,15,65) is fully inside the bottom-left quadrant
        assert total == 3

    def test_straddling_annotation_cut_with_cut_to_fit(self):
        split_page = SplitPage.from_full_page(
            _page(), _quadrant_splits(), cut_to_fit=True
        )

        # with cut_to_fit, the straddling annotation appears (cut) in both columns
        tl = split_page.subpages[0][0]
        tr = split_page.subpages[0][1]
        tl_widths = [a.bbox.width for a in tl.all_annotations()]
        tr_widths = [a.bbox.width for a in tr.all_annotations()]
        assert 10 in tl_widths  # cut at x=50: 40..50 -> width 10
        assert 10 in tr_widths  # cut at x=50: 50..60 -> width 10

    def test_save_to_file_naming(self, tmp_path):
        split_page = SplitPage.from_full_page(_page(), _quadrant_splits())
        from odtools.conversions.formats import OutputFormat

        split_page.save_to_file(tmp_path, "page", OutputFormat.YOLO_DETECTION)

        expected = {
            "page-0-0.txt",
            "page-0-1.txt",
            "page-1-0.txt",
            "page-1-1.txt",
        }
        assert {p.name for p in tmp_path.iterdir()} == expected

    def test_save_to_file_naming_with_tag(self, tmp_path):
        split_page = SplitPage.from_full_page(_page(), _quadrant_splits())
        from odtools.conversions.formats import OutputFormat

        split_page.save_to_file(
            tmp_path, "page", OutputFormat.YOLO_DETECTION, add_tags="x150"
        )

        assert (tmp_path / "page-0-0-x150.txt").exists()
        assert (tmp_path / "page-1-1-x150.txt").exists()
