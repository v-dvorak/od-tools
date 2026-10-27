import numpy as np
import pytest
from PIL import Image

from odtools.conversions.annotations import FullPage
from odtools.conversions.formats import InputFormat, OutputFormat

IMAGE_SIZE = (200, 100)  # (width, height), deliberately non-square


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "page.png"
    Image.new("L", IMAGE_SIZE, color=255).save(path)
    return path


def _make_page() -> FullPage:
    return FullPage(
        IMAGE_SIZE,
        [
            [_annot(0, 10, 10, 20, 30)],
            [_annot(1, 50, 40, 40, 20), _annot(1, 100, 10, 30, 30)],
        ],
        ["class_a", "class_b"],
    )


def _annot(class_id, left, top, width, height, confidence=1.0):
    from odtools.conversions.annotations import Annotation

    return Annotation(class_id, left, top, width, height, None, confidence=confidence)


def _boxes(page: FullPage):
    return sorted(
        (a.class_id, a.bbox.left, a.bbox.top, a.bbox.width, a.bbox.height)
        for a in page.all_annotations()
    )


class TestYOLODetectionRoundTrip:
    def test_round_trip(self, image_path, tmp_path):
        page = _make_page()
        page.save_to_file(tmp_path, "page", OutputFormat.YOLO_DETECTION)

        # the annotation extension must match what save_to_file produced
        assert (tmp_path / "page.txt").exists()

        loaded = FullPage.load_from_file(
            tmp_path / "page.txt",
            image_path,
            {"class_a": 0, "class_b": 1},
            ["class_a", "class_b"],
            InputFormat.YOLO_DETECTION,
        )

        # regression: image size must be parsed as (width, height)
        assert loaded.size == IMAGE_SIZE

        # coordinates may shift by at most one pixel due to int truncation
        original = _boxes(page)
        got = _boxes(loaded)
        assert len(original) == len(got)
        for (c1, l1, t1, w1, h1), (c2, l2, t2, w2, h2) in zip(original, got):
            assert c1 == c2
            assert abs(l1 - l2) <= 1
            assert abs(t1 - t2) <= 1
            assert abs(w1 - w2) <= 1
            assert abs(h1 - h2) <= 1

    def test_with_confidence(self, tmp_path):
        page = _make_page()
        page.save_to_file(
            tmp_path, "page", OutputFormat.YOLO_DETECTION, with_confidence=True
        )

        lines = (tmp_path / "page.txt").read_text().splitlines()
        for line in lines:
            parts = line.split()
            assert len(parts) == 6  # class_id x y w h confidence


class TestYOLOSegmentationParsing:
    def test_polygon_parses_to_correct_bbox(self, image_path, tmp_path):
        # square polygon covering the middle half of the image
        content = "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
        annot_path = tmp_path / "seg.txt"
        annot_path.write_text(content)

        page = FullPage.load_from_file(
            annot_path,
            image_path,
            {"class_a": 0},
            ["class_a"],
            InputFormat.YOLO_SEGMENTATION,
        )

        annot = next(iter(page.all_annotations()))
        # regression: boxes used to be inflated by (left, top)
        assert annot.bbox.left == 50
        assert annot.bbox.top == 25
        assert annot.bbox.right == 150
        assert annot.bbox.bottom == 75

    def test_malformed_line_raises(self, image_path, tmp_path):
        annot_path = tmp_path / "seg.txt"
        annot_path.write_text("0 0.5 0.5\n")

        with pytest.raises(ValueError):
            FullPage.load_from_file(
                annot_path,
                image_path,
                {"class_a": 0},
                ["class_a"],
                InputFormat.YOLO_SEGMENTATION,
            )


class TestCOCORoundTrip:
    def test_round_trip(self, tmp_path):
        page = FullPage(
            IMAGE_SIZE,
            [
                [_annot(0, 10, 10, 20, 30)],
                [_annot(1, 50, 40, 40, 20)],
            ],
            ["class_a", "class_b"],
        )
        # COCO output needs segmentation masks
        for annot in page.all_annotations():
            b = annot.bbox
            annot.mask = [
                (b.left, b.top),
                (b.right, b.top),
                (b.right, b.bottom),
                (b.left, b.bottom),
            ]

        page.save_to_file(tmp_path, "page", OutputFormat.COCO)
        assert (tmp_path / "page.json").exists()

        loaded = FullPage.load_from_file(
            tmp_path / "page.json",
            None,
            {"class_a": 0, "class_b": 1},
            ["class_a", "class_b"],
            InputFormat.COCO,
        )

        assert loaded.size == IMAGE_SIZE
        assert _boxes(page) == _boxes(loaded)


class TestMuNGRoundTrip:
    def test_round_trip(self, image_path, tmp_path):
        page = _make_page()

        page.save_to_file(tmp_path, "page", OutputFormat.MUNG)
        assert (tmp_path / "page.xml").exists()

        # MuNG does not store page size, the image is needed to recover it
        loaded = FullPage.load_from_file(
            tmp_path / "page.xml",
            image_path,
            {"class_a": 0, "class_b": 1},
            ["class_a", "class_b"],
            InputFormat.MUNG,
        )

        assert _boxes(page) == _boxes(loaded)


class TestSemanticSegmentationOutput:
    def test_two_classes_saved_as_rgb(self, tmp_path):
        page = FullPage(
            IMAGE_SIZE,
            [
                [_masked_annot(0, 10, 10, 20, 20)],
                [_masked_annot(1, 60, 40, 30, 20)],
            ],
            ["class_a", "class_b"],
        )

        page.save_to_file(tmp_path, "page", OutputFormat.SEMANTIC_SEGMENTATION)

        mask = Image.open(tmp_path / "page.png")
        assert mask.mode == "RGB"
        assert mask.size == IMAGE_SIZE
        arr = np.asarray(mask)
        # each class occupies its own channel
        assert arr[15, 15, 0] > 0 and arr[15, 15, 1] == 0
        assert arr[50, 70, 1] > 0 and arr[50, 70, 0] == 0

    def test_many_classes_saved_as_single_mask(self, tmp_path):
        page = FullPage(
            IMAGE_SIZE,
            [
                [_masked_annot(0, 10, 10, 20, 20)],
                [_masked_annot(1, 60, 40, 30, 20)],
                [_masked_annot(2, 10, 60, 20, 20)],
                [_masked_annot(3, 60, 10, 30, 20)],
            ],
            ["a", "b", "c", "d"],
        )

        page.save_to_file(tmp_path, "page", OutputFormat.SEMANTIC_SEGMENTATION)

        mask = Image.open(tmp_path / "page.png")
        assert mask.mode == "L"
        arr = np.asarray(mask)
        # all classes merged into one binary mask
        assert arr[15, 15] > 0
        assert arr[50, 70] > 0
        assert arr[5, 5] == 0


def _masked_annot(class_id, left, top, width, height):
    from odtools.conversions.annotations import Annotation

    return Annotation(
        class_id,
        left,
        top,
        width,
        height,
        np.ones((height, width), dtype=np.uint8),
    )
