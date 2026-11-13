"""
Tests for the `--musicorpus` CLI tag and for `format_musicorpus_dataset`,
the code path the tag enables when formatting a dataset.

The Musicorpus structure is faked on disk: a manifest, a splits file and
page folders, each holding an `image.png` and a `transcription.mung`.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from odtools import Format, FullPage
from odtools.annotations import Annotation
from odtools.conversions.formatter import format_musicorpus_dataset

# the whole module needs the musicorpus package to build fake datasets
pytest.importorskip(
    "musicorpus.dataset", reason="the musicorpus package is required for these tests"
)

IMAGE_SIZE = (200, 100)  # (width, height)
CLASS_REFERENCE_TABLE = {"noteheadBlack": 0, "stem": 1}
CLASS_OUTPUT_NAMES = ["noteheadBlack", "stem"]


def _annot(class_id: int, left: int, top: int, width: int, height: int) -> Annotation:
    return Annotation(
        class_id, left, top, width, height, None, CLASS_OUTPUT_NAMES[class_id]
    )


def _make_page() -> FullPage:
    """A page with one notehead and one stem, one row per class."""
    return FullPage(
        IMAGE_SIZE,
        [[_annot(0, 10, 10, 20, 10)], [_annot(1, 50, 40, 30, 20)]],
        CLASS_OUTPUT_NAMES,
    )


def _write_manifest(dataset_dir: Path) -> None:
    manifest = {
        "musicorpus_version": "1.0",
        "full_institution_name": "Test Institution",
        "short_institution_name": "Test",
        "institution_url": "https://example.com",
        "full_dataset_name": "Test Dataset",
        "short_dataset_name": "TestDataset",
        "dataset_url": "https://example.com/dataset",
        "dataset_version": "1.0",
        "created_at": "2026-01-01T00:00:00Z",
        "author_emails": ["test@example.com"],
    }
    (dataset_dir / "musicorpus.json").write_text(json.dumps(manifest, indent=4))


def _write_page(
    dataset_dir: Path,
    name: str,
    page: FullPage | None = None,
    with_image: bool = True,
    with_mung: bool = True,
) -> Path:
    """Creates a Musicorpus page folder, optionally incomplete."""
    page_dir = dataset_dir / name
    page_dir.mkdir(parents=True)

    if with_image:
        Image.new("L", IMAGE_SIZE, color=255).save(page_dir / "image.png")

    if with_mung:
        page = page if page is not None else _make_page()
        page.save_to_file(page_dir, "transcription", Format.MUNG)
        # Musicorpus names MuNG transcriptions transcription.mung,
        # while odtools writes the .xml extension
        (page_dir / "transcription.xml").rename(page_dir / "transcription.mung")

    return page_dir


def _write_dolores_page(dataset_dir: Path, name: str) -> Path:
    """Creates a page whose transcription is a Dolores COCO file."""
    page_dir = _write_page(dataset_dir, name, with_mung=False)
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "image.png",
                "width": IMAGE_SIZE[0],
                "height": IMAGE_SIZE[1],
            }
        ],
        "categories": [
            {"id": 1, "name": "noteheadBlack"},
            {"id": 2, "name": "stem"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 40, 30, 20]},
        ],
    }
    (page_dir / "coco-object-detection.json").write_text(json.dumps(coco))
    return page_dir


def _write_splits(dataset_dir: Path, splits: dict[str, list[str]]) -> None:
    (dataset_dir / "splits.json").write_text(json.dumps(splits, indent=4))


@pytest.fixture
def musicorpus_dir(tmp_path: Path) -> Path:
    """A minimal Musicorpus dataset with train/validation/test splits."""
    dataset_dir = tmp_path / "Test.TestDataset"
    dataset_dir.mkdir()
    _write_manifest(dataset_dir)
    _write_splits(
        dataset_dir,
        {
            "train": ["train_page_1", "train_page_2"],
            "validation": ["val_page_1"],
            "test": ["test_page_1"],
        },
    )
    _write_page(dataset_dir, "train_page_1")
    _write_page(dataset_dir, "train_page_2", page=_make_page())
    _write_page(dataset_dir, "val_page_1", page=_make_page())
    _write_page(dataset_dir, "test_page_1", page=_make_page())
    return dataset_dir


def _format(
    output_dir: Path,
    source_dir: Path,
    output_format: Format = Format.YOLO_DETECTION,
    **kwargs,
) -> None:
    format_musicorpus_dataset(
        output_dir,
        source_dir,
        class_reference_table=CLASS_REFERENCE_TABLE,
        class_output_names=CLASS_OUTPUT_NAMES,
        input_format=Format.MUNG,
        output_format=output_format,
        image_format="png",
        **kwargs,
    )


class TestFormatMusicorpusDataset:
    def test_creates_one_subdirectory_per_split(self, musicorpus_dir, tmp_path):
        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        expected_pages = {"train": 2, "validation": 1, "test": 1}
        for split, n_pages in expected_pages.items():
            images = output / split / "images"
            labels = output / split / "labels"
            assert images.is_dir()
            assert labels.is_dir()
            assert len(list(images.glob("*.png"))) == n_pages
            assert len(list(labels.glob("*.txt"))) == n_pages

    def test_pages_are_named_after_their_folders(self, musicorpus_dir, tmp_path):
        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        for split, expected in {
            "train": {"train_page_1", "train_page_2"},
            "validation": {"val_page_1"},
            "test": {"test_page_1"},
        }.items():
            names = {p.stem for p in (output / split / "labels").glob("*.txt")}
            assert names == expected

    def test_annotations_are_converted_to_output_format(self, musicorpus_dir, tmp_path):
        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        yolo_file = output / "train" / "labels" / "train_page_1.txt"
        lines = [line.split() for line in yolo_file.read_text().splitlines()]
        assert len(lines) == 2
        assert sorted(line[0] for line in lines) == ["0", "1"]

        # one notehead at (10, 10, 20, 10) on a 200x100 page:
        # x_center 20/200 = 0.1, y_center 15/100 = 0.15,
        # width 20/200 = 0.1, height 10/100 = 0.1
        notehead = next(line for line in lines if line[0] == "0")
        x, y, w, h = (float(v) for v in notehead[1:])
        assert x == pytest.approx(0.1)
        assert y == pytest.approx(0.15)
        assert w == pytest.approx(0.1)
        assert h == pytest.approx(0.1)

    def test_single_yolo_config_in_dataset_root(self, musicorpus_dir, tmp_path):
        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        # one config in the root, none in the per-split directories
        assert (output / "config.yaml").is_file()
        assert not (output / "train" / "config.yaml").exists()
        assert not (output / "validation" / "config.yaml").exists()

        with open(output / "config.yaml") as f:
            config = yaml.safe_load(f)

        assert config["names"] == {0: "noteheadBlack", 1: "stem"}
        assert Path(config["train"]) == (output / "train" / "images").resolve()
        assert Path(config["val"]) == (output / "validation" / "images").resolve()
        assert Path(config["test"]) == (output / "test" / "images").resolve()

    def test_non_yolo_output_creates_no_config(self, musicorpus_dir, tmp_path):
        output = tmp_path / "out"
        _format(output, musicorpus_dir, output_format=Format.COCO)

        assert not (output / "config.yaml").exists()
        for split in ("train", "validation", "test"):
            labels = output / split / "labels"
            assert len(list(labels.glob("*.json"))) == (2 if split == "train" else 1)

        # MuNG masks are (height, width) binary arrays with no polygon
        # representation: the COCO segmentation falls back to the bbox corners
        coco_file = output / "train" / "labels" / "train_page_1.json"
        coco = json.loads(coco_file.read_text())
        notehead = coco["noteheadBlack"][0]
        assert notehead["segmentation"] == [[10, 10, 30, 10, 30, 20, 10, 20]]
        assert (notehead["left"], notehead["top"]) == (10, 10)

    def test_empty_split_is_skipped(self, musicorpus_dir, tmp_path):
        _write_splits(
            musicorpus_dir,
            {
                "train": ["train_page_1"],
                "validation": [],
                "test": ["test_page_1"],
            },
        )

        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        # no directory and no YOLO config entry for the empty split,
        # val falls back to the train images
        assert not (output / "validation").exists()
        with open(output / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert Path(config["val"]) == (output / "train" / "images").resolve()

    def test_split_with_only_missing_pages_is_skipped(
        self, musicorpus_dir, tmp_path, capsys
    ):
        _write_splits(
            musicorpus_dir,
            {
                "train": ["train_page_1"],
                "validation": ["ghost_page"],
                "test": ["test_page_1"],
            },
        )

        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        assert "split 'validation' contains no usable pages" in capsys.readouterr().out
        assert not (output / "validation").exists()

    def test_split_ratio_creates_inner_train_val(self, musicorpus_dir, tmp_path):
        output = tmp_path / "out"
        _format(output, musicorpus_dir, split_ratio=0.5)

        for split in ("train", "validation", "test"):
            assert (output / split / "images" / "train").is_dir()
            assert (output / split / "images" / "val").is_dir()

        with open(output / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert Path(config["train"]) == (output / "train" / "images" / "train").resolve()
        assert Path(config["val"]) == (
            output / "validation" / "images" / "val"
        ).resolve()

    def test_splits_variant_uses_alternative_splits(self, musicorpus_dir, tmp_path):
        # musicorpus requires train, validation and test in every splits file,
        # additional splits (like "extra") are passed through as extra kwargs
        (musicorpus_dir / "splits.alternative.json").write_text(
            json.dumps(
                {
                    "train": ["test_page_1"],
                    "validation": [],
                    "test": [],
                    "extra": ["val_page_1"],
                }
            )
        )

        output = tmp_path / "out"
        _format(output, musicorpus_dir, splits_variant="alternative")

        assert (output / "train" / "labels" / "test_page_1.txt").is_file()
        assert (output / "extra" / "labels" / "val_page_1.txt").is_file()
        # the default splits are not used
        assert not (output / "validation").exists()

    def test_missing_page_is_skipped_with_warning(self, musicorpus_dir, tmp_path, capsys):
        _write_splits(
            musicorpus_dir,
            {
                "train": ["train_page_1"],
                "validation": ["ghost_page", "val_page_1"],
                "test": ["test_page_1"],
            },
        )

        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        assert "ghost_page" in capsys.readouterr().out
        labels = [p.name for p in (output / "validation" / "labels").glob("*.txt")]
        assert labels == ["val_page_1.txt"]

    def test_page_without_image_is_skipped(self, musicorpus_dir, tmp_path, capsys):
        _write_page(musicorpus_dir, "train_page_3", with_image=False)
        _write_splits(
            musicorpus_dir,
            {
                "train": ["train_page_1", "train_page_3"],
                "validation": ["val_page_1"],
                "test": ["test_page_1"],
            },
        )

        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        assert "has no image" in capsys.readouterr().out
        labels = {p.name for p in (output / "train" / "labels").glob("*.txt")}
        assert labels == {"train_page_1.txt"}

    def test_page_without_transcription_is_skipped(
        self, musicorpus_dir, tmp_path, capsys
    ):
        _write_page(musicorpus_dir, "train_page_3", with_mung=False)
        _write_splits(
            musicorpus_dir,
            {
                "train": ["train_page_1", "train_page_3"],
                "validation": ["val_page_1"],
                "test": ["test_page_1"],
            },
        )

        output = tmp_path / "out"
        _format(output, musicorpus_dir)

        assert "has no mung transcription" in capsys.readouterr().out
        labels = {p.name for p in (output / "train" / "labels").glob("*.txt")}
        assert labels == {"train_page_1.txt"}

    def test_dolores_coco_input(self, tmp_path):
        dataset_dir = tmp_path / "Test.Dolores"
        dataset_dir.mkdir()
        _write_manifest(dataset_dir)
        _write_splits(
            dataset_dir,
            {"train": ["dolores_page_1"], "validation": [], "test": []},
        )
        _write_dolores_page(dataset_dir, "dolores_page_1")

        output = tmp_path / "out"
        format_musicorpus_dataset(
            output,
            dataset_dir,
            class_reference_table=CLASS_REFERENCE_TABLE,
            class_output_names=CLASS_OUTPUT_NAMES,
            input_format=Format.DOLORES_COCO,
            output_format=Format.YOLO_DETECTION,
            image_format="png",
        )

        yolo_file = output / "train" / "labels" / "dolores_page_1.txt"
        lines = [line.split() for line in yolo_file.read_text().splitlines()]
        assert sorted(line[0] for line in lines) == ["0", "1"]

    def test_not_a_musicorpus_dataset_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="not a Musicorpus dataset"):
            _format(tmp_path / "out", empty)

    def test_unsupported_input_format_raises(self, musicorpus_dir, tmp_path):
        with pytest.raises(ValueError, match="Unsupported input format"):
            format_musicorpus_dataset(
                tmp_path / "out",
                musicorpus_dir,
                class_reference_table=CLASS_REFERENCE_TABLE,
                class_output_names=CLASS_OUTPUT_NAMES,
                input_format=Format.COCO,
                output_format=Format.YOLO_DETECTION,
            )


class TestMusicorpusCLITag:
    @pytest.fixture
    def config_path(self, tmp_path: Path) -> Path:
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps({"class_mapping": CLASS_OUTPUT_NAMES, "window_size": [640, 640]})
        )
        return config

    @staticmethod
    def _run_cli(monkeypatch, *argv: str):
        from odtools.__main__ import main

        monkeypatch.setattr(sys, "argv", ["odtools", *argv])
        return main()

    def test_musicorpus_tag_formats_dataset(
        self, musicorpus_dir, config_path, tmp_path, monkeypatch
    ):
        output = tmp_path / "out"
        result = self._run_cli(
            monkeypatch,
            "form",
            str(output),
            str(musicorpus_dir),
            "--musicorpus",
            "-i",
            "mung",
            "-o",
            "yolod",
            "--image_format",
            "png",
            "--config",
            str(config_path),
        )

        assert result == 0
        assert (output / "train" / "labels" / "train_page_1.txt").is_file()
        assert (output / "validation" / "labels" / "val_page_1.txt").is_file()
        assert (output / "config.yaml").is_file()

    def test_without_musicorpus_tag_annot_dir_is_required(
        self, musicorpus_dir, config_path, tmp_path, monkeypatch, capsys
    ):
        with pytest.raises(SystemExit):
            self._run_cli(
                monkeypatch,
                "form",
                str(tmp_path / "out"),
                str(musicorpus_dir),
                "--config",
                str(config_path),
            )

        assert "annot_dir is required" in capsys.readouterr().err

    def test_annot_dir_is_rejected_with_musicorpus(
        self, musicorpus_dir, config_path, tmp_path, monkeypatch, capsys
    ):
        with pytest.raises(SystemExit):
            self._run_cli(
                monkeypatch,
                "form",
                str(tmp_path / "out"),
                str(musicorpus_dir),
                str(musicorpus_dir),
                "--musicorpus",
                "--config",
                str(config_path),
            )

        assert "annot_dir is not needed with --musicorpus" in capsys.readouterr().err

    def test_splits_variant_tag(
        self, musicorpus_dir, config_path, tmp_path, monkeypatch
    ):
        (musicorpus_dir / "splits.alternative.json").write_text(
            json.dumps({"train": ["test_page_1"], "validation": [], "test": []})
        )

        output = tmp_path / "out"
        result = self._run_cli(
            monkeypatch,
            "form",
            str(output),
            str(musicorpus_dir),
            "--musicorpus",
            "--splits_variant",
            "alternative",
            "-i",
            "mung",
            "-o",
            "yolod",
            "--image_format",
            "png",
            "--config",
            str(config_path),
        )

        assert result == 0
        assert (output / "train" / "labels" / "test_page_1.txt").is_file()
        assert not (output / "validation").exists()
