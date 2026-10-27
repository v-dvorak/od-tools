import numpy as np
import pytest

from odtools.conversions.bounding_box import BoundingBox
from odtools.splitting import split_scripts
import itertools


class TestSplitSectionToStarts:
    def test_basic_properties(self):
        starts = split_scripts.split_section_to_starts(1000, 300, 75)
        assert starts[0] == 0
        assert len(starts) >= 2
        # monotonic, never going backwards
        assert list(starts) == sorted(starts)
        # full coverage: last window reaches the end
        assert starts[-1] + 300 == 1000

    def test_adjacent_windows_overlap(self):
        starts = split_scripts.split_section_to_starts(1000, 300, 75)
        for prev, cur in itertools.pairwise(starts):
            assert cur - prev <= 300 - 75  # step keeps minimum overlap

    def test_section_larger_than_total(self):
        with pytest.warns(UserWarning):
            assert split_scripts.split_section_to_starts(100, 300, 10) == [0]

    def test_invalid_section_length(self):
        with pytest.raises(ValueError):
            split_scripts.split_section_to_starts(100, 0, 0)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            split_scripts.split_section_to_starts(100, 50, 50)


class TestCreateSplitBoxMatrix:
    def test_matrix_dimensions(self):
        matrix = split_scripts.create_split_box_matrix((1000, 800), (300, 300), 0.25)
        len(matrix)
        cols = len(matrix[0])
        assert all(len(row) == cols for row in matrix)
        # rows cover height 800, cols cover width 1000
        assert matrix[-1][-1].bottom >= 800
        assert matrix[-1][-1].right >= 1000

    def test_boxes_are_grid_aligned(self):
        matrix = split_scripts.create_split_box_matrix((1000, 800), (300, 300), 0.25)
        for row in matrix:
            for bb in row:
                assert bb.width == 300
                assert bb.height == 300
        # same top per row, same left per column
        assert {bb.top for bb in matrix[0]} == {matrix[0][0].top}
        assert {row[0].left for row in matrix} == {matrix[0][0].left}

    def test_single_window_when_image_fits(self):
        with pytest.warns(UserWarning):
            matrix = split_scripts.create_split_box_matrix((100, 100), (300, 300), 0.25)
        assert len(matrix) == 1
        assert len(matrix[0]) == 1
        assert matrix[0][0].coordinates() == (0, 0, 300, 300)

    def test_invalid_image_size(self):
        with pytest.raises(ValueError):
            split_scripts.create_split_box_matrix((0, 100), (10, 10))

    def test_window_count_matches_matrix(self):
        matrix = split_scripts.create_split_box_matrix((1000, 800), (300, 300), 0.25)
        cols, rows = split_scripts.get_window_count((1000, 800), (300, 300), 0.25)
        assert (rows, cols) == (len(matrix), len(matrix[0]))


class TestCreateSplitImages:
    def test_crops_from_ndarray(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)  # H x W x C
        splits = [[BoundingBox(0, 0, 100, 100), BoundingBox(100, 0, 200, 100)]]
        tiles = split_scripts.create_split_images(image, splits)
        assert len(tiles) == 2
        assert tiles[0].shape == (100, 100, 3)
        assert tiles[1].shape == (100, 100, 3)

    def test_clamps_out_of_bounds_boxes(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        # box extends beyond the image on the right and bottom
        splits = [[BoundingBox(150, 50, 300, 200)]]
        tiles = split_scripts.create_split_images(image, splits)
        assert tiles[0].shape == (50, 50, 3)

    def test_include_indices(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        splits = [
            [BoundingBox(0, 0, 100, 100), BoundingBox(100, 0, 200, 100)],
            [BoundingBox(0, 100, 100, 200), BoundingBox(100, 100, 200, 200)],
        ]
        tiles = split_scripts.create_split_images(image, splits, include_indices=True)
        assert [(r, c) for r, c, _ in tiles] == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_empty_splits_rejected(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            split_scripts.create_split_images(image, [])

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            split_scripts.create_split_images(
                tmp_path / "nope.png", [[BoundingBox(0, 0, 5, 5)]]
            )
