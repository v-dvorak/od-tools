import json
from pathlib import Path

import pytest

from odtools.conversions import conversion_utils


def _data(n: int) -> list[tuple[Path, Path]]:
    return [(Path(f"img_{i}.png"), Path(f"annot_{i}.xml")) for i in range(n)]


class TestSplitDataset:
    def test_ratio(self):
        train, val = conversion_utils.split_dataset(
            _data(100), split_ratio=0.9, seed=42
        )
        assert len(train) == 90
        assert len(val) == 10
        # no overlaps, nothing lost
        assert len({p for p, _ in train + val}) == 100

    def test_seed_determinism(self):
        first = conversion_utils.split_dataset(_data(50), split_ratio=0.5, seed=7)
        second = conversion_utils.split_dataset(_data(50), split_ratio=0.5, seed=7)
        assert first == second

    def test_different_seeds_differ(self):
        first = conversion_utils.split_dataset(_data(50), split_ratio=0.5, seed=1)
        second = conversion_utils.split_dataset(_data(50), split_ratio=0.5, seed=2)
        assert first != second

    def test_input_not_mutated(self):
        data = _data(20)
        original = list(data)
        conversion_utils.split_dataset(data, split_ratio=0.5, seed=1)
        assert data == original

    def test_split_config(self, tmp_path):
        config = {
            "train": ["img_0", "img_1"],
            "test": ["img_2"],
        }
        config_path = tmp_path / "split.json"
        config_path.write_text(json.dumps(config))

        train, val = conversion_utils.split_dataset(
            _data(3), split_ratio=0.9, seed=42, split_config=config_path
        )

        assert [p.name for p, _ in train] == ["img_0.png", "img_1.png"]
        assert [p.name for p, _ in val] == ["img_2.png"]

    def test_split_config_unknown_file_raises(self, tmp_path):
        config = {"train": ["img_0"], "test": ["img_1"]}
        config_path = tmp_path / "split.json"
        config_path.write_text(json.dumps(config))

        with pytest.raises(ValueError):
            conversion_utils.split_dataset(
                _data(3), split_ratio=0.9, seed=42, split_config=config_path
            )
