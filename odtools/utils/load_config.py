from pathlib import Path
import json
import yaml

from .mapping_and_names_from_config import get_mapping_and_names_from_config

DEFAULT_CONFIG_PATH = Path("configs/default.json")


def load_config(
    config_path: Path | str | None, default_loader: str = "json", verbose: bool = False
) -> tuple[dict[str, int], list[str], tuple[int, int], float]:
    """
    class id mapping, class output names, window size, overlap ratio
    """

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config_path = Path(config_path)
    loader = default_loader
    if config_path.suffix.lower() == ".yaml":
        loader = "yaml"

    if loader == "json":
        with open(config_path, "r", encoding="utf8") as f:
            loaded_config: dict = json.load(f)
    elif loader == "yaml":
        with open(config_path, "r", encoding="utf8") as f:
            loaded_config: dict = yaml.safe_load(f)
    else:
        raise ValueError

    # Window size
    window_size = loaded_config.get("window_size")
    if window_size is None:
        window_size = (640, 640)
    else:
        window_size = (int(window_size[0]), int(window_size[1]))

    # Overlap ratio
    overlap_ratio = float(loaded_config.get("overlap_ratio", 0.25))

    return (
        *get_mapping_and_names_from_config(loaded_config, verbose),
        window_size,
        overlap_ratio,
    )
