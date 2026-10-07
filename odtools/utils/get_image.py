from pathlib import Path


def get_image_path(
    folder_path: Path,
    image_name: str = "image",
    suffixes: list[str] = [".jpg", ".jpeg", ".png"],  # noqa: B006
) -> Path:
    for suffix in suffixes:
        p = (folder_path / image_name).with_suffix(suffix)
        if p.exists():
            return p
    raise ValueError(
        f"Image file was not found in {folder_path / image_name}{suffixes}."
    )
