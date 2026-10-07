import shutil
import re
from pathlib import Path
from tqdm import tqdm
import json
from argparse import ArgumentParser

from .get_image import get_image_path


DEFAULT_MUNG_FILE_NAME = "transcription.mung"


def is_uuid_pair(s: str) -> bool:
    uuid_pattern = r"[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    full_pattern = f"^{uuid_pattern}_{uuid_pattern}$"
    return re.fullmatch(full_pattern, s) is not None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--splits", type=Path, default=None, help="Path to splits config"
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.splits is None:
        args.splits = args.source / "splits.json"

    print(f"Collecting Musicorpus dataset from {args.source} to {args.output}")

    with open(args.splits, "r") as f:
        splits: dict[str, list[str]] = json.load(f)

    for split, data in splits.items():
        print(f"    Processing {split}")
        (args.output / split).mkdir(parents=True, exist_ok=True)
        for i, dato in tqdm(enumerate(data)):
            source_directory = args.source / dato
            # for system in (source_directory / "Systems").iterdir():
            # if system.is_dir():
            img_path = get_image_path(source_directory)
            img_suffix = img_path.name.split(".")[-1]
            shutil.copyfile(
                img_path,
                args.output / split / f"{source_directory.name}.{img_suffix}",
            )

            shutil.copyfile(
                source_directory / DEFAULT_MUNG_FILE_NAME,
                args.output / split / f"{source_directory.name}.xml",
            )

            # import imagesize

            # width, height = imagesize.get(img_path)
            # # print(width, height)
            # with open(source_directory / DEFAULT_ANNOT_FILE_NAME, "r") as f:
            #     layout = json.load(f)

            # layout["images"][0]["width"] = width
            # layout["images"][0]["height"] = height

            # with open(OMNIOMR_OUTPUT_DIR / split / f"{source_directory.name}.json", "w") as f:
            #     json.dump(layout, f)
