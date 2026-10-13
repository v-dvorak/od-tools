import shutil
import re
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from musicorpus import Dataset


DEFAULT_MUNG_FILE_NAME = "transcription.mung"
OMNIOMR_OUTPUT_DIR = Path("datasets/omniomr-data")


def is_uuid_pair(s: str) -> bool:
    uuid_pattern = r"[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    full_pattern = f"^{uuid_pattern}_{uuid_pattern}$"
    return re.fullmatch(full_pattern, s) is not None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input directory",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OMNIOMR_OUTPUT_DIR,
    )

    args = parser.parse_args()

    dataset = Dataset(args.source)
    args.output = Path(args.output)
    args.output.mkdir(exist_ok=True, parents=True)

    print(f"Collecting OmniOMR dataset from {args.source} to {args.output}")

    split_names = dataset.splits(args.split_name).split_names()

    for split_name in split_names:
        (args.output / split_name).mkdir(parents=True, exist_ok=True)
        for page in tqdm(
            dataset.split(split_name, args.split_name),
            desc=f"Processing '{split_name}'",
        ):
            image_path = page.image_path()
            assert image_path is not None
            img_suffix = image_path.name.split(".")[-1]

            shutil.copyfile(
                image_path,
                args.output / split_name / f"{page.name}.{img_suffix}",
            )

            mung_transcription = page.transcription_path("mung")
            assert mung_transcription is not None
            shutil.copyfile(
                mung_transcription,
                args.output / split_name / f"{page.name}.xml",
            )
