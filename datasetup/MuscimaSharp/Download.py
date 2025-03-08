from pathlib import Path

from tqdm import tqdm

from odtools.Download import download_image
from ..DatasetDownload import download_dataset

# datasets
DEFAULT_DOWNLOAD_FOLDER = Path("datasets")
MUSCIMAPP_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11372/LRT-2372/MUSCIMA-pp_v1.0.zip"
CVCMUSCIMA_URL = "http://datasets.cvc.uab.es/muscima/CVCMUSCIMA_SR.zip"

# background storage
BACKGROUND_LIST = Path(__file__).parent / "backgrounds.txt"
BACKGROUNDS_DIR = Path("datasets/backgrounds/")
BACKGROUNDS_DIR.mkdir(exist_ok=True, parents=True)


def download_background_images():
    with open(BACKGROUND_LIST, "r") as f:
        background_list = f.read().splitlines()

    for background_id in tqdm(background_list):
        download_image(background_id, Path(background_id + ".png"), BACKGROUNDS_DIR)


def download_muscima():
    download_dataset(MUSCIMAPP_URL, DEFAULT_DOWNLOAD_FOLDER / "muscimapp")
    download_dataset(CVCMUSCIMA_URL, DEFAULT_DOWNLOAD_FOLDER / "cvcmuscima")
