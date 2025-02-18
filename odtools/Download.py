"""
Takes care of model downloading and updating.
It provides methods to download and setup images for demo inference.
"""

from pathlib import Path

import requests
from packaging import version
from tqdm import tqdm

# GitHug repo details
OWNER = "v-dvorak"
REPO = "omr-layout-analysis"

OLA_TAG = "ola"
NOTA_TAG = "nota"
MODEL_TAGS = [OLA_TAG, NOTA_TAG]

DOWNLOAD_DIR = Path("models")
IMAGE_DIR = Path("images")

DEMO_IMAGE_IDS = [
    "16e3cbb5-bd89-48c2-80a6-cccbcaeb7893",
    "9d4412a1-0cf3-4475-a022-9f37984272fb",
    "2405cebe-37f0-4a60-932c-f443027246e6",
    "2e117f2e-4c19-4bc3-ba6b-5531ca623e22"
]


def get_path_to_latest_version(tag: str, model_dir: Path = None) -> Path:
    """
    Searches given directory for a model that begins with a given tag
    and returns the path to the latest version of it.

    :param tag: the tag to search for
    :param model_dir: the directory to search in
    :return: the path to the latest version of the model
    """
    if model_dir is None:
        model_dir = DOWNLOAD_DIR

    all_pts = list(model_dir.glob(f"{tag}*.pt"))

    if len(all_pts) == 0:
        raise FileNotFoundError(f"{tag} not found in {model_dir}")

    latest_release = max(all_pts, key=lambda file: version.parse(file.name.split("-")[-4]))
    return latest_release


def _get_latest_release(tag: str):
    # GitHub API URL to fetch releases for a given tag
    api_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases"

    # get the releases from GitHub API
    response = requests.get(api_url)
    releases = response.json()

    # filter releases by the given tag and extract the versions
    relevant_releases = [release for release in releases if release["tag_name"].startswith(f"{tag}-")]

    if not relevant_releases:
        print(f"No releases found for tag: {tag}")
        return None

    # find the release with the highest version
    latest_release = max(relevant_releases, key=lambda r: version.parse(r["tag_name"].split("-v")[-1]))

    return latest_release


def _download_pt_file(release, download_dir: Path = None):
    if download_dir is None:
        download_dir = DOWNLOAD_DIR
    # find .pt in the release assets
    pt_file = next((asset for asset in release["assets"] if asset["name"].endswith(".pt")), None)

    if pt_file:
        download_url = pt_file["browser_download_url"]
        file_name = pt_file["name"]

        file_path = download_dir / file_name
        if file_path.exists():
            print(f"> Latest model version {release['tag_name']} already downloaded")
            return

        response = requests.get(download_url, stream=True)
        # get info for progress bar
        total_size = int(response.headers.get("Content-Length", 0))

        with open(file_path, "wb") as file, tqdm(
                desc=file_name,
                total=total_size,
                unit="B",
                unit_scale=True,
                ncols=100
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

    else:
        print("> Model file not found in the release assets")


def update_models(model_dir: Path = None):
    """
    Checks if models are downloaded and are the newest version available,
    downloads the newest version if necessary.

    :param model_dir: directory to download models to
    """
    if model_dir is None:
        model_dir = DOWNLOAD_DIR

    model_dir.mkdir(exist_ok=True, parents=True)

    for tag in MODEL_TAGS:
        print(f"Fetching latest release for model: {tag}")

        latest_release = _get_latest_release(tag)

        if latest_release:
            print(f"> Latest release for {tag}: {latest_release['tag_name']}")
            _download_pt_file(latest_release, download_dir=model_dir)

        print()


def download_image(
        img_id: str,
        file_name: Path,
        output_dir: Path
):
    api_url = "https://api.kramerius.mzk.cz/search/iiif/uuid:{img_id}/full/{size}/0/default.jpg"
    url = api_url.format(img_id=img_id, size="max")

    response = requests.get(url)
    if response.status_code == 200:
        file_path = Path(output_dir / file_name)
        # write to file
        with open(file_path, "wb") as file:
            file.write(response.content)
    else:
        print(f"Error: {response.status_code}")


def update_demo_images(image_dir: Path = None, verbose: bool = False):
    """
    Checks if demo images are downloaded, if not, downloads them.

    :param image_dir: directory with demo images
    :param verbose: make script verbose
    """
    print("Fetching demo images")

    if image_dir is None:
        image_dir = IMAGE_DIR

    image_dir.mkdir(exist_ok=True, parents=True)

    for img_id in tqdm(DEMO_IMAGE_IDS, disable=not verbose):
        file_name = Path(img_id + ".jpg")
        if (image_dir / file_name).exists():
            continue
        else:
            download_image(img_id, file_name, image_dir)


def load_demo_images() -> list[Path]:
    """
    Returns list of paths to demo images.

    :return: list of paths to demo images
    """
    return list(IMAGE_DIR.glob("*.jpg"))


def update_all_default_resources():
    print("Downloading latest models and demo images to default folders:")
    print(f"> Models: {DOWNLOAD_DIR.resolve()}")
    print(f"> Demo images: {IMAGE_DIR.resolve()}")
    print()
    try:
        update_models()
        update_demo_images(verbose=True)
    except requests.exceptions.ConnectionError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    update_all_default_resources()
