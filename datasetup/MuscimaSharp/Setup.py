import shutil
from pathlib import Path

import PIL.ImageOps
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm

from .Download import BACKGROUNDS_DIR

# entry points for annotation and image retrieval
MUSCIMA_IMAGES_DIR = Path("datasets/cvcmuscima/CvcMuscima-Distortions")
MUSCIMA_ANNOTATIONS_DIR = Path("datasets/muscimapp/v1.0/data/cropobjects_withstaff")

# output paths
OUTPUT_DIR = Path("datasets/muscima-sharp")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "labels").mkdir(exist_ok=True, parents=True)

DISTORTION_NAMES = [
    "ideal",
    "interrupted",
    "kanungo",
    "staffline-thickness-variation-v1",
    "staffline-thickness-variation-v2",
    "staffline-y-variation-v1",
    "staffline-y-variation-v2",
    "whitespeckles"
]


def paste_black_pixels(background_image_path: Path, overlay_image_path: Path, zoom_value: float):
    """
    Copies zoomed in background image under a binary overlay image.

    :param background_image_path: Path to background image
    :param overlay_image_path: Path to overlay image
    :param zoom_value: how much to zoom the background image, equal to or greater than 1
    """
    # open images
    background = Image.open(background_image_path).convert("RGBA")
    # turn overlay into true BW
    overlay = Image.open(overlay_image_path).convert("L")

    def zoom_in(img: PIL.ImageFile, zoom_factor: float) -> PIL.ImageFile:
        """
        Zooms into the image by given amount.

        :param img: loaded PIL Image
        :param zoom_factor: how much to zoom in
        :return: zoomed PIL Image
        """
        if zoom_factor < 1:
            raise ValueError("Zoom factor for zooming in must be greater than or equal to 1.")

        w, h = img.size
        zoom2 = zoom_factor * 2
        img = img.crop(
            (w / 2 - w / zoom2,
             h / 2 - h / zoom2,
             w / 2 + w / zoom2,
             h / 2 + h / zoom2)
        )
        return img.resize((w, h), Image.LANCZOS)

    zoomed_background = zoom_in(background, zoom_value)

    # check if background fully fills the overlay area
    o_width, o_height = overlay.size
    zb_width, zb_height = zoomed_background.size

    if zb_width < o_width or zb_height < o_height:
        # resize zoomed background to cover overlay size if necessary
        scale_factor = max(o_width / zb_width, o_height / zb_height)
        zoomed_background = zoomed_background.resize(
            (int(zb_width * scale_factor), int(zb_height * scale_factor)),
            Image.LANCZOS
        )
    # background is now set to cover the whole overlay

    # overlay to binary mask of 1 nad 0
    overlay_mask = overlay.point(lambda p: 255 if p == 0 else 0)

    # overlay -> RGBA
    mask_rgba = overlay_mask.convert("RGBA")
    mask_rgba = mask_rgba.split()[0]  # Take the single channel (L) of the mask to use as alpha channel

    # paste only the black pixels
    zoomed_background.paste(
        (0, 0, 0, 255),
        (0, 0, o_width, o_height),
        mask_rgba
    )

    # ensure the final image is the same size as the overlay
    final_image = Image.new("RGBA", (o_width, o_height), (0, 0, 0, 0))
    final_image.paste(zoomed_background, (0, 0))

    # final_image.show()
    final_image.save(overlay_image_path)


def setup_images_and_annotations():
    for annot_file in tqdm(list(MUSCIMA_ANNOTATIONS_DIR.iterdir())):
        for distortion_name in DISTORTION_NAMES:
            name = annot_file.stem
            temp = name.split("_")
            writer_id = temp[1].lower()
            image_id = int(temp[2][2:])

            path_to_image = MUSCIMA_IMAGES_DIR / distortion_name / writer_id / "image" / f"p{image_id:03}.png"

            stripped_name = annot_file.stem.split("D")[0]
            new_image_name = f"{stripped_name}D-{distortion_name}.png"
            new_annot_name = f"{stripped_name}D-{distortion_name}.xml"

            shutil.copy(path_to_image, OUTPUT_DIR / "images" / new_image_name)
            shutil.copy(annot_file, OUTPUT_DIR / "labels" / new_annot_name)


def invert_image_colors():
    for image_path in tqdm(list((OUTPUT_DIR / "images").rglob("*.png"))):
        image = Image.open(image_path)
        PIL.ImageOps.invert(image).save(image_path)


def paste_background_to_whole_dataset():
    background_images = list(BACKGROUNDS_DIR.rglob("*.png"))
    from random import Random

    rnd = Random()
    rnd.seed(42)

    for image_path in tqdm(list((OUTPUT_DIR / "images").rglob("*.png"))):
        rnd_index = rnd.randint(0, len(background_images) - 1)
        background_image = background_images[rnd_index]
        paste_black_pixels(background_image, image_path, 1.3)
