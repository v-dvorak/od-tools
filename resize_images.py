from pathlib import Path

from PIL import Image
import shutil

def copy_and_resize_image(image_path: Path | str, output_path: Path | str, max_size: int = None) -> None:
    """
    Resizes an image so that its larger side equals max_size while maintaining aspect ratio.
    Uses bilinear interpolation for resizing.

    :param image_path: path to image
    :param output_path: path to output image
    :param max_size: maximum size of image
    """

    if max_size is None:
        shutil.copy(image_path, output_path)
        return

    with Image.open(image_path) as img:
        # Get original dimensions
        width, height = img.size

        # Determine the scaling factor to resize the larger side to max_size
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)

        # Resize the image using bilinear interpolation
        img_resized = img.resize((new_width, new_height), Image.BILINEAR)

        # Save the resized image back to the same path
        img_resized.save(output_path)
