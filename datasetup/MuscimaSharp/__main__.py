from .Download import download_muscima, download_background_images
from .Setup import setup_images_and_annotations, invert_image_colors, paste_background_to_whole_dataset

print("Downloading and setting up Muscima++ and CVC Muscima datasets...")
download_muscima()
print()

print("Downloading and setting up background images ...")
download_background_images()
print()

print("Extracting images and copying annotations ...")
setup_images_and_annotations()
print()

print("Inverting image colors ...")
invert_image_colors()
print()

print("Pasting background to whole dataset ...")
paste_background_to_whole_dataset()
print()
