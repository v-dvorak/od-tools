import cv2

from .SplitUtils import create_split_box_matrix, draw_rectangles_on_image
from ..Conversions.ConversionUtils import get_num_pixels
from ..Splitting.SplitUtils import BoundingBox

# output_path = "app/Splitting/30d6c780-c8fe-11e7-9c14-005056827e51_375c6850-f593-11e7-b30f-5ef3fc9ae867.jpg"
image_path = "app/Splitting/3bb9e322-bc61-4307-856b-6f8fb1a640df_2d5f652c-1df0-474c-ae23-3fb699afe808.jpg"

splits = create_split_box_matrix(get_num_pixels(image_path), overlap_ratio=0.25)


# draw_rectangles_on_image(output_path, [x for xs in splits for x in xs])
#
# overlaps = find_overlaps(splits)
# draw_rectangles_on_image(output_path, [x for xs in overlaps for x in xs], shift_based_on_thickness=True)
#
# visualize_cutouts(output_path, splits, overlaps, output_path="split_viz.jpg", spacing=15, opacity=0.5)

def create_split_images(image_path, splits: list[list[BoundingBox]]):
    output = []
    img = cv2.imread(image_path)
    for split in [x for xs in splits for x in xs]:
        # cv2.imshow("", img[split.top:split.bottom, split.left:split.right])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        output.append(img[split.top:split.bottom, split.left:split.right])
    return output


# create_split_images(output_path, splits)

# quit()

from ultralytics import YOLO
from ..Conversions.Annotations.FullPage import FullPage

model = YOLO("best.pt")

tiles = create_split_images(image_path, splits)  # [22:23]

results = model.predict(tiles)
from tqdm import tqdm
import numpy as np

from time import time

start = time()

subpages = []
for i in tqdm(range(len(results))):
    result = results[i]
    res = FullPage.from_yolo_result(result)
    subpages.append(res)

resolved = FullPage.combine_multiple_pages_and_resolve(subpages, splits)

print(time() - start)

draw_rectangles_on_image(
    image_path,
    [x.bbox for xs in resolved.annotations for x in xs],
    color=(0, 255, 0),
    thickness=2
)
# print(subpages.shape)
#
# print(len(splits), len(splits[0]))
# print(splits)
# SplitPage()
