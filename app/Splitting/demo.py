import cv2

from .SplitUtils import create_split_box_matrix, draw_rectangles_on_image
from ..Conversions.ConversionUtils import get_num_pixels
from ..Splitting.SplitUtils import BoundingBox

rec1 = BoundingBox(0, 0, 4, 4)
rec2 = BoundingBox(3, 3, 5, 5)

print(rec1.intersection_area(rec2))

rec2 = BoundingBox(0, 0, 3, 3)
print(rec2.area())
print((rec2.top - rec2.bottom) * (rec2.right - rec2.left))

# quit()


# image_path = "app/Splitting/30d6c780-c8fe-11e7-9c14-005056827e51_375c6850-f593-11e7-b30f-5ef3fc9ae867.jpg"
image_path = "app/Splitting/3bb9e322-bc61-4307-856b-6f8fb1a640df_2d5f652c-1df0-474c-ae23-3fb699afe808.jpg"

splits = create_split_box_matrix(get_num_pixels(image_path), overlap_ratio=0.25)


# draw_rectangles_on_image(image_path, [x for xs in splits for x in xs])
#
# overlaps = find_overlaps(splits)
# draw_rectangles_on_image(image_path, [x for xs in overlaps for x in xs], shift_based_on_thickness=True)
#
# visualize_cutouts(image_path, splits, overlaps, output_path="split_viz.jpg", spacing=15, opacity=0.5)

def create_split_images(image_path, splits: list[list[BoundingBox]]):
    output = []
    img = cv2.imread(image_path)
    for split in [x for xs in splits for x in xs]:
        # cv2.imshow("", img[split.top:split.bottom, split.left:split.right])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        output.append(img[split.top:split.bottom, split.left:split.right])
    return output


# create_split_images(image_path, splits)

# quit()

from ultralytics import YOLO
from ..Conversions.COCO.AnnotationClasses import COCOFullPage

model = YOLO("best.pt")

tiles = create_split_images(image_path, splits)  # [22:23]

results = model.predict(tiles)
from tqdm import tqdm
import numpy as np

subpages = []
for i in tqdm(range(len(results))):
    result = results[i]
    # print(results[0].boxes.xywh)
    # print(results[0].boxes.conf)
    # print(results[0].boxes.cls)
    # print(results[0].names)

    res = COCOFullPage.from_yolo_result(result)
    # draw_rectangles_on_image(
    #     tiles[0],
    #     [BoundingBox.from_coco_annotation(x) for xs in res.annotations for x in xs],
    #     loaded=True,
    #     color=(0, 255, 0),
    #     thickness=2
    # )
    subpages.append(res)

# subpages = np.array(subpages).reshape((len(splits), len(splits[0])))

# for row in subpages:
#     for box in row:
#         print(". ", end="")
#     print()

# split_page = COCOSplitPage((0, 0), subpages, ["noteheadFull", "noteheadHalf"], splits)
resolved = COCOFullPage.resolve_multiple_pages_to_one(subpages, splits)

draw_rectangles_on_image(
    image_path,
    [BoundingBox.from_coco_annotation(x) for xs in resolved.annotations for x in xs],
    color=(0, 255, 0),
    thickness=2
)
# print(subpages.shape)
#
# print(len(splits), len(splits[0]))
# print(splits)
# COCOSplitPage()
