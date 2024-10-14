from ..Conversions.ConversionUtils import get_num_pixels
from .SplitUtils import create_split_box_matrix, draw_rectangles_on_image, find_overlaps, visualize_cutouts

from ..Splitting.SplitUtils import Rectangle

rec1 = Rectangle(0, 0, 4, 4)
rec2 = Rectangle(3, 3, 5, 5)

print(rec1.intersection_area(rec2))

rec2 = Rectangle(0, 0, 3, 3)
print(rec2.area())
print((rec2.top - rec2.bottom) * (rec2.right - rec2.left))

# quit()


image_path = "app/Splitting/30d6c780-c8fe-11e7-9c14-005056827e51_375c6850-f593-11e7-b30f-5ef3fc9ae867.jpg"

splits = create_split_box_matrix(get_num_pixels(image_path))
draw_rectangles_on_image(image_path, [x for xs in splits for x in xs])

overlaps = find_overlaps(splits)
draw_rectangles_on_image(image_path, [x for xs in overlaps for x in xs], shift_based_on_thickness=True)

visualize_cutouts(image_path, splits, overlaps, output_path="split_viz.png")
