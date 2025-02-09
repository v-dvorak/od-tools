from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from .Graph.Names import NodeName
from .Graph.Node import Node, VirtualNode
from .Graph.Tags import SYMBOL_PITCH_TAG
from ..Splitting import draw_rectangles_on_image


def write_numbers_on_image(image_path, measures: list[Node]):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        font = ImageFont.load_default()

    for i, measure in enumerate(measures, start=1):
        draw.text((measure.annot.bbox.left, measure.annot.bbox.top), str(i), font=font, fill=(255, 0, 0))

    image.show()


def write_note_heights_to_image(image, measures: list[Node]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        font = ImageFont.load_default()

    for note in measures:
        draw.text(
            (note.annot.bbox.left, note.annot.bbox.top),
            str(round(note.get_tag(SYMBOL_PITCH_TAG))),
            font=font,
            fill=(0, 255, 0)
        )

    image.show()


def visualize_result(
        image_path: Path,
        measures: list[Node],
        events: list[VirtualNode],
        grand_staff: list[Node] = None
):
    if grand_staff is not None:
        canvas = draw_rectangles_on_image(
            image_path,
            [gs.annot.bbox for gs in grand_staff],
            color=(0, 255, 0),
            thickness=2,
        )
    else:
        canvas = cv2.imread(str(image_path))

    canvas = draw_rectangles_on_image(
        canvas,
        [m.annot.bbox for m in measures],
        color=(0, 0, 255),
        thickness=2,
    )

    canvas = draw_rectangles_on_image(
        canvas,
        [e.total_bbox for e in events],
        color=(255, 0, 0),
        thickness=2
    )

    write_note_heights_to_image(
        canvas,
        [n for c in events for n in c.children()] + [acc for acc in events if acc.name == NodeName.ACCIDENTAL])


def print_info(name: str, header: str, content: list[str], separator: str = "-"):
    print(len(header) * separator)
    print(name)
    if header is not None:
        print(header)
    print(len(header) * separator)
    print("\n".join(content))
    print(len(header) * separator)
    print()


def visualize_input_data(image_path: Path, measures: list[Node], notehead_full: list[Node], notehead_half: list[Node]):
    viz_data = [
        ((0, 0, 255), [m.annot.bbox for m in measures]),
        ((0, 255, 0), [n.annot.bbox for n in notehead_full]),
        ((255, 0, 0), [n.annot.bbox for n in notehead_half]),
    ]

    temp = cv2.imread(str(image_path))
    for (i, (color, data)) in enumerate(viz_data):
        temp = draw_rectangles_on_image(
            temp,
            data,
            color=color,
            thickness=2,
            show=(i == len(viz_data) - 1)
        )
