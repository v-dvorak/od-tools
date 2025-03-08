import argparse
from pathlib import Path

import cv2
from mung.io import read_nodes_from_file
from tqdm import tqdm

from odtools import FullPage
from odtools.Conversions.Formats import OutputFormat
from odtools.Splitting.SplitUtils import draw_rectangles_on_image
from .barlines import process_barlines_to_rows
from .barlines import setup_barlines
from .grand_staffs import staffs_to_grand_staffs
from .groupings import setup_groupings
from .measures import rows_of_barlines_to_measures
from .other_symbols import OTHER_SYMBOLS
from .separators import refine_measure_separators
from .settings import CLASS_NAMES
from .staffs import staff_lines_to_staffs
from .system_measures import rows_of_separators_to_system_measures
from .systems import filter_out_system_spans
from .systems import staffs_to_systems
from .utils import bboxes_to_annots, page_to_bboxes

parser = argparse.ArgumentParser(
    "MuNG to OLA dataset building script"
)

# ARGS SETUP
parser.add_argument("output_dir", help="Transformed annotation destination.")
parser.add_argument("image_dir", help="Path to images.")
parser.add_argument("annot_dir", help="Path to annotations.")

parser.add_argument("--image_format", default="jpg", help="Input image format.")
parser.add_argument("-o", "--output_format", default=OutputFormat.COCO.value,
                    choices=OutputFormat.get_all_value())
parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
parser.add_argument("--output_visualize", action="store_true", help="Store image with visualized classes")
parser.add_argument("--debug", action="store_true", help="Debug mode, with visualization")

# ARGS PARSING
args = parser.parse_args()

args.image_dir = Path(args.image_dir)
args.annot_dir = Path(args.annot_dir)
args.output_dir = Path(args.output_dir)
args.output_format = OutputFormat.from_string(args.output_format)
args.output_dir.mkdir(exist_ok=True, parents=True)

# LOAD IMAGES AND MUNG ANNOTS
image_paths: list[Path] = list(args.image_dir.rglob(f"*.{args.image_format}"))
mung_paths: list[Path] = list(args.annot_dir.rglob(f"*.{OutputFormat.to_annotation_extension(OutputFormat.MUNG)}"))

if len(image_paths) != len(mung_paths):
    raise ValueError("Number of found images and annotation do not match.")

for image_source, mung_source in tqdm(
        list(zip(image_paths, mung_paths)),
        desc=f"Converting MuNG to {args.output_format.value}",
        disable=args.verbose
):
    if args.verbose:
        print(f">>> {image_source}")

    image_source_without_extension = image_source.parent / image_source.stem

    # LOAD BOUNDING BOXES FROM MUNG
    nodes = read_nodes_from_file(mung_source)
    barlines, staff_lines, groupings, separators, other_symbols = page_to_bboxes(
        nodes,
        [
            ["barlineHeavy", "barline"],
            ["staffLine"],
            ["staffGrouping"],
            ["measureSeparator"],
            OTHER_SYMBOLS,
        ]
    )

    # TURN RETRIEVED BBOXES INTO ANNOTATIONS
    # clean, fill in and clean up retrieved bboxes from mung
    staffs = sorted(staff_lines_to_staffs(staff_lines), key=lambda x: x.top)
    groupings = setup_groupings(groupings)
    barlines = setup_barlines(barlines, staffs, groupings, verbose=args.verbose)

    separators = refine_measure_separators(separators, barlines)
    # separators are used to retrieve possible locations of systems
    # we assume the worst possible input and continue the computation with separators
    # that are at the middle of given system (not visual middle, rather that they are at the middle of the list)
    system_spans = sorted(filter_out_system_spans(separators), key=lambda x: x.top)
    systems = staffs_to_systems(staffs, system_spans)

    # code for barlines is used to process as they are almost the same thing
    rows_of_separators = process_barlines_to_rows(separators, systems, 0.3, verbose=args.verbose)
    system_measures = rows_of_separators_to_system_measures(rows_of_separators, other_symbols)

    rows_of_barlines = process_barlines_to_rows(barlines, staffs, 0.3, verbose=args.verbose)
    measures = rows_of_barlines_to_measures(rows_of_barlines, system_measures, 0.7)

    grand_staffs = staffs_to_grand_staffs(staffs, groupings + system_spans, verbose=args.verbose)

    if args.output_visualize:
        for i, layout_objects in enumerate([system_measures, measures, staffs, systems, grand_staffs]):
            draw_rectangles_on_image(
                image_source,
                layout_objects,
                thickness=2,
                color=(0, 0, 255),
                output_path=args.output_dir / (image_source.stem + f"_{i}.png")
            )

    # SETUP AND SAVE PAGE OF ANNOTATIONS
    height, width = cv2.imread(str(image_source)).shape[:2]
    annots = bboxes_to_annots([system_measures, measures, staffs, systems, grand_staffs])
    page = FullPage((width, height), annots, CLASS_NAMES)
    page.save_to_file(
        args.output_dir,
        image_source.stem,
        args.output_format
    )
