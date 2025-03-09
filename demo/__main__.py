import argparse
from pathlib import Path
from timeit import default_timer as timer

import cv2
from PIL import Image

from odtools.Download import (get_path_to_latest_version, update_models, update_demo_images, load_demo_images,
                              OLA_TAG, NOTA_TAG)
from odtools.Inference import InferenceJob, SplitSettings, run_multiple_prediction_jobs
from odtools.Inference.ModelWrappers import YOLODetectionModelWrapper
from odtools.Splitting.SplitUtils import draw_rectangles_on_image

parser = argparse.ArgumentParser(
    prog="Notehead experiments demo"
)

parser.add_argument("-i", "--image_path", type=str, help="Path to image")
parser.add_argument("-o", "--output_dir", type=str, help="Path to output directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
parser.add_argument("--visualize", action="store_true", help="Visualize inference")
parser.add_argument("--update", action="store_true", help="Update models and demo images")

args = parser.parse_args()

# LOAD MODELS
if args.update:
    update_models()

notehead_detector = YOLODetectionModelWrapper(get_path_to_latest_version(NOTA_TAG))
staff_detector = YOLODetectionModelWrapper(get_path_to_latest_version(OLA_TAG))

if args.output_dir:
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True, parents=True)

# LOAD IMAGES
if args.image_path:
    args.image_path = Path(args.image_path)
    if args.image_path.is_dir():
        images_to_process = list(args.image_path.iterdir())
    else:
        images_to_process = [Path(args.image_path)]
else:
    if args.update:
        update_demo_images(verbose=args.verbose)
    images_to_process = load_demo_images()

time_spent_inference = 0

for image_path in images_to_process:
    if args.verbose:
        print(f">>> {image_path}")

    # SETUP INFERENCE JOBS
    # convert image to bw beforehand
    # (color to bw conversion from cv2 does not work in this case)
    image = Image.open(image_path)
    bw_image = image.convert("L")

    # staff
    staff_job = InferenceJob(
        image=bw_image,
        model_wrapper=staff_detector,
        # retrieve only measures and grand staffs
        wanted_ids=[1, 4]
    )

    # noteheads
    notehead_job = InferenceJob(
        image=cv2.imread(str(image_path)),
        model_wrapper=notehead_detector,
        # retrieve only full and empty noteheads
        wanted_ids=[0, 1],
        split_settings=SplitSettings(
            width=640,
            height=640,
            overlap_ratio=0.10,
            iou_threshold=0.25,
            edge_offset_ratio=0.04
        )
    )

    # RUN INFERENCE JOBS
    start = timer()
    combined = run_multiple_prediction_jobs(
        [
            staff_job,
            notehead_job,
        ],
        verbose=False
    )
    time_spent_inference += timer() - start

    # VISUALIZE
    measures = combined.annotations[0]
    grand_staffs = combined.annotations[1]
    noteheads = combined.annotations[2] + combined.annotations[3]

    if len(measures) == 0:
        print("Warning: No measures were found")

    if args.visualize:
        temp = draw_rectangles_on_image(
            image_path,
            [a.bbox for a in grand_staffs],
            thickness=2,
            color=(255, 0, 0)
        )
        temp = draw_rectangles_on_image(
            temp,
            [a.bbox for a in measures],
            thickness=2,
            color=(0, 0, 255)
        )
        draw_rectangles_on_image(
            temp,
            [a.bbox for a in noteheads],
            thickness=2,
            color=(0, 255, 0),
            show=True,
            output_path=str(args.output_dir) if args.output_dir else None
        )

if len(images_to_process) > 0:
    print()
    print(f"Average time spent inference: {time_spent_inference / len(images_to_process)}")
