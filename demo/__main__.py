import argparse
from pathlib import Path
from timeit import default_timer as timer

import cv2
from PIL import Image
from ultralytics import YOLO

from app.Download import get_path_to_latest_version, update_models, OLA_TAG, NOTA_TAG
from app.Download import update_demo_images, load_demo_images
from app.Inference import InferenceJob, SplitSettings, run_multiple_prediction_jobs, ModelType
from app.Reconstruction import preprocess_annots_for_reconstruction, reconstruct_note_events

parser = argparse.ArgumentParser(
    prog="Notehead experiments demo"
)

parser.add_argument("-i", "--image_path", type=str, help="Path to image")
parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
parser.add_argument("--visualize", action="store_true", help="Visualize every step of inference")

args = parser.parse_args()

# LOAD MODELS
update_models()
notehead_detector = YOLO(get_path_to_latest_version(NOTA_TAG))
staff_detector = YOLO(get_path_to_latest_version(OLA_TAG))

# LOAD IMAGES
if args.image_path:
    images_to_process = [Path(args.image_path)]
else:
    update_demo_images(verbose=args.verbose)
    images_to_process = load_demo_images()

time_spent_inference = 0
time_spent_reconstruction = 0

for image_path in images_to_process:
    # SETUP INFERENCE JOBS

    # convert image to bw beforehand
    # (color to bw conversion from cv2 does not work in this case)
    image = Image.open(image_path)
    bw_image = image.convert("L")

    # staff
    staff_job = InferenceJob(
        image=bw_image,
        model=staff_detector,
        model_type=ModelType.YOLO_DETECTION,
    )

    # noteheads
    notehead_job = InferenceJob(
        image=cv2.imread(image_path),
        model=notehead_detector,
        model_type=ModelType.YOLO_DETECTION,
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

    # INITIALIZE GRAPH
    from app.Reconstruction.Graph import NOTEHEAD_TYPE_TAG, NoteheadType, ACCIDENTAL_TYPE_TAG, AccidentalType
    from app.Reconstruction.Graph import NodeName

    prepro_def = [
        (
            NodeName.NOTEHEAD, [NOTEHEAD_TYPE_TAG],
            [
                (combined.annotations[5], [NoteheadType.FULL]),
                (combined.annotations[6], [NoteheadType.HALF])
            ]
        ),
        (
            NodeName.ACCIDENTAL, [ACCIDENTAL_TYPE_TAG],
            [
                (combined.annotations[-3], [AccidentalType.FLAT]),
                (combined.annotations[-2], [AccidentalType.NATURAL]),
                (combined.annotations[-1], [AccidentalType.SHARP])
            ]
        ),
        (
            NodeName.MEASURE, combined.annotations[1]
        ),
        (
            NodeName.GRAND_STAFF, combined.annotations[4]
        )
    ]

    start = timer()

    noteheads, accidentals, measures, grand_staffs = preprocess_annots_for_reconstruction(prepro_def)

    # RECONSTRUCT PAGE
    events = reconstruct_note_events(
        measures,
        grand_staffs,
        noteheads + accidentals,
        image_path=Path(image_path),
        neiou_threshold=0.4,
        verbose=args.verbose,
        visualize=args.visualize
    )
    time_spent_reconstruction += timer() - start

    from app.Reconstruction.VizUtils import visualize_result

    visualize_result(Path(image_path), measures, [ob for group in events for ob in group.children()], grand_staffs)

if len(images_to_process) > 0:
    print()
    print(f"Average time spent inference: {time_spent_inference / len(images_to_process)}")
    print(f"Average time spent reconstruction: {time_spent_reconstruction / len(images_to_process)}")
