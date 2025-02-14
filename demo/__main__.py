import argparse
from pathlib import Path
from timeit import default_timer as timer

import cv2
from PIL import Image

from app.Download import get_path_to_latest_version, update_models, OLA_TAG, NOTA_TAG
from app.Download import update_demo_images, load_demo_images
from app.Inference import InferenceJob, SplitSettings, run_multiple_prediction_jobs
from app.Inference.ModelWrappers import YOLODetectionModelWrapper
from app.Reconstruction import preprocess_annots_for_reconstruction, reconstruct_note_events

from app.Reconstruction.PageReconstruction import linearize_note_events_to_lmx
from app.Linearize.lmx_to_musicxml import lmx_to_musicxml

parser = argparse.ArgumentParser(
    prog="Notehead experiments demo"
)

parser.add_argument("-i", "--image_path", type=str, help="Path to image")
parser.add_argument("-o", "--output_dir", type=str, help="Path to output directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
parser.add_argument("--visualize", action="store_true", help="Visualize every step of inference")

args = parser.parse_args()

# LOAD MODELS
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
        model_wrapper=staff_detector,
        # retrieve only measures and grand staffs
        wanted_ids=[1, 4]
    )

    # noteheads
    notehead_job = InferenceJob(
        image=cv2.imread(image_path),
        model_wrapper=notehead_detector,
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

    if args.verbose:
        print(f"Class names: {', '.join(combined.class_names)}")
        print()

    # INITIALIZE GRAPH
    from app.Reconstruction.Graph import NOTEHEAD_TYPE_TAG, NoteheadType, ACCIDENTAL_TYPE_TAG, AccidentalType
    from app.Reconstruction.Graph import NodeName

    prepro_def = [
        (
            NodeName.MEASURE, combined.annotations[0]
        ),
        (
            NodeName.GRAND_STAFF, combined.annotations[1]
        ),
        (
            NodeName.NOTEHEAD, [NOTEHEAD_TYPE_TAG],
            [
                (combined.annotations[2], [NoteheadType.FULL]),
                (combined.annotations[3], [NoteheadType.HALF])
            ]
        ),
        (
            NodeName.ACCIDENTAL, [ACCIDENTAL_TYPE_TAG],
            [
                (combined.annotations[4], [AccidentalType.FLAT]),
                (combined.annotations[5], [AccidentalType.NATURAL]),
                (combined.annotations[6], [AccidentalType.SHARP])
            ]
        ),
    ]

    start = timer()

    measures, grand_staffs, noteheads, accidentals = preprocess_annots_for_reconstruction(prepro_def)

    if len(measures) == 0:
        print("Warning: No measures were found")

    # from app.Reconstruction.VizUtils import visualize_input_data
    # visualize_input_data(
    #     image_path,
    #     measures,
    #     notehead_full=noteheads,
    #     notehead_half=accidentals
    # )

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

    # print(linearize_note_events_to_lmx(events, human_readable=False))
    if args.output_dir:
        with open(args.output_dir / (image_path.stem + ".musicxml"), "w", encoding="utf8") as f:
            predicted_lmx = linearize_note_events_to_lmx(events, human_readable=False)
            print(predicted_lmx)
            f.write(lmx_to_musicxml(predicted_lmx))


    from app.Reconstruction.VizUtils import visualize_result
    #
    visualize_result(Path(image_path), measures, [ob for row in events for group in row for ob in group.children()], grand_staffs)

if len(images_to_process) > 0:
    print()
    print(f"Average time spent inference: {time_spent_inference / len(images_to_process)}")
    print(f"Average time spent reconstruction: {time_spent_reconstruction / len(images_to_process)}")
