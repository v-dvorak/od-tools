# Object Detection Tools (ODT)

ODT is a Python library and framework designed for handling annotations used as both inputs and outputs in object detection models. It provides a streamlined way to work with detection data, making it easier to preprocess, analyze, and utilize annotations in OD workflows.

For usage details, check out [TonIC](https://github.com/v-dvorak/tonic).

## Core features

### Conversions between annotation formats

- Supported formats:
    - [MuNG](https://github.com/OMR-Research/mung)
    - [COCO](https://cocodataset.org/#home)
    - [YOLO detection](https://docs.ultralytics.com/datasets/detect/) (YOLOd)
    - [YOLO segmentation](https://docs.ultralytics.com/datasets/segment/) (YOLOs)
- Supported conversions:

| from \ to | MuNG | COCO | YOLOd | YOLOs |
|-----------|------|------|-------|-------|
| **MuNG**  | ➖    | ✅    | ✅     | ✅     |
| **COCO**  | ❌    | ➖    | ✅     | ✅     |
| **YOLOd** | ❌    | ✅    | ➖     | ✅     |
| **YOLOs** | ❌    | ✅    | ✅     | ➖     |

### Image splitting with overlaps with annotations adjusted accordingly

When a model has a fixed input resolution, objects that are too small in the original image may be lost due to downscaling and compression during preprocessing. This can negatively impact detection performance.

That's where image splitting and stitching comes in. The **ODT** library allows the user to specify input tile size, overlap ratio between tiles (to which the input image is split), based on these parameters, predictions are made for each tile separately and detected annotations are automatically merged and resolved as if the splitting never happened.

![](docs/splitviz/unique-viz.jpg)

### Easy-to-setup pipeline

Loaded models, transformed images, and split settings are encapsulated in an `InferenceJob`, which is passed to a method that returns the finalized predictions:

```python
import cv2
from odtools.Inference import InferenceJob, SplitSettings, run_multiple_prediction_jobs

job = InferenceJob(
    image=loaded_image,
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

result = run_multiple_prediction_jobs([job])
```

This job processes an image using a loaded model (`notehead_detector`). The image is divided into `640 × 640` pixel tiles with a `10%` overlap (`64 px`) between neighboring tiles. `wanted_ids` are used to tell the method to not bother with conversion of annotations that won't be used later.

When merging detected annotations:

- Detections too close to the edge (within `4%` of the tile’s side length) are removed to avoid partially detected objects interfering when removing duplicates.
- Annotations with an IoU greater than `0.25` are considered duplicates and are filtered out.

Example of output of multiple models merged and resolved. In this case two models were used:

- Layout detector without splitting
- Notehead detector with image splitting, noteheads are just too small to be detected after compressing the image to `640 * 640` px

![](docs/analysis-showcase.png)

### Dataset visualization

- Average number of annotations per page (with stddev):

![](docs/graphs/annot_counts.png)

- Annotation bounding box heatmaps:

![](docs/graphs/combined.png)

### Easy-to-implement model wrappers

**ODT** is designed primarily for use with YOLO models. However, it can be extended to support other architectures, such as Fast R-CNN or U-Net. To integrate a different model, users simply need to implement a custom wrapper class that inherits from the abstract class [`IModelWrapper`](odtools/Inference/ModelWrappers/ModelWrapper.py) and define its two methods: `predict_single` and `predict_multiple`.

[//]: # (### Model validation)

[//]: # (TODO:)

## Acknowledgments

The evaluation module is based on the **Object Detection Metrics** by Rafael Padilla. Available at [GitHub](https://github.com/rafaelpadilla/review_object_detection_metrics). Published as:

> Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B. A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit. Journal Electronics, 2021. doi:10.3390/electronics10030279

### Image sources

> ROLLE, Johann Heinrich, Karl Friedrich Wilhelm HERROSEE a Eduard ZACHARIÄ. Gedor, oder das Erwachen zum bessern Leben: von Herrosee, Prediger in Berlin. Leipzig: Auf Kosten der Wittwe des Autors, und in Commision bey Schwickert, 1787, p. 2. Available online at: [https://www.digitalniknihovna.cz/mzk/uuid/uuid:2d5f652c-1df0-474c-ae23-3fb699afe808](https://www.digitalniknihovna.cz/mzk/uuid/uuid:2d5f652c-1df0-474c-ae23-3fb699afe808)

> BÉRIOT, Charles Auguste de. Sonate pour deux pianos, op. 61. Paris: Hamelle, [1890?], p. 28. Available online at: [https://www.digitalniknihovna.cz/mzk/uuid/uuid:2e117f2e-4c19-4bc3-ba6b-5531ca623e22](https://www.digitalniknihovna.cz/mzk/uuid/uuid:2e117f2e-4c19-4bc3-ba6b-5531ca623e22)

## Contact

<img src="https://ufal.mff.cuni.cz/~hajicj/2024/images/logo-large.png" width="600px" alt="PMCG logo">

Developed and maintained by [Vojtěch Dvořák](https://github.com/v-dvorak) ([dvorak@ufal.mff.cuni.cz](mailto:dvorak@ufal.mff.cuni.cz)) as part of the [Prague Music Computing Group](https://ufal.mff.cuni.cz/pmcg) lead by [Jan Hajič jr.](https://ufal.mff.cuni.cz/jan-hajic-jr) ([hajicj@ufal.mff.cuni.cz](mailto:hajicj@ufal.mff.cuni.cz)).
