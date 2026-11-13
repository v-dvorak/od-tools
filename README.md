# Object Detection Tools (ODT)

ODT is a Python library and command-line tool for working with object detection annotations in optical music recognition (OMR). It covers the parts of an OMR experiment pipeline that sit around the model itself:

- converting annotations between formats (MuNG, COCO, YOLO, semantic segmentation),
- splitting large scans into fixed-size tiles for training and inference,
- stitching tile predictions back into full-page annotations.

It was developed as infrastructure for the [TonIC](https://github.com/v-dvorak/tonic) pipeline and for experiments on the [Musicorpus](https://github.com/OmniOMR/musicorpus) dataset. Written in Python ≥ 3.11.

## Installation

```bash
pip install -e .
```

For development install `pytest`, `ruff`, `mypy` (`pip install -e ".[dev]"`).

## Command-line usage

All functionality is available through `python -m odtools <command>`:

| Command | Purpose |
|-|-|
| `form` | Convert a dataset between annotation formats, optionally resizing images, splitting them into tiles, and applying zoom augmentation. |
| `split` | Split a dataset into train/test directories (no other processing). |
| `stats` | Compute and plot dataset statistics (class frequencies, bounding box heatmaps). |
| `eval` | Evaluate stored predictions against ground truth using COCO metrics, with optional bounding box rescaling. |
| `confcheck` | Validate a config file. |

Run `python -m odtools <command> --help` for the full list of arguments.

Most commands take a `--config` file that maps class IDs to class names and sets tile size and overlap. The format is described in [configs/README.md](configs/README.md); see the files in [configs/](configs/) for examples (JSON and YAML).

### Example: formatting a Musicorpus dataset

Datasets in the Musicorpus format are converted with the `form` job. Pass the dataset root as `image_dir` and add the `--musicorpus` flag; every split the dataset defines (train, validation, test and any additional splits) is processed into its own subdirectory of the output:

```bash
python -m odtools form datasets/my-converted-dataset path-to/musicorpus-source-dataset \
    --musicorpus \
    -i mung \
    -o yolod \
    --config my-conversion-config.json
```

An alternative splits file can be selected with `--splits_variant`. Image splitting (slicing images into tiles of set size with set overlap) can be selected with `--image_splitting`.

## Library usage

### Annotation conversions

Supported formats:

- [MuNG](https://github.com/OMR-Research/mung)
- [COCO](https://cocodataset.org/#home)
- [YOLO detection](https://docs.ultralytics.com/datasets/detect/) (YOLOd)
- [YOLO segmentation](https://docs.ultralytics.com/datasets/segment/) (YOLOs)
- Semantic segmentation (SemSeg) [What is semantic segmentation?](https://roboflow.com/blog/what-is-semantic-segmentation)

Supported conversions:

| From \ To | MuNG | COCO | YOLOd | YOLOs | SemSeg |
|-|-|-|-|-|-|
| **MuNG**  | ☑️ | ✅ | ✅ | ✅ | ✅ |
| **COCO**  | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YOLOd** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YOLOs** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **SemSeg** | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ |

> ☑️ - With loss of information (inlinks, outlinks).
>
> ✖️ - Semantic segmentation does not encode instances, it is not possible to convert it to instance-based formats.

### Image splitting and stitching

When a model has a fixed input resolution, objects that are small relative to the page (noteheads in particular) may become undetectable after downscaling. ODT splits images into overlapping tiles, runs detection per tile, and merges the results back into a single annotation set for the full page - the merging removes detections cut off at tile edges and deduplicates overlapping detections via IoU.

![](docs/splitviz/unique-viz.jpg)

## Tests and Formatting

First install development dependencies `pip install -e ".[dev]"`.

```bash
python -m pytest
python -m ruff format
```

## Acknowledgment

This package was developed and first used for WoRMS 2024 paper by Dvorak et al., [Staff Layout Analysis using the YOLO Platform](https://arxiv.org/pdf/2411.15741#page=19). The repository has undergone multiple significant changes since then.

## Citation

If you use this software in your research in this current version, please cite this:

```bibtex
@mastersthesis{dvorak2025fast,
  author = {Dvořák, Vojtěch},
  title = {Fast Optical Music Recognition Using the YOLO Platform},
  school = {Charles University, Faculty of Mathematics and Physics},
  year = {2025},
  address = {Prague},
  type = {Bachelor's thesis},
  department = {Institute of Formal and Applied Linguistics},
  advisor = {Mayer, Jiří},
  url = {http://hdl.handle.net/20.500.11956/200867}
}
```

## Contact

<img src="https://ufal.mff.cuni.cz/~hajicj/2024/images/logo-large.png" width="600px" alt="PMCG logo">

Developed and maintained by [Vojtěch Dvořák](https://github.com/v-dvorak) ([dvorak@ufal.mff.cuni.cz](mailto:dvorak@ufal.mff.cuni.cz)) as part of the [Prague Music Computing Group](https://ufal.mff.cuni.cz/pmcg) lead by [Jan Hajič jr.](https://ufal.mff.cuni.cz/jan-hajic-jr) ([hajicj@ufal.mff.cuni.cz](mailto:hajicj@ufal.mff.cuni.cz)).
