from pathlib import Path

from ..Linearize.LMXWrapper import LMXWrapper
from ..Linearize.MXMLSimplifier import complex_musicxml_file_to_lmx

ground_truth_dir = Path("datasets/valdata/mxml/val")
predicted_dir = Path("predicted")

counter = 1
for ground_truth, predicted in zip(ground_truth_dir.rglob("*.musicxml"), predicted_dir.rglob("*.musicxml")):
    print(f">>> {counter}")
    print(f"Validating {ground_truth.name}")
    try:
        p_lmx = LMXWrapper.from_musicxml_file(predicted)
        gt_lmx = complex_musicxml_file_to_lmx(ground_truth)
        print(LMXWrapper.normalized_levenstein_distance(p_lmx, gt_lmx))
    except Exception as e:
        print(e)
    print()
    counter += 1
