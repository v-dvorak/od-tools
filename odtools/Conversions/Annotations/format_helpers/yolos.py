
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
import cv2

from ..Annotation import Annotation
from ..annotation_type import AnnotationType
from ... import ConversionUtils
if TYPE_CHECKING:
    from ..FullPage import FullPage


class _YOLOSegmentationHelper:
    @staticmethod
    def from_yolo_segmentation(
            annot_path: Path,
            image_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> "FullPage":
        from ..FullPage import FullPage
        image_width, image_height = ConversionUtils.get_num_pixels(image_path)
        annots = []

        with open(annot_path, "r") as file:
            for line in file:
                annots.append(_YOLOSegmentationHelper._parse_single_line_yolo_segmentation(
                    line,
                    image_width,
                    image_height,
                    an_type=an_type
                ))

        return FullPage.from_list_of_coco_annotations(
            (image_width, image_height),
            annots,
            class_output_names
        )

    @staticmethod
    def _parse_single_line_yolo_segmentation(
            line: str,
            image_width: int,
            image_height: int,
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> Annotation:
        parts = line.strip().split()
        assert len(parts) > 2 and len(parts) % 2 == 1

        class_id = int(parts[0])

        segm = []
        i = 0
        coords = parts[1:]
        while i + 1 < len(coords):
            x, y = int(float(coords[i]) * image_width), int(float(coords[i + 1]) * image_height)
            segm.append((x, y))
            i += 2

        l, t, w, h = Annotation.bounding_box_from_segmentation(segm)

        return Annotation(class_id, l, t, w, h, segm, an_type=an_type)

    @staticmethod
    def save_yolo_segmentation(
            page: "FullPage",
            output_path: Path,
            with_confidence: bool,
            type_: Literal["convex_hull", "contours", "contours_holes", "npz"] = "contours_holes"
    ) -> None:
            if type_ == "npz":
                records = []
                annotations = page.all_annotations()
                    # assert annotation.mask is not None
                    # records.append({
                    #     "class_id": annotation.class_id,
                    #     "top":      annotation.bbox.top,
                    #     "left":     annotation.bbox.left,
                    #     "mask":     np.array(annotation.mask, dtype=np.uint8),
                    # })
                arrays = {}
                count = 0
                for i, ann in enumerate(annotations):
                    assert ann.mask is not None, f"Annotation {i} has no mask"
                    arrays[f"class_id_{i}"] = np.array(ann.class_id, dtype=np.int32)
                    arrays[f"top_{i}"]      = np.array(ann.bbox.top,  dtype=np.int32)
                    arrays[f"left_{i}"]     = np.array(ann.bbox.left, dtype=np.int32)
                    arrays[f"mask_{i}"]     = np.asarray(ann.mask, dtype=np.uint8)
                    count += 1
                
                np.savez_compressed(
                    output_path.with_suffix(".npz"),
                    count=np.array(count, dtype=np.int32),
                    img_w=np.array(page.width,  dtype=np.int32),
                    img_h=np.array(page.height, dtype=np.int32),
                    **arrays,
                )

                with open(output_path, "w") as file:
                    for a in page.all_annotations():
                        # assert annotation.mask is not None
                        
                        dumb_mask = np.array([
                            [a.bbox.left, a.bbox.top],
                            [a.bbox.right, a.bbox.top],
                            [a.bbox.right, a.bbox.bottom],
                            [a.bbox.left, a.bbox.bottom],
                        ], np.float64)
                        dumb_mask[:, 0] /= page.width
                        dumb_mask[:, 1] /= page.height
                        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in dumb_mask)
                        file.write(f"{a.class_id} {coords_str}\n")
                return
            
            with open(output_path, "w") as file:
                for annotation in page.all_annotations():
                    assert annotation.mask is not None
                    if type_ == "convex_hull":
                        file.write(_YOLOSegmentationHelper._convex_hull_from_mask(
                            annotation.mask,
                            annotation.bbox.top,
                            annotation.bbox.left,
                            annotation.class_id,
                            page.width,
                            page.height
                        ))
                        file.write("\n")
                    elif type_ == "contours":
                        contour = _YOLOSegmentationHelper._contours_from_mask(
                            annotation.mask,
                            annotation.bbox.top,
                            annotation.bbox.left,
                        )
                        normalized = contour.copy()
                        normalized[:, 0] /= page.width
                        normalized[:, 1] /= page.height
                        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
                        file.write(f"{annotation.class_id} {coords_str}\n")
                    elif type_ == "contours_holes":
                        contour = binary_mask_to_single_yolo_polygon(
                            annotation.mask,
                            page.width,
                            page.height,
                            annotation.bbox.left,
                            annotation.bbox.top,
                            )
                        # contour = _YOLOSegmentationHelper._contours_from_mask_with_holes(
                        #     annotation.mask,
                        #     annotation.bbox.top,
                        #     annotation.bbox.left,
                        # )
                        # normalized = contour.copy()
                        # normalized[:, 0] /= page.width
                        # normalized[:, 1] /= page.height
                        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in contour)
                        file.write(f"{annotation.class_id} {coords_str}\n")

                    else:
                        raise ValueError(f"Unknown mask conversion type '{type_}")

    @staticmethod
    def _convex_hull_from_mask(mask_2d: np.ndarray, top, left, class_id, img_width, img_height):
        """
        Convert a 2D binary mask to YOLO segmentation format.
        Holes are ignored - only the outer convex hull is returned.

        :param mask_2d: 2D list/array of 0s and 1s, shape (height, width)
        :param top, left: position of the mask's top-left corner in the full image
        :param class_id: integer class index
        :param img_width, img_height: full image dimensions for normalization

        :return: YOLO segmentation string: "<class_id> x1 y1 x2 y2 ... xn yn"
        """
        mask_arr = np.array(mask_2d, dtype=np.uint8)
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No foreground pixels found in mask.")
        
        # Take the largest contour and compute its convex hull
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        
        # hull shape is (N, 1, 2); flatten to list of (x, y) in image space
        points = [(left + x + 0.5, top + y + 0.5) for [[x, y]] in hull]
        
        normalized = [(x / img_width, y / img_height) for x, y in points]
        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
        return f"{class_id} {coords_str}"
    
    @staticmethod
    def _contours_from_mask(mask_2d: np.ndarray, top: int, left: int):
        mask_arr = np.array(mask_2d, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(
            mask_arr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if not contours or hierarchy is None:
            raise ValueError("No foreground pixels found in mask.")

        hierarchy = hierarchy[0]

        merged = []
        for i, h in enumerate(hierarchy):
            if h[3] != -1:
                continue

            parent = contours[i]

            child_idx = h[2]
            while child_idx >= 0:
                hole = contours[child_idx]

                if _YOLOSegmentationHelper._is_clockwise(parent):
                    parent = parent[::-1]
                if not _YOLOSegmentationHelper._is_clockwise(hole):
                    hole = hole[::-1]

                pi, hi = _YOLOSegmentationHelper._closest_point_pair(parent, hole)
                parent = _YOLOSegmentationHelper._stitch(parent, hole, pi, hi)

                child_idx = hierarchy[child_idx][0]

            merged.append(parent)

        while len(merged) > 1:
            pi, qi = _YOLOSegmentationHelper._closest_point_pair(merged[0], merged[1])
            merged[0] = _YOLOSegmentationHelper._stitch(merged[0], merged[1], pi, qi)
            merged.pop(1)

        pts = merged[0][:, 0, :].astype(float)
        pts[:, 0] += left
        pts[:, 1] += top
        return pts


    @staticmethod
    def _contours_from_mask_with_holes(mask_2d: np.ndarray, top: int, left: int):
        mask_arr = np.array(mask_2d, dtype=np.uint8)
        h, w = mask_arr.shape
        padded = np.pad(mask_arr, 1, constant_values=0)

        # Use a list per corner to handle ambiguous corners where two loops
        # share the same vertex (checkerboard diagonal touch case)
        from collections import defaultdict
        edge_map = defaultdict(list)

        rows, cols = np.where(mask_arr == 1)
        for r, c in zip(rows.tolist(), cols.tolist()):
            pr, pc = r + 1, c + 1

            if padded[pr - 1, pc] == 0:
                edge_map[(pr, pc)].append((0, 1))
            if padded[pr + 1, pc] == 0:
                edge_map[(pr + 1, pc + 1)].append((0, -1))
            if padded[pr, pc - 1] == 0:
                edge_map[(pr + 1, pc)].append((-1, 0))
            if padded[pr, pc + 1] == 0:
                edge_map[(pr, pc + 1)].append((1, 0))

        if not edge_map:
            raise ValueError("No foreground pixels found in mask.")

        def extract_loops(edges):
            # make mutable stacks
            remaining = {k: list(v) for k, v in edges.items()}

            def pop_edge(corner):
                val = remaining[corner].pop()
                if not remaining[corner]:
                    del remaining[corner]
                return val

            loops = []
            while remaining:
                start = next(iter(remaining))
                loop = [start]
                cur = start
                while True:
                    dy, dx = pop_edge(cur)
                    nxt = (cur[0] + dy, cur[1] + dx)
                    if nxt == start:
                        break
                    loop.append(nxt)
                    cur = nxt
                loops.append(loop)
            return loops

        loops = extract_loops(edge_map)

        def to_image_space(loop):
            return np.array(
                [[[left + vc - 1, top + vr - 1]] for vr, vc in loop],
                dtype=np.float32
            )

        contours = [to_image_space(l) for l in loops]

        def signed_area(pts):
            p = pts[:, 0, :]
            r = np.roll(p, -1, axis=0)
            return 0.5 * np.sum((r[:, 0] - p[:, 0]) * (r[:, 1] + p[:, 1]))

        outers = [c for c in contours if signed_area(c) <= 0]
        holes  = [c for c in contours if signed_area(c) >  0]

        for hole in holes:
            dists = [np.min(((o[:, 0, None, :] - hole[None, :, 0, :]) ** 2).sum(-1)) for o in outers]
            best = outers[int(np.argmin(dists))]
            pi, hi = _YOLOSegmentationHelper._closest_point_pair(best, hole)
            stitched = _YOLOSegmentationHelper._stitch(best, hole, pi, hi)
            outers[outers.index(best)] = stitched

        while len(outers) > 1:
            pi, qi = _YOLOSegmentationHelper._closest_point_pair(outers[0], outers[1])
            outers[0] = _YOLOSegmentationHelper._stitch(outers[0], outers[1], pi, qi)
            outers.pop(1)

        return outers[0][:, 0, :]
    @staticmethod
    def _is_clockwise(contour):
        pts = contour[:, 0, :]
        # shoelace sign
        rolled = np.roll(pts, -1, axis=0)
        return np.sum((rolled[:, 0] - pts[:, 0]) * (rolled[:, 1] + pts[:, 1])) < 0

    @staticmethod
    def _closest_point_pair(c1, c2):
        """Return (i, j) indices of the closest point between two contours."""
        p1 = c1[:, 0, :]  # (N, 2)
        p2 = c2[:, 0, :]  # (M, 2)
        dists = ((p1[:, None, :] - p2[None, :, :]) ** 2).sum(-1)  # (N, M)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        return int(i), int(j)

    @staticmethod
    def _stitch(c1, c2, i, j):
        """
        Merge c2 into c1 by cutting at index i (c1) and j (c2).
        Duplicates the junction points to form a zero-area seam.
        """
        c2_rolled = np.roll(c2, -j, axis=0)
        return np.concatenate([
            c1[:i + 1],
            c2_rolled,
            c2_rolled[:1],   # bridge back
            c1[i:],          # resume c1 (i is duplicated - valid in YOLO)
        ])
    
"""
mask_to_polygon.py — Binary mask → single stitched YOLO polygon.
ALL components (outers, holes, disjoint islands) are connected via
zero-area bridges.  Returns exactly one (N,2) polygon per mask.
"""

from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np


def binary_mask_to_single_yolo_polygon(
    mask: np.ndarray,
    img_width: int,
    img_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
    normalize: bool = True,
    simplify_epsilon: float | None = None,
) -> np.ndarray | None:
    """
    Convert a binary mask into exactly ONE YOLO polygon.

    Returns:
        (N,2) float32, normalized to [0,1] if normalize=True,
        or None if the mask is empty.
    """
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask_arr.shape}")
    if not np.any(mask_arr):
        return None

    h, w = mask_arr.shape

    # ── Edge map (handles checkerboard diagonal-touch correctly) ───────
    edge_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    padded = np.pad(mask_arr, 1, mode="constant", constant_values=0)
    fg_rows, fg_cols = np.where(mask_arr == 1)

    for r, c in zip(fg_rows.tolist(), fg_cols.tolist()):
        pr, pc = r + 1, c + 1
        if padded[pr - 1, pc] == 0:
            edge_map[(pr, pc)].append((0, 1))
        if padded[pr + 1, pc] == 0:
            edge_map[(pr + 1, pc + 1)].append((0, -1))
        if padded[pr, pc - 1] == 0:
            edge_map[(pr + 1, pc)].append((-1, 0))
        if padded[pr, pc + 1] == 0:
            edge_map[(pr, pc + 1)].append((1, 0))

    if not edge_map:
        return None

    # ── Extract all loops ─────────────────────────────────────────────
    remaining = {k: list(v) for k, v in edge_map.items()}
    all_loops: List[List[Tuple[int, int]]] = []

    while remaining:
        start = next(iter(remaining))
        loop = [start]
        cur = start
        while True:
            edges = remaining[cur]
            dy, dx = edges.pop()
            if not edges:
                del remaining[cur]
            nxt = (cur[0] + dy, cur[1] + dx)
            if nxt == start:
                break
            loop.append(nxt)
            cur = nxt
            if nxt not in remaining and nxt != start:
                break
        all_loops.append(loop)

    # ── Convert loops → contour arrays (N,1,2) in image coords ────────
    contours: List[np.ndarray] = []
    for loop in all_loops:
        pts = np.array(
            [[[offset_x + vc - 1, offset_y + vr - 1]] for vr, vc in loop],
            dtype=np.float32,
        )
        contours.append(pts)

    # ── Stitch EVERYTHING into one polygon ────────────────────────────
    merged = _stitch_all_into_one(contours)
    if merged is None or len(merged) == 0:
        return None

    # Flatten (N,1,2) → (N,2)
    polygon_2d = merged[:, 0, :].astype(np.float32)

    # Optional simplification
    if simplify_epsilon is not None and simplify_epsilon > 0:
        import cv2
        polygon_2d = cv2.approxPolyDP(
            polygon_2d.reshape(-1, 1, 2).astype(np.float32),
            epsilon=simplify_epsilon,
            closed=True,
        )[:, 0, :]

    if normalize:
        polygon_2d[:, 0] /= img_width
        polygon_2d[:, 1] /= img_height
        polygon_2d = np.clip(polygon_2d, 0.0, 1.0)

    return polygon_2d


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


def _stitch_all_into_one(contours: List[np.ndarray]) -> np.ndarray | None:
    """
    Stitch all contours (outers, holes, disjoint components) into one (N,1,2).

    Steps:
      1) Classify: clockwise → outer, counter‑clockwise → hole.
      2) Stitch each hole into its nearest outer.
      3) Stitch all remaining outers together into a single polygon.
    """
    if not contours:
        return None
    if len(contours) == 1:
        return contours[0]

    outers: List[np.ndarray] = []
    holes:  List[np.ndarray] = []

    for c in contours:
        if _signed_area(c) <= 0:
            outers.append(c)
        else:
            holes.append(c)

    # Fallback: all holes, no outers → promote the largest hole
    if not outers and holes:
        holes.sort(key=lambda c: abs(_signed_area(c)), reverse=True)
        outers.append(holes.pop(0))

    # Holes first
    for hole in holes:
        best_idx, best_i, best_j = _find_nearest(hole, outers)
        outers[best_idx] = _bridge_stitch(outers[best_idx], hole, best_i, best_j)

    # Then all outers into one
    while len(outers) > 1:
        last = outers.pop()
        best_idx, best_i, best_j = _find_nearest(last, outers)
        outers[best_idx] = _bridge_stitch(outers[best_idx], last, best_i, best_j)

    return outers[0]


def _signed_area(contour: np.ndarray) -> float:
    """Shoelace: ≤ 0 = outer, > 0 = hole (image y‑down coords)."""
    pts = contour[:, 0, :] if contour.ndim == 3 else contour
    rolled = np.roll(pts, -1, axis=0)
    return 0.5 * np.sum(rolled[:, 0] * pts[:, 1] - pts[:, 0] * rolled[:, 1])


def _find_nearest(
    query: np.ndarray,
    candidates: List[np.ndarray],
) -> Tuple[int, int, int]:
    """Return (candidate_index, i_in_candidate, j_in_query) of closest pair."""
    q_pts = query[:, 0, :]
    best_idx = 0
    best_dist = float("inf")
    best_i = best_j = 0

    for idx, cand in enumerate(candidates):
        c_pts = cand[:, 0, :]
        sq_dists = np.sum((c_pts[:, None, :] - q_pts[None, :, :]) ** 2, axis=-1)
        i, j = np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
        d = sq_dists[i, j]
        if d < best_dist:
            best_dist = d
            best_idx = idx
            best_i, best_j = int(i), int(j)

    return best_idx, best_i, best_j


def _bridge_stitch(
    base: np.ndarray,
    other: np.ndarray,
    i: int,
    j: int,
) -> np.ndarray:
    """
    Merge `other` into `base` at indices i (base) and j (other)
    with a zero‑area bridge: base[i] → other → base[i].

    Sequence:  base[:i+1] → other[j:] → other[:j+1] → base[i:]
    """
    other_rolled = np.roll(other, -j, axis=0)

    return np.concatenate(
        [
            base[: i + 1],
            other_rolled,
            other_rolled[:1],   # duplicate anchor for zero-area bridge
            base[i:],
        ],
        axis=0,
    )