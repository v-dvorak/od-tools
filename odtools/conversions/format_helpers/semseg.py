from typing import TYPE_CHECKING
from collections.abc import Iterable
from pathlib import Path
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from ...annotations.full_page import FullPage
    from ...annotations.annotation import Annotation


class _SemanticSegmentationHelper:
    @staticmethod
    def save_semantic_segmentation(
        page: "FullPage",
        output_path: Path,
        with_confidence: bool,
    ) -> None:
        """
        If there are two or three classes of annotations,
        they are put into a single RGB mask, in other cases
        all the annotations are squished into a single BW mask.
        """

        if len(page.class_names) not in {2, 3}:
            annotations = page.all_annotations()
            comp_mask = _SemanticSegmentationHelper.composite_mask(
                annotations, page.height, page.width
            )
            _SemanticSegmentationHelper.save_mask(
                comp_mask, output_path, multiple=False
            )
        else:
            comps = []
            for a in page.annotations:
                comps.append(
                    _SemanticSegmentationHelper.composite_mask(
                        a, page.height, page.width
                    )
                )
            for _ in range(3 - len(comps)):
                comps.append(
                    _SemanticSegmentationHelper.composite_mask(
                        [], page.height, page.width
                    )
                )
            _SemanticSegmentationHelper.save_mask(
                np.stack(comps, axis=-1), output_path, multiple=True
            )

    @staticmethod
    def composite_mask(
        annotations: Iterable["Annotation"],
        canvas_height: int = 512,
        canvas_width: int = 512,
    ) -> np.ndarray:
        """
        Pastes each annotation's mask onto a (canvas_size, canvas_size)
        uint8 canvas at (top, left), OR-ing
        overlapping regions together. Crops that would run past the canvas edge
        are clipped rather than raising an error.

        Returns: full_size_binary_mask (uint8 array, 0/255)
        """
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        for a in annotations:
            assert isinstance(a.mask, np.ndarray)
            obj_mask = a.mask
            top, left = a.bbox.top, a.bbox.left
            h, w = obj_mask.shape

            # Clip the crop + placement so it never runs off the canvas edge
            top_clip = max(0, -top)
            left_clip = max(0, -left)
            top_dst = max(0, top)
            left_dst = max(0, left)
            bottom_dst = min(canvas_height, top + h)
            right_dst = min(canvas_width, left + w)

            if bottom_dst <= top_dst or right_dst <= left_dst:
                continue  # entirely off-canvas, skip

            h_clip = bottom_dst - top_dst
            w_clip = right_dst - left_dst
            cropped_obj = obj_mask[
                top_clip : top_clip + h_clip, left_clip : left_clip + w_clip
            ]

            canvas[top_dst:bottom_dst, left_dst:right_dst] |= cropped_obj

        return canvas.astype(np.uint8) * 255

    @staticmethod
    def save_mask(
        mask_array: np.ndarray, output_path: Path, multiple: bool = False
    ) -> None:
        if multiple:
            mode = "RGB"
        else:
            mode = "L"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask_array, mode=mode).save(output_path)
