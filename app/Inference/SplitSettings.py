import math


class SplitSettings:
    """
    Configuration for splitting an image into tiles before inference.

    This class defines the tile size, overlap between tiles, and handling of bounding boxes
    near tile edges.

    :ivar width: Width of each tile in pixels. Cannot be set together with ``tals``.
    :ivar height: Height of each tile in pixels. Cannot be set together with ``tals``.
    :ivar tals: Number of "tiles along the longer side".
        Number of tiles spanning the longer side of the image.
        If set, ``width`` and ``height`` are computed dynamically.
    :ivar overlap_ratio: Minimum horizontal/vertical overlap between tiles when splitting.
    :ivar iou_threshold: Lower bound for overlaps of bounding boxes in stitched tiles
        to be further resolved.
    :ivar edge_offset: Determines the distance from tile edges
        within which bounding boxes are discarded.
    """

    def __init__(
            self,
            width: int = None,
            height: int = None,
            tals: int = None,
            overlap_ratio: float = 0.10,
            iou_threshold: float = 0.25,
            edge_offset_ratio: float = 0.04
    ):
        """
        .. note::
            ``width`` and ``height`` cannot be set at the same time as ``tals``. If ``tals`` is provided,
            ``width`` and ``height`` will be automatically calculated and the given values will be overridden
            to ensure the specified number of tiles covers the longer side of the image.

        :param width: width of each tile in pixels, default is 640
        :param height: height of each tile in pixels, default is 640
        :param tals: number of "tiles along the longer side"
        :param overlap_ratio: minimum horizontal/vertical overlap ratio between tiles when splitting
        :param iou_threshold: minimum IoU for overlaps of bounding boxes in stitched tiles
            to be further resolved
        :param edge_offset_ratio: ratio used to computer ``edge_offset``
        """
        if (width is not None or height is not None) and tals is not None:
            print("Warning: `width` and `height` are set at the same time as `tals`,",
                  "`width` and `height` will be overridden.")

        if width is None:
            width = 640
        if height is None:
            height = 640

        self.width = width
        self.height = height
        self.tals = tals
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold
        self.edge_offset = round((width + height) / 2 * edge_offset_ratio)

    def update_window_size_based_on_tals(self, longer_side_px: int):
        """
        Updates tile size based on given longer side of an image, only if ``tals`` is set.

        :param longer_side_px: longer side of an image in pixels
        """
        if self.tals is not None:
            tiles_width = math.ceil(longer_side_px / (self.tals * (1 - self.overlap_ratio) - self.overlap_ratio))
            self.width, self.height = tiles_width, tiles_width
