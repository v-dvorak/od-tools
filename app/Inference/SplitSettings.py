import math


class SplitSettings:
    """
    Stores data for image splitting into tiles.


    :ivar width: window width
    :ivar height: window height
    :ivar tols: stands for number of "tiles over longer side",
        width and height of a tile are computed according to it
        so that there are exactly `tols` windows needed to cover the longer side of the image
    :ivar overlap_ratio: minimum horizontal/vertical overlap between tiles when splitting
    :ivar iou_threshold: lower bound for overlaps of bounding boxes in stitched tiles
        to be further resolved
    :ivar edge_offset: bounding boxes that lie too close to the edge
        (their distance from the edge is `edge_offset` or less) of the tile are discarded
    """

    def __init__(
            self,
            width: int = None,
            height: int = None,
            tols: int = None,
            overlap_ratio: float = 0.10,
            iou_threshold: float = 0.25,
            edge_offset_ratio: float = 0.04
    ):
        """
        `width` and `height` cannot be set at the same time as `tols`.

        :param width: window width, default is 640 px
        :param height: window height, default is 640 px
        :param tols: number of "tiles over longer side"
        :ivar overlap_ratio: minimum horizontal/vertical overlap between tiles when splitting
        :ivar iou_threshold: lower bound for overlaps of bounding boxes in stitched tiles
            to be further resolved
        :ivar edge_offset_ratio: for computing `edge_offset`
        """
        if (width is not None or height is not None) and tols is not None:
            raise ValueError("`width` and `height` cannot be set at the same time as `tols`.")
        if width is None:
            width = 640
        if height is None:
            height = 640

        self.width = width
        self.height = height
        self.tols = tols
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold
        self.edge_offset = round((width + height) / 2 * edge_offset_ratio)

    def update_window_size_based_on_tols(self, longer_side_px: int):
        """
        Updates tile size based on given longer side of an image, only if `tols` is set.

        :param longer_side_px: longer side of an image in pixels
        """
        if self.tols is not None:
            tiles_width = math.ceil(longer_side_px / (self.tols * (1 - self.overlap_ratio) - self.overlap_ratio))
            self.width, self.height = tiles_width, tiles_width
