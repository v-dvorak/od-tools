class SplitSettings:
    def __init__(
            self,
            width: int = 640,
            height: int = 640,
            overlap_ratio: float = 0.10,
            iou_threshold: float = 0.25,
            edge_offset_ratio: float = 0.04
    ):
        self.width = width
        self.height = height
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold
        self.edge_offset = round((width + height) / 2 * edge_offset_ratio)
