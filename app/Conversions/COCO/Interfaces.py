from typing import Self


# from ...Splitting.SplitUtils import IRectangle


class ICOCOAnnotation:
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int,
                 segmentation: list[tuple[int, int]], confidence: float = 1.0):
        self.class_id = class_id
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.segmentation = segmentation
        self.confidence = confidence

    def __str__(self):
        return f"({self.class_id=}, {self.left}, {self.top}, {self.width}, {self.height}, {self.segmentation})"

    def adjust_position_copy(self, left_shift: int, top_shift: int) -> Self:
        raise NotImplementedError()


class ICOCOFullPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            annotations: list[list[ICOCOAnnotation]],
            class_names: list[str]
    ):
        """
        Stores all subpages inside a single page (path_to_image).
        The subpages are stored in a list of lists
        where each list corresponds to single class id.

        :param image_size: path_to_image size, (width, height)
        :param annotations: list of COCOAnnotation
        :param class_names: list of class names
        """
        self.size = image_size
        self.class_names = class_names
        self.annotations: list[list[ICOCOAnnotation]] = annotations

    def __str__(self):
        return f"({self.class_names=}, {self.size=}, {self.annotations})"

    def all_annotations(self) -> list[ICOCOAnnotation]:
        for row in self.annotations:
            for annotation in row:
                yield annotation

    def annotation_count(self) -> int:
        return sum([len(self.annotations[i]) for i in range(len(self.annotations))])


class ICOCOSplitPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            subpages: list[list[ICOCOFullPage]],
            class_names: list[str],
            splits
    ):
        self.size = image_size
        self.subpages = subpages
        self.class_names = class_names
        self.splits = splits
