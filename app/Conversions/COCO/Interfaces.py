class ICOCOAnnotation:
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int,
                 segmentation: list[tuple[int, int]]):
        self.class_id = class_id
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.segmentation = segmentation

    def __str__(self):
        return f"({self.class_id=}, {self.left}, {self.top}, {self.width}, {self.height}, {self.segmentation})"


class ICOCOFullPage:
    def __init__(
            self,
            image_size: tuple[int, int],
            annotations: list[ICOCOAnnotation],
            class_names: list[str]
    ):
        """
        :param image_size: image size, (width, height)
        :param annotations: list of COCOAnnotation
        :param class_names: list of class names
        """
        self.size = image_size
        self.classes = class_names
        self.annotations: list[list[ICOCOAnnotation]] = None
        self._sort_annotations_by_class(annotations, len(class_names))

    def _sort_annotations_by_class(self, annotations: list[ICOCOAnnotation], class_count: int):
        self.annotations = [[] for _ in range(class_count)]
        for annot in annotations:
            self.annotations[annot.class_id].append(annot)

    def __str__(self):
        return f"({self.classes=}, {self.size=}, {self.annotations})"
