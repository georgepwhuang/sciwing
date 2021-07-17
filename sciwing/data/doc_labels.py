from sciwing.data.label import Label
from typing import List


class DocLabels:
    def __init__(
        self, labels: List[str]
    ):
        self.labels = [Label(text=label) for label in labels]
