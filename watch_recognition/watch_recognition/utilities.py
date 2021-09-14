import dataclasses
from typing import Optional, Tuple

from matplotlib import pyplot as plt


@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float
    name: str = ""
    score: Optional[float] = None

    def scale(self, x: float, y: float) -> "Point":
        return Point(self.x * x, self.y * y, self.name, self.score)

    def translate(self, x: float, y: float) -> "Point":
        return Point(self.x + x, self.y + y, self.name, self.score)

    @property
    def as_coordinates_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    @property
    def as_label_studio_object(self) -> dict:
        if self.x > 1 or self.y > 1:
            raise ValueError("keypoint coordinates have to be normalized")
        return {
            "keypointlabels": [self.name],
            # label studio requires coordinates from 0-100 (aka percentage of the image)
            # upper left corner coordinates
            "x": self.x * 100,
            "y": self.y * 100,
            "width": 0.3943217665615142,
            # TODO add score?
        }


@dataclasses.dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: str
    score: Optional[float] = None

    def contains(self, point: Point) -> bool:
        contains_x = self.x_min < point.x < self.x_max
        contains_y = self.y_min < point.y < self.y_max
        return contains_x and contains_y

    def scale(self, x: float, y: float) -> "BBox":
        return BBox(
            self.x_min * x, self.y_min * y, self.x_max * x, self.y_max * y, self.name
        )

    @property
    def as_coordinates_tuple(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def as_label_studio_object(self) -> dict:
        return {
            "rectanglelabels": [self.name],
            # label studio requires coordinates from 0-100 (aka percentage of the image)
            # upper left corner coordinates
            "x": self.x_min * 100,
            "y": self.y_min * 100,
            "width": self.width * 100,
            "height": self.height * 100,
        }
