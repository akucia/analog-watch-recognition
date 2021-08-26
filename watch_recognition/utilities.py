import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float
    name: str

    def scale(self, x: float, y: float) -> "Point":
        return Point(self.x * x, self.y * y, self.name)

    def translate(self, x: float, y: float) -> "Point":
        return Point(self.x + x, self.y + y, self.name)


@dataclasses.dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: str

    def contains(self, point: Point) -> bool:
        contains_x = self.x_min < point.x < self.x_max
        contains_y = self.y_min < point.y < self.y_max
        return contains_x and contains_y

    def scale(self, x: float, y: float) -> "BBox":
        return BBox(
            self.x_min * x, self.y_min * y, self.x_max * x, self.y_max * y, self.name
        )

    def astuple(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max
