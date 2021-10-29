import dataclasses
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt


@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float
    name: str = ""
    score: Optional[float] = None

    @classmethod
    def none(cls) -> "Point":
        return Point(0.0, 0.0, "", 0.0)

    def scale(self, x: float, y: float) -> "Point":
        return Point(self.x * x, self.y * y, self.name, self.score)

    def translate(self, x: float, y: float) -> "Point":
        return Point(self.x + x, self.y + y, self.name, self.score)

    def distance(self, other: "Point") -> float:
        diff = np.array(self.as_coordinates_tuple) - np.array(
            other.as_coordinates_tuple
        )
        return float(np.sqrt((diff ** 2).sum()))

    def rotate_around_origin_point(self, origin: "Point", angle: float) -> "Point":
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        point = np.array([self.as_coordinates_tuple]).T
        origin = np.array([origin.as_coordinates_tuple]).T
        rotated = (R @ (point - origin) + origin).flatten()
        return Point(rotated[0], rotated[1], self.name, self.score)

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
            "width": 0.3943217665615142,  # magic value used in label studio
            "score": self.score,
        }

    def plot(self, ax=None, color="red", marker="x", size=20, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(
            self.x,
            self.y,
            label=self.name,
            color=color,
            marker=marker,
            s=size,
            **kwargs
        )


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
    def center(self) -> "Point":
        return Point((self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2)

    def center_scale(self, x: float, y: float) -> "BBox":
        w, h = self.width * x, self.height * y
        cx, cy = self.center.x, self.center.y

        x_min = cx - w / 2
        x_max = cx + w / 2

        y_min = cy - h / 2
        y_max = cy + h / 2

        return dataclasses.replace(
            self, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
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


@dataclasses.dataclass(frozen=True)
class Line:
    start: Point
    end: Point

    @property
    def poly1d(self) -> np.poly1d:
        return np.poly1d(
            np.polyfit(
                [
                    self.start.x,
                    self.end.x,
                ],
                [self.start.y, self.end.y],
                deg=1,
            )
        )

    @property
    def length(self) -> float:
        return self.start.distance(self.end)

    def scale(self, x: float, y: float) -> "Line":
        return Line(self.start.scale(x, y), self.end.scale(x, y))

    def projection_point(self, point: Point) -> Point:
        line_fit = self.poly1d
        m = line_fit.coeffs[0]
        k = line_fit.coeffs[1]
        proj_point_x = (point.x + m * point.y - m * k) / (m ** 2 + 1)
        proj_point_y = m * proj_point_x + k
        return Point(proj_point_x, proj_point_y)

    def plot(self, ax=None, color=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(
            [self.start.x, self.end.x],
            [self.start.y, self.end.y],
            color=color,
            **kwargs
        )
