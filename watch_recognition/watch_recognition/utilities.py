import dataclasses
from itertools import combinations
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import line as draw_line
from skimage.measure import approximate_polygon, find_contours, label, regionprops


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
    def as_array(self) -> np.ndarray:
        return np.array(self.as_coordinates_tuple)

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
    score: float = 0

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
    def slope(self) -> float:
        return self.poly1d.coeffs[0]

    @property
    def unit_vector(self) -> np.ndarray:
        vector = self.end.as_array - self.start.as_array
        return vector / np.linalg.norm(vector)

    @property
    def center(self) -> Point:
        x = (self.start.x + self.end.x) / 2
        y = (self.start.y + self.end.y) / 2
        return Point(x=x, y=y)

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

    def draw(self, img: np.array):
        img = img.copy()
        start = self.start.as_array[
            ::-1
        ]  # reverse order of x and y to get columns and rows
        end = self.end.as_array[::-1]

        rr, cc = draw_line(*start, *end)
        img[rr, cc] = 1
        return img


@dataclasses.dataclass(frozen=True)
class Polygon:
    coords: np.ndarray


def mean_line(lines: List[Line], weighted=True) -> Line:
    lengths = [l.length for l in lines]
    mean_slope = np.average([l.slope for l in lines], weights=lengths)
    max_distance = 0
    best_line = None
    for l1, l2 in combinations(lines, 2):
        d = l1.start.distance(l2.end)
        if d > max_distance:
            max_distance = d
            # best_line = Line(l1.start, l2.end)
    line_length = max_distance
    print(line_length, np.mean(lengths))
    center = Point(*np.median(np.array([l.center.as_array for l in lines]), axis=0))
    end = center.translate(line_length / 2, mean_slope * line_length / 2)
    start = center.translate(-line_length / 2, -mean_slope * line_length / 2)

    return Line(start, end)


def minmax_line(lines: List[Line]) -> Line:
    start_points = np.array([line.start.as_coordinates_tuple for line in lines])
    end_points = np.array([line.end.as_coordinates_tuple for line in lines])
    start = np.min(start_points, axis=0)
    end = np.max(end_points, axis=0)
    return Line(Point(*start), Point(*end))


def predictions_to_polygon(predicted_img, debug=False, approximation_tolerance=0.05):
    predicted_img = predicted_img.squeeze()
    thresholded_image = predicted_img > 0.1
    label_image = label(thresholded_image)
    regions = regionprops(label_image)
    region = sorted(regions, key=lambda r: r.area, reverse=True)[0]
    contour = find_contours(label_image == region.label, fully_connected="high")[0]
    print(len(contour))
    contour = approximate_polygon(contour, tolerance=approximation_tolerance)
    print(len(contour))
    polygon_coords = contour[:, ::-1]
    if debug:
        fig, ax = plt.subplots(figsize=(10, 6))
        poly_patch = mpatches.Polygon(
            polygon_coords, fill=False, edgecolor="red", linewidth=2, closed=True
        )
        ax.plot(polygon_coords[:, 0], polygon_coords[:, 1])
        ax.imshow(thresholded_image, cmap=plt.cm.gray_r)
        ax.add_patch(poly_patch)
        plt.show()
    return polygon_coords
