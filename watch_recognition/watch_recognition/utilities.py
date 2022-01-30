import dataclasses
from functools import cached_property
from itertools import combinations
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from numpy.polynomial import Polynomial
from scipy import odr
from skimage.measure import approximate_polygon, find_contours, label, regionprops


@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float
    name: str = ""  # TODO name and score could be moved to a separate class
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
            **kwargs,
        )

    def draw_marker(self, image: np.ndarray, color=None, thickness=3) -> np.ndarray:
        original_image_np = image.astype(np.uint8)

        x, y = self.as_coordinates_tuple
        x = int(x)
        y = int(y)

        cv2.drawMarker(
            original_image_np, (x, y), color, cv2.MARKER_CROSS, thickness=thickness
        )

        return original_image_np.astype(np.uint8)

    def draw(self, image: np.ndarray, color=None) -> np.ndarray:
        if color is not None:
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"To draw colored point on image, it has to have 3 channels. Got image with shape {image.shape}"
                )
            value = color
        else:
            value = 1
        image = image.copy()
        image[self.y, self.x] = value
        return image


@dataclasses.dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: str
    score: Optional[float] = None

    @classmethod
    def from_center_width_height(
        cls, center_x, center_y, width, height, name="", score=None
    ):
        return BBox(
            x_min=center_x - width / 2,
            y_min=center_y - height / 2,
            x_max=center_x + width / 2,
            y_max=center_y + height / 2,
            name=name,
            score=score,
        )

    @classmethod
    def from_ltwh(cls, left, top, width, height, name="", score=None):
        return BBox(
            x_min=left,
            y_min=top,
            x_max=left + width,
            y_max=top + height,
            name=name,
            score=score,
        )

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
    def area(self):
        return self.height * self.width

    @property
    def top(self):
        return self.y_min

    @property
    def bottom(self):
        return self.y_max

    @property
    def left(self):
        return self.x_min

    @property
    def right(self):
        return self.x_max

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

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

    def plot(self, ax=None, color="red", **kwargs):
        if ax is None:
            ax = plt.gca()
        rect = patches.Rectangle(
            (self.left, self.top),
            self.width,
            self.height,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

    def draw(
        self, image: np.ndarray, color: Tuple[int, int, int] = (255, 0, 255)
    ) -> np.ndarray:
        original_image_np = image.astype(np.uint8)
        xmin, ymin, xmax, ymax = tuple(map(int, self.as_coordinates_tuple))

        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(
            original_image_np,
            self.name,
            (xmin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        return original_image_np.astype(np.uint8)


@dataclasses.dataclass(frozen=True)
class Line:
    start: Point
    end: Point
    score: float = 0

    @classmethod
    def from_multiple_points(cls, points: List[Point]) -> "Line":
        if len(points) < 2:
            raise ValueError(f"Need at least 2 points to fit a lint, got {len(points)}")
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        # vertical line
        if np.std(x_coords) < 2:
            x_const = float(np.mean(x_coords))
            return Line(Point(x_const, y_min), Point(x_const, y_max))

        # other cases
        poly1d = cls._fit_line(x_coords, y_coords)
        start = Point(x_min, poly1d(x_min))
        end = Point(x_max, poly1d(x_max))
        window = BBox(x_min, y_min, x_max, y_max, "")
        l1 = Line(start, end)
        l2 = l1.clip(window)
        return l2

    @cached_property
    def poly1d(self) -> np.poly1d:
        x_coords = [self.start.x, self.end.x]
        y_coords = [self.start.y, self.end.y]
        return self._fit_line(x_coords, y_coords)

    @classmethod
    def _fit_line(
        cls, x_coords: Union[List, np.ndarray], y_coords: Union[List, np.ndarray]
    ) -> np.poly1d:
        """
        Fit 1st degree polynomial using ODR = Orthogonal Distance Regression
        Least squares regression won't work for perfectly vertical lines.

        Notes:
          Check out this Stack Overflow question and answer:
            https://stackoverflow.com/a/10982488/8814045
          and scipy docs with an example:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.polynomial.html#scipy.odr.polynomial

        Args:
            x_coords:
            y_coords:

        Returns:

        """
        # poly_coeffs = np.polyfit(x_coords, y_coords, deg=1)
        # return np.poly1d(poly_coeffs)

        poly_model = odr.polynomial(1)
        data = odr.Data(x_coords, y_coords)
        odr_obj = odr.ODR(data, poly_model)
        output = odr_obj.run()
        poly = np.poly1d(output.beta[::-1])

        return poly

    @property
    def slope(self) -> float:
        return self.poly1d.coeffs[0]

    @property
    def vector(self) -> np.ndarray:
        return self.end.as_array - self.start.as_array

    @property
    def unit_vector(self) -> np.ndarray:
        vector = self.vector
        return vector / np.linalg.norm(vector)

    @property
    def center(self) -> Point:
        x = (self.start.x + self.end.x) / 2
        y = (self.start.y + self.end.y) / 2
        return Point(x=x, y=y)

    @property
    def angle(self):
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return np.arctan2(dy, dx)

    @property
    def length(self) -> float:
        return self.start.distance(self.end)

    def scale(self, x: float, y: float) -> "Line":
        return Line(self.start.scale(x, y), self.end.scale(x, y))

    def translate(self, x: float, y: float) -> "Line":
        return Line(self.start.translate(x, y), self.end.translate(x, y))

    def projection_point(self, point: Point) -> Point:
        line_fit = self.poly1d
        m = line_fit.coeffs[0]
        k = line_fit.coeffs[1]
        proj_point_x = (point.x + m * point.y - m * k) / (m ** 2 + 1)
        proj_point_y = m * proj_point_x + k
        return Point(proj_point_x, proj_point_y)

    def plot(self, ax=None, color=None, draw_arrow: bool = True, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(
            [self.start.x, self.end.x],
            [self.start.y, self.end.y],
            color=color,
            **kwargs,
        )

        if draw_arrow:
            dx = np.sign(self.unit_vector[0])
            dy = self.slope * dx
            ax.arrow(
                self.center.x,
                self.center.y,
                dx,
                dy,
                shape="full",
                edgecolor="black",
                facecolor=color,
                width=0.5,
            )

    def draw(self, img: np.ndarray, color=(0, 255, 0), thickness=10) -> np.ndarray:
        original_image_np = img.astype(np.uint8)
        start = self.start.as_coordinates_tuple
        end = self.end.as_coordinates_tuple
        start = tuple(map(int, start))
        end = tuple(map(int, end))
        cv2.line(original_image_np, start, end, color, thickness=thickness)
        return original_image_np.astype(np.uint8)

    def clip(self, bbox: BBox) -> "Line":
        start, end = self.start, self.end
        new_points = []
        for p in (start, end):
            x = min(max(p.x, bbox.x_min), bbox.x_max)
            y = min(max(p.y, bbox.y_min), bbox.y_max)
            new_p = Point(x, y)
            new_points.append(new_p)
        start, end = new_points
        return Line(start, end)


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
