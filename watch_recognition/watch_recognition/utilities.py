import dataclasses
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy import odr
from skimage import measure
from skimage.measure import (
    LineModelND,
    approximate_polygon,
    find_contours,
    label,
    ransac,
    regionprops,
)


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
        return float(np.sqrt((diff**2).sum()))

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

    @classmethod
    def from_label_studio_object(cls, data: dict) -> "Point":
        return Point(
            data["x"],
            data["y"],
            data["keypointlabels"][0],
        ).scale(1 / 100, 1 / 100)

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

    def rename(self, new_name: str) -> "Point":
        return dataclasses.replace(self, name=new_name)

    @property
    def center(self) -> "Point":
        return self


@dataclasses.dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    name: str = ""
    score: Optional[float] = None

    def __post_init__(self):
        x_min, x_max = min(self.x_min, self.x_max), max(self.x_min, self.x_max)
        y_min, y_max = min(self.y_min, self.y_max), max(self.y_min, self.y_max)
        object.__setattr__(self, "x_min", x_min)
        object.__setattr__(self, "x_max", x_max)
        object.__setattr__(self, "y_min", y_min)
        object.__setattr__(self, "y_max", y_max)

    @classmethod
    def unit(cls, name="", score=None):
        return BBox(
            x_min=0,
            y_min=0,
            x_max=1,
            y_max=1,
            name=name,
            score=score,
        )

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
            self.x_min * x,
            self.y_min * y,
            self.x_max * x,
            self.y_max * y,
            self.name,
            self.score,
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

    def translate(self, x: float = 0.0, y: float = 0.0) -> "BBox":
        return BBox(
            x_min=self.x_min + x,
            x_max=self.x_max + x,
            y_min=self.y_min + y,
            y_max=self.y_max + y,
            score=self.score,
            name=self.name,
        )

    def reflect(self, x: Optional[float] = None, y: Optional[float] = None) -> "BBox":
        # TODO this should be more generic
        if x is not None:
            reflected_box = BBox(
                x_min=2 * x - self.x_min,
                x_max=2 * x - self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                score=self.score,
                name=self.name,
            )
        if y is not None:
            reflected_box = BBox(
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=2 * y - self.y_min,
                y_max=2 * y - self.y_max,
                score=self.score,
                name=self.name,
            )
        return reflected_box

    @property
    def as_coordinates_tuple(self) -> Tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def convert_to_int_coordinates_tuple(
        self, method: str = "round"
    ) -> Tuple[int, int, int, int]:
        if method == "round":
            return tuple(np.round(self.as_coordinates_tuple).astype(int))
        elif method == "floor":
            return tuple(np.floor(self.as_coordinates_tuple).astype(int))
        else:
            raise ValueError(f"unrecognized method {method}, choose one of round|floor")

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
    def corners(self) -> List[Point]:
        return [
            Point(self.x_min, self.y_min),
            Point(self.x_max, self.y_min),
            Point(self.x_max, self.y_max),
            Point(self.y_min, self.y_max),
        ]

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

    @classmethod
    def from_label_studio_object(cls, data) -> "BBox":
        return BBox.from_ltwh(
            data["x"],
            data["y"],
            data["width"],
            data["height"],
            data["rectanglelabels"][0],
        ).scale(1 / 100, 1 / 100)

    def to_coco_object(
        self,
        image_id: str,
        object_id: Union[str, int],
        category_id: Optional[str] = None,
    ) -> dict:
        return {
            "segmentation": [point.as_coordinates_tuple for point in self.corners],
            "area": self.area,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [self.x_min, self.y_min, self.width, self.height],
            "category_id": category_id if category_id is not None else self.name,
            "id": object_id,
            "score": self.score,
        }

    def intersection(self, other: "BBox") -> "BBox":

        x_min = max(self.x_min, other.x_min)
        x_max = min(self.x_max, other.x_max)
        # x_min, x_max = min(x_min, x_max), max(x_min, x_max)

        y_min = max(self.y_min, other.y_min)
        y_max = min(self.y_max, other.y_max)

        width = x_max - x_min
        height = y_max - y_min
        if width < 0 or height < 0:
            x_min = 0
            y_min = 0
            x_max = 0
            y_max = 0
        return BBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            name=self.name,
            score=self.score,
        )

    def iou(self, other: "BBox") -> float:
        intersection_area = self.intersection(other).area
        if not intersection_area:
            return 0.0
        union_area = self.area + other.area - intersection_area
        return intersection_area / union_area

    def plot(
        self,
        ax=None,
        color: str = "red",
        linewidth: int = 1,
        draw_name_label: bool = True,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()
        rect = patches.Rectangle(
            (self.left, self.top),
            self.width,
            self.height,
            edgecolor=color,
            facecolor="none",
            linewidth=linewidth,
            **kwargs,
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        if self.name and draw_name_label:
            ax.text(
                self.x_min,
                self.y_min,
                self.name,
                bbox={"facecolor": color, "alpha": 0.4},
                clip_box=ax.clipbox,
                clip_on=True,
            )

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

    def rename(self, new_name: str) -> "BBox":
        return dataclasses.replace(self, name=new_name)


@dataclasses.dataclass(frozen=True)
class Line:
    start: Point
    end: Point
    score: float = 0

    @classmethod
    def from_multiple_points(
        cls, points: List[Point], use_ransac: bool = False
    ) -> "Line":
        if len(points) < 2:
            raise ValueError(f"Need at least 2 points to fit a lint, got {len(points)}")
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        # vertical line
        # TODO < 2 is still pretty broad, maybe it should be less than 1e-3?
        if np.std(x_coords) < 2:
            x_const = float(np.mean(x_coords))
            return Line(Point(x_const, y_min), Point(x_const, y_max))
        # horizontal line
        if np.std(y_coords) < 2:
            y_const = float(np.mean(y_coords))
            return Line(Point(x_min, y_const), Point(x_max, y_const))

        if use_ransac and len(points) > 2:
            data = np.column_stack([x_coords, y_coords])
            model_robust, inliers = ransac(
                data, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000
            )
            line_x = [x_min, x_max]
            line_y_min, line_y_max = model_robust.predict_y(line_x)
            start = Point(x_min, line_y_min)
            end = Point(x_max, line_y_max)
        else:

            # other cases
            poly1d = cls._fit_line(x_coords, y_coords)
            start = Point(x_min, poly1d(x_min))
            end = Point(x_max, poly1d(x_max))
        window = BBox(x_min, y_min, x_max, y_max, "")
        return Line(start, end).clip(window)

    @property
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

    def angle_between(self, other):
        vec_1 = self.vector
        vec_2 = other.vector
        unit_vector_1 = vec_1 / np.linalg.norm(vec_1)
        unit_vector_2 = vec_2 / np.linalg.norm(vec_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        return angle

    @property
    def length(self) -> float:
        return self.start.distance(self.end)

    def scale(self, x: float, y: float) -> "Line":
        return Line(self.start.scale(x, y), self.end.scale(x, y), score=self.score)

    def translate(self, x: float, y: float) -> "Line":
        return Line(
            self.start.translate(x, y), self.end.translate(x, y), score=self.score
        )

    def projection_point(self, point: Point) -> Point:
        line_fit = self.poly1d
        m = line_fit.coeffs[0]
        k = line_fit.coeffs[1]
        proj_point_x = (point.x + m * point.y - m * k) / (m**2 + 1)
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
        return Line(start, end, score=self.score)


@dataclasses.dataclass(frozen=True)
class Polygon:
    coords: np.ndarray
    # name should be changed to label, and support either str or int
    name: str = ""

    @classmethod
    def from_binary_mask(
        cls, mask: np.ndarray, simplification_tolerance: float = 1.0
    ) -> "Polygon":
        if mask.dtype != np.bool:
            raise ValueError(
                f"argument mask must be of type {np.bool}, got {mask.dtype}"
            )
        labeled_mask = measure.label(mask)
        regions = sorted(regionprops(labeled_mask), key=lambda r: r.area, reverse=True)
        if not regions:
            return Polygon(np.array([]).reshape(-1, 2))
        region = regions[0]
        contours = measure.find_contours(
            labeled_mask == region.label, fully_connected="high"
        )
        if not contours:
            return Polygon(np.array([]).reshape(-1, 2))

        contour = sorted(contours, key=lambda c: len(c))[0]
        appr_poly = approximate_polygon(contour, tolerance=simplification_tolerance)
        return Polygon(appr_poly[:, ::-1])

    def to_binary_mask(self, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width))
        cv2.fillPoly(mask, [self.coords.astype(int)], 1)
        return mask.astype(bool)

    def scale(self, x: float, y: float) -> "Polygon":
        coords = self.coords.copy()
        coords[:, 0] *= x
        coords[:, 1] *= y
        return Polygon(coords=coords, name=self.name)

    def translate(self, x: float, y: float) -> "Polygon":
        coords = self.coords.copy()
        coords[:, 0] += x
        coords[:, 1] += y
        return Polygon(coords=coords, name=self.name)

    @property
    def is_empty(self) -> bool:
        return self.coords.size > 0

    @property
    def as_label_studio_object(self) -> dict:
        return {
            "polygonlabels": [self.name],
            # label studio requires coordinates from 0-100 (aka percentage of the image)
            # upper left corner coordinates
            "points": (self.coords * 100).tolist(),
        }

    @classmethod
    def from_label_studio_object(cls, data) -> "Polygon":
        return Polygon(
            coords=np.array(data["points"]),
            name=data["polygonlabels"][0],
        ).scale(1 / 100, 1 / 100)

    @property
    def center(self) -> "Point":
        mean_coord = np.mean(self.coords, axis=0)
        return Point(x=mean_coord[0], y=mean_coord[1])

    def rename(self, new_name: str) -> "Polygon":
        return dataclasses.replace(self, name=new_name)

    def plot(
        self,
        ax=None,
        color: str = "red",
        linewidth: int = 1,
        draw_name_label: bool = True,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()
        rect = patches.Polygon(
            self.coords,
            edgecolor=color,
            facecolor="none",
            linewidth=linewidth,
            **kwargs,
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        if self.name and draw_name_label:
            ax.text(
                np.min(self.coords[:, 0]),
                np.min(self.coords[:, 1]),
                self.name,
                bbox={"facecolor": color, "alpha": 0.4},
                clip_box=ax.clipbox,
                clip_on=True,
            )


def mean_line(lines: List[Line], weighted=True) -> Line:
    lengths = [linr.length for linr in lines]
    mean_slope = np.average([line.slope for line in lines], weights=lengths)
    max_distance = 0
    for l1, l2 in combinations(lines, 2):
        d = l1.start.distance(l2.end)
        if d > max_distance:
            max_distance = d
    line_length = max_distance
    center = Point(
        *np.median(np.array([line.center.as_array for line in lines]), axis=0)
    )
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
    contour = approximate_polygon(contour, tolerance=approximation_tolerance)
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


def retinanet_prepare_image(image):
    # TODO replace with ImageOps.pad
    image, _, ratio = resize_and_pad_image(image, jitter=None, max_side=384)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio


def resize_and_pad_image(
    image,
    min_side=800.0,
    max_side=1333.0,
    jitter: Optional[Tuple[int, int]] = (640, 1024),
    stride=128.0,
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


def match_objects_to_bboxes(
    bboxes: List[BBox], objects: List[Union[Point, Polygon, BBox]]
) -> Dict[BBox, List[Union[Point, Polygon, BBox]]]:
    rectangles_to_kps = defaultdict(list)
    for bbox in bboxes:
        for obj in objects:
            if bbox.contains(obj.center):
                rectangles_to_kps[bbox].append(obj)
    return dict(rectangles_to_kps)


def iou_bbox_matching(a: List[BBox], b: List[BBox]) -> Dict[BBox, Optional[BBox]]:
    if not b:
        return dict.fromkeys(a)
    matching = {}
    for bbox_a in a:
        scores = sorted(b, key=lambda bbox_b: bbox_b.iou(bbox_a), reverse=True)
        top_score_bbox = scores[0]
        if top_score_bbox in matching.values():
            matching[bbox_a] = None
        elif top_score_bbox.iou(bbox_a) > 0:
            matching[bbox_a] = top_score_bbox
        else:
            matching[bbox_a] = None
    # TODO unmatched assigned to matching[None]?

    return matching
