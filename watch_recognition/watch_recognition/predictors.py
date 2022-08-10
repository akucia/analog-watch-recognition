import abc
import dataclasses
import hashlib
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.Image import BICUBIC
from PIL.Image import Image as ImageType

from watch_recognition.models import points_to_time
from watch_recognition.targets_encoding import (
    decode_single_point,
    extract_points_from_map,
    fit_lines_to_hands_mask,
    line_selector,
)
from watch_recognition.utilities import (
    BBox,
    Line,
    Point,
    Polygon,
    retinanet_prepare_image,
)


class KPPredictor(ABC):
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = tuple(self.model.inputs[0].shape[1:3])
        self.output_size = tuple(self.model.outputs[0].shape[1:3])
        self.cache = {}

    @abc.abstractmethod
    def _decode_keypoints(
        self, image: ImageType, predicted: np.ndarray
    ) -> Tuple[Point, Point]:
        pass

    def predict(
        self, image: ImageType, rotation_predictor: Optional["RotationPredictor"] = None
    ) -> List[Point]:
        """Runs predictions on a crop of a watch face.
        Returns keypoints in pixel coordinates of the image
        """
        image_hash = _hash_image(image)
        if image_hash in self.cache:
            return self.cache[image_hash]
        # TODO switch to ImageOps.pad
        correction_angle = 0
        if rotation_predictor is not None:
            image, correction_angle = rotation_predictor.predict_and_correct(image)

        with image.resize(self.input_size, BICUBIC) as resized_image:
            image_np = np.expand_dims(resized_image, 0)
            predicted = self.model.predict(image_np)[0]
            center, top = self._decode_keypoints(image, predicted)
        if correction_angle:
            center = center.rotate_around_origin_point(
                Point(image.width / 2, image.height / 2), angle=correction_angle
            )
            top = top.rotate_around_origin_point(
                Point(image.width / 2, image.height / 2), angle=correction_angle
            )
        return [center, top]

    def predict_from_image_and_bbox(
        self,
        image: ImageType,
        bbox: BBox,
        rotation_predictor: Optional["RotationPredictor"] = None,
    ) -> List[Point]:
        """Runs predictions on full image using bbox to crop area of interest before
        running the model.
        Returns keypoints in pixel coordinates of the image
        """
        with image.crop(box=bbox.as_coordinates_tuple) as crop:
            points = self.predict(crop, rotation_predictor=rotation_predictor)
            points = [point.translate(bbox.left, bbox.top) for point in points]
            return points


class KPHeatmapPredictor(KPPredictor):
    def _decode_keypoints(self, image, predicted):
        # transpose to get different kp channels into 0th axis
        predicted = predicted.transpose((2, 0, 1))
        # Center
        center = decode_single_point(predicted[0])
        scale_x = image.width / self.output_size[0]
        scale_y = image.height / self.output_size[1]
        center = dataclasses.replace(center, name="Center")
        center = center.scale(scale_x, scale_y)
        # Top
        top_points = extract_points_from_map(
            predicted[1],
        )
        if not top_points:
            top_points = [decode_single_point(predicted[1])]
        top = sorted(top_points, key=lambda x: x.score)[-1]
        top = dataclasses.replace(top, name="Top")
        top = top.scale(scale_x, scale_y)
        return center, top


class KPRegressionPredictor(KPPredictor):
    def _decode_keypoints(self, image, predicted) -> Tuple[Point, Point]:
        center = Point(predicted[0], predicted[1], name="Center")
        top = Point(predicted[2], 0.25, name="Top")
        scale_x = image.width
        scale_y = image.height
        center = center.scale(scale_x, scale_y)
        top = top.scale(scale_x, scale_y)
        return center, top


class HandPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = tuple(self.model.inputs[0].shape[1:3])
        self.output_size = tuple(self.model.outputs[0].shape[1:3])
        self.cache = {}

    def predict(
        self,
        image: ImageType,
        center_point: Point,
        threshold: float = 0.999,
        debug: bool = False,
    ) -> Tuple[Tuple[Line, Line], List[Line], Polygon]:
        """Runs predictions on a crop of a watch face.
        Returns keypoints in pixel coordinates of the image
        """
        # TODO switch to ImageOps.pad
        # ImageOps.pad(image, size=self.input_size, method=BICUBIC)
        with image.resize(self.input_size, BICUBIC) as resized_image:
            image_np = np.expand_dims(resized_image, 0)
            predicted = self.model.predict(image_np)[0]

            predicted = predicted > threshold
            predicted = (predicted * 255).astype("uint8").squeeze()
            polygon = Polygon.from_binary_mask(predicted)

            scale_x = image.width / self.output_size[0]
            scale_y = image.height / self.output_size[1]

            center_scaled_to_segmask = center_point.scale(1 / scale_x, 1 / scale_y)

            all_hands_lines = fit_lines_to_hands_mask(
                predicted, center=center_scaled_to_segmask, debug=debug
            )
            all_hands_lines = [line.scale(scale_x, scale_y) for line in all_hands_lines]
            polygon = polygon.scale(scale_x, scale_y)
            return *line_selector(all_hands_lines, center=center_point), polygon

    def predict_from_image_and_bbox(
        self,
        image: ImageType,
        bbox: BBox,
        center_point: Point,
        threshold: float = 0.5,
        debug: bool = False,
    ) -> Tuple[Tuple[Line, Line], List[Line], Polygon]:
        """Runs predictions on full image using bbox to crop area of interest before
        running the model.
        Returns keypoints in pixel coordinates of the image
        """
        with image.crop(box=bbox.as_coordinates_tuple) as crop:
            center_point_inside_bbox = center_point.translate(-bbox.left, -bbox.top)
            valid_lines, other_lines, polygon = self.predict(
                crop, center_point_inside_bbox, threshold=threshold, debug=debug
            )

            valid_lines = [line.translate(bbox.left, bbox.top) for line in valid_lines]
            other_lines = [line.translate(bbox.left, bbox.top) for line in other_lines]
            polygon = polygon.translate(bbox.left, bbox.top)
            return valid_lines, other_lines, polygon


class TFLiteDetector:
    def __init__(self, model_path: Path):
        self.temp_file = "/tmp/test-image.png"
        if model_path.is_dir():
            model_path /= "model.tflite"
        self.model = tf.lite.Interpreter(model_path=str(model_path))
        _, input_height, input_width, _ = self.model.get_input_details()[0]["shape"]
        self.input_size = (input_width, input_height)
        self.model.allocate_tensors()
        self.cache = {}

    @classmethod
    def detect_objects(cls, interpreter, image, threshold):
        """Returns a list of detection results, each a dictionary of object info."""
        # Feed the input image to the model
        cls.set_input_tensor(interpreter, image)
        interpreter.invoke()

        # Get all outputs from the model
        boxes = cls.get_output_tensor(interpreter, 0)
        classes = cls.get_output_tensor(interpreter, 1)
        scores = cls.get_output_tensor(interpreter, 2)
        count = int(cls.get_output_tensor(interpreter, 3))

        results = []
        for i in range(count):
            if scores[i] >= threshold:
                result = {
                    "bounding_box": boxes[i],
                    "class_id": classes[i],
                    "score": scores[i],
                }
                results.append(result)
        return results

    # functions to run object detector in tflite from object detector model maker
    @staticmethod
    def set_input_tensor(interpreter, image):
        """Set the input tensor."""
        tensor_index = interpreter.get_input_details()[0]["index"]
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    @staticmethod
    def get_output_tensor(interpreter, index):
        """Retur the output tensor at the given index."""
        output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details["index"]))
        return tensor

    @staticmethod
    def preprocess_image(image_path, input_size):
        """Preprocess the input image to feed to the TFLite model"""
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)
        original_image = img
        resized_img = tf.image.resize(img, input_size)
        resized_img = resized_img[tf.newaxis, :]
        return resized_img, original_image

    def predict(self, image: ImageType) -> List[BBox]:
        """Run object detection on the input image and draw the detection results"""
        image_hash = _hash_image(image)
        if image_hash in self.cache:
            return self.cache[image_hash]
        im = image.copy()
        im.thumbnail((512, 512), Image.BICUBIC)
        # TODO skip temp file?
        im.save(self.temp_file, "PNG")

        # Load the input image and preprocess it
        preprocessed_image, original_image = self.preprocess_image(
            self.temp_file, (self.input_size[1], self.input_size[0])
        )

        # Run object detection on the input image
        results = self.detect_objects(self.model, preprocessed_image, threshold=0.5)

        bboxes = []
        for obj in results:
            ymin, xmin, ymax, xmax = obj["bounding_box"]
            # Find the class index of the current object
            # class_id = int(obj["class_id"])
            score = float(obj["score"])
            bboxes.append(
                BBox(
                    x_min=xmin,
                    y_min=ymin,
                    x_max=xmax,
                    y_max=ymax,
                    name="bbox",
                    score=score,
                ).scale(image.width, image.height)
            )
        self.cache[image_hash] = bboxes

        return bboxes


class RetinanetDetector:
    def __init__(
        self,
        model_path: Path,
        class_to_label_name: Dict[int, str],
        input_size: Tuple[int, int] = (334, 334),
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = input_size
        self.class_to_label_name = class_to_label_name
        self.cache = {}

    def predict(self, image: ImageType) -> List[BBox]:
        """Run object detection on the input image and draw the detection results"""
        image_hash = _hash_image(image)
        if image_hash in self.cache:
            return self.cache[image_hash]
        # TODO integrate the image preprocessing with the exported model
        input_image, ratio = retinanet_prepare_image(np.array(image))
        ratio = ratio.numpy()
        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = self.model.predict(
            input_image
        )
        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
            nmsed_boxes[0],
            nmsed_scores[0],
            nmsed_classes[0],
            valid_detections[0],
        )
        nmsed_boxes = nmsed_boxes[:valid_detections]
        nmsed_classes = nmsed_classes[:valid_detections]
        nmsed_scores = nmsed_scores[:valid_detections]
        nmsed_boxes = nmsed_boxes / ratio
        bboxes = []
        for box, cls, score in zip(nmsed_boxes, nmsed_classes, nmsed_scores):
            bbox = BBox(*box, name=self.class_to_label_name[cls], score=score)
            bboxes.append(bbox)

        self.cache[image_hash] = bboxes
        return bboxes


def _hash_image(image: ImageType) -> str:
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()


class RotationPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = tuple(self.model.inputs[0].shape[1:3])
        self.output_size = self.model.outputs[0].shape[1]
        self.bin_size = 360 // self.output_size
        self.cache = {}

    def predict(
        self, image: ImageType, debug: bool = False, threshold: float = 0.5
    ) -> float:
        image_hash = _hash_image(image)
        # if image_hash in self.cache:
        #     return self.cache[image_hash]
        # TODO switch to ImageOps.pad
        with image.resize(self.input_size, BICUBIC) as resized_image:
            image_np = np.expand_dims(resized_image, 0)
            predicted = self.model.predict(image_np)[0]
            if debug:
                print((predicted * 100).astype(int))
            argmax = predicted.argmax()
            if predicted[argmax] > threshold:
                angle = argmax * self.bin_size
            else:
                angle = 0
            self.cache[image_hash] = angle
            return angle

    def predict_and_correct(
        self, image: ImageType, debug: bool = False
    ) -> Tuple[ImageType, float]:
        angle = self.predict(image, debug=debug)
        return image.rotate(-angle, resample=BICUBIC), -angle


class ClockTimePredictor:
    def __init__(self):
        self.detector: TFLiteDetector = ""
        self.rotation_predictor: RotationPredictor = ""
        self.kp_predictor: KPPredictor = ""
        self.hands_predictor: HandPredictor = ""

    def predict(self, image) -> List[BBox]:
        bboxes = self.detector.predict(image)
        results = []
        for box in bboxes:
            pred_center, pred_top = self.kp_predictor.predict_from_image_and_bbox(
                image, box, rotation_predictor=self.rotation_predictor
            )
            # TODO remove debug drawing and move it to a different method
            # frame = pred_center.draw_marker(frame, thickness=2)
            # frame = pred_top.draw_marker(frame, thickness=2)
            minute_and_hour, other = self.hands_predictor.predict_from_image_and_bbox(
                image, box, pred_center
            )
            if minute_and_hour:
                pred_minute, pred_hour = minute_and_hour
                read_hour, read_minute = points_to_time(
                    pred_center, pred_hour.end, pred_minute.end, pred_top
                )
                # frame = pred_minute.draw(frame, thickness=3)
                # frame = pred_minute.end.draw_marker(frame, thickness=2)
                # frame = pred_hour.draw(frame, thickness=5)
                # frame = pred_hour.end.draw_marker(frame, thickness=2)

                time = f"{read_hour:.0f}:{read_minute:.0f}"
                box = dataclasses.replace(box, name=time)
                results.append(box)
        return results
