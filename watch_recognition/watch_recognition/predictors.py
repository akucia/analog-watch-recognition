import dataclasses
import hashlib
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.Image import BICUBIC

from watch_recognition.targets_encoding import (
    decode_single_point,
    extract_points_from_map,
    fit_lines_to_hands_mask,
    line_selector,
)
from watch_recognition.utilities import BBox, Line, Point


class KPPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = tuple(self.model.inputs[0].shape[1:3])
        self.output_size = tuple(self.model.outputs[0].shape[1:3])
        self.cache = {}

    def predict(self, image: Image) -> List[Point]:
        """Runs predictions on a crop of a watch face.
        Returns keypoints in pixel coordinates of the image
        """
        image_hash = _hash_image(image)
        if image_hash in self.cache:
            return self.cache[image_hash]
        # TODO switch to ImageOps.pad
        with image.resize(self.input_size, BICUBIC) as resized_image:
            image_np = np.expand_dims(resized_image, 0)
            predicted = self.model.predict(image_np)[0]
            # transpose to get different kp channels into 0th axis
            predicted = predicted.transpose((2, 0, 1))
            # TODO check if these are correct coords
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
            return [center, top]

    def predict_from_image_and_bbox(self, image: Image, bbox: BBox) -> List[Point]:
        """Runs predictions on full image using bbox to crop area of interest before
        running the model.
        Returns keypoints in pixel coordinates of the image
        """
        # Returns a rectangular region from this image. The box is a
        #         4-tuple defining the left, upper, right, and lower pixel
        with image.crop(box=bbox.as_coordinates_tuple) as crop:
            points = self.predict(crop)
            points = [point.translate(bbox.left, bbox.top) for point in points]
            return points


class HandPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = tuple(self.model.inputs[0].shape[1:3])
        self.output_size = tuple(self.model.outputs[0].shape[1:3])
        self.cache = {}

    def predict(
        self, image: Image, center_point: Point
    ) -> Tuple[Tuple[Line, Line], List[Line]]:
        """Runs predictions on a crop of a watch face.
        Returns keypoints in pixel coordinates of the image
        """
        image_hash = _hash_image(image)
        # if image_hash in self.cache:
        #     return self.cache[image_hash]
        # TODO switch to ImageOps.pad
        with image.resize(self.input_size, BICUBIC) as resized_image:
            image_np = np.expand_dims(resized_image, 0)
            predicted = self.model.predict(image_np)[0]
            predicted = predicted > 0.1
            predicted = (predicted * 255).astype("uint8").squeeze()
            scale_x = image.width / self.output_size[0]
            scale_y = image.height / self.output_size[1]
            center_scaled_to_segmask = center_point.scale(1 / scale_x, 1 / scale_y)
            all_hands_lines = fit_lines_to_hands_mask(
                predicted, center=center_scaled_to_segmask, debug=False
            )
            all_hands_lines = [line.scale(scale_x, scale_y) for line in all_hands_lines]
            return line_selector(all_hands_lines)

    def predict_from_image_and_bbox(
        self, image: Image, bbox: BBox, center_point: Point
    ) -> Tuple[Tuple[Line, Line], List[Line]]:
        """Runs predictions on full image using bbox to crop area of interest before
        running the model.
        Returns keypoints in pixel coordinates of the image
        """
        with image.crop(box=bbox.as_coordinates_tuple) as crop:
            center_point_inside_bbox = center_point.translate(-bbox.left, -bbox.top)
            valid_lines, other_lines = self.predict(crop, center_point_inside_bbox)

            valid_lines = [line.translate(bbox.left, bbox.top) for line in valid_lines]
            other_lines = [line.translate(bbox.left, bbox.top) for line in other_lines]
            return valid_lines, other_lines


class TFLiteDetector:
    def __init__(self, model_path):
        self.temp_file = "/tmp/test-image.png"
        self.model = tf.lite.Interpreter(model_path=model_path)
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

    def predict(self, image: Image) -> List[BBox]:
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


def _hash_image(image: Image) -> str:
    md5hash = hashlib.md5(image.tobytes())
    return md5hash.hexdigest()


class RotationPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = tuple(self.model.inputs[0].shape[1:3])
        self.output_size = self.model.outputs[0].shape[1]
        self.bin_size = 360 // self.output_size
        self.cache = {}

    def predict(self, image: Image) -> float:
        image_hash = _hash_image(image)
        if image_hash in self.cache:
            return self.cache[image_hash]
        # TODO switch to ImageOps.pad
        with image.resize(self.input_size, BICUBIC) as resized_image:
            image_np = np.expand_dims(resized_image, 0)
            predicted = self.model.predict(image_np)[0]
            angle = predicted.argmax(axis=1) * self.bin_size
            self.cache[image_hash] = angle
            return angle

    def predict_and_correct(self, image: Image) -> Image:
        angle = self.predict(image)
        return image.rotate(-angle, resample=BICUBIC)
