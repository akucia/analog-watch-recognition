import abc
import dataclasses
import hashlib
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import grpc
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.Image import BICUBIC
from PIL.Image import Image as ImageType
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.apis.predict_pb2 import PredictResponse

from watch_recognition.models import points_to_time
from watch_recognition.targets_encoding import (
    decode_single_point_from_heatmap,
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


class KPHeatmapPredictorV2(ABC):
    def __init__(self, class_to_label_name, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.class_to_label_name = class_to_label_name

    @abc.abstractmethod
    def _batch_predict(self, batch_image_np: np.ndarray) -> np.ndarray:
        pass

    def predict(self, image: Union[ImageType, np.ndarray]) -> List[Point]:
        """Runs predictions on a crop of a watch face.
        Returns keypoints in pixel coordinates of the image
        """

        image_np = np.array(image)
        batch_image_np = np.expand_dims(image_np, 0).astype("float32")
        predicted = self._batch_predict(batch_image_np)[0]
        points = self._decode_keypoints(
            predicted,
            crop_image_width=image_np.shape[1],
            crop_image_height=image_np.shape[0],
        )

        return points

    def predict_from_image_and_bbox(
        self,
        image: ImageType,
        bbox: BBox,
    ) -> List[Point]:
        """Runs predictions on full image using bbox to crop area of interest before
        running the model.
        Returns keypoints in pixel coordinates of the image
        """
        with image.crop(box=bbox.as_coordinates_tuple) as crop:
            points = self.predict(crop)
            points = [point.translate(bbox.left, bbox.top) for point in points]
            return points

    def _decode_keypoints(
        self, predicted: np.ndarray, crop_image_width: int, crop_image_height: int
    ) -> List[Point]:
        """Decodes predictions for every channel for output heatmap
        Returns keypoints in pixel coordinates of the BBox crop
        """

        points = []
        for cls, name in self.class_to_label_name.items():
            maybe_point = decode_single_point_from_heatmap(
                predicted[:, :, cls], threshold=self.confidence_threshold
            )
            if maybe_point is not None:
                maybe_point = maybe_point.rename(name)
                maybe_point = maybe_point.scale(
                    crop_image_width / predicted.shape[1],
                    crop_image_height / predicted.shape[0],
                )
                points.append(maybe_point)
        return points


class KPHeatmapPredictorV2Local(KPHeatmapPredictorV2):
    def __init__(
        self, model_path, class_to_label_name, confidence_threshold: float = 0.5
    ):
        self.model: tf.keras.models.Model = tf.keras.models.load_model(
            model_path, compile=False
        )
        super().__init__(class_to_label_name, confidence_threshold)

    def _batch_predict(self, batch_image_np: np.ndarray) -> np.ndarray:
        predicted = self.model.predict(batch_image_np, verbose=0)
        return predicted


class KPHeatmapPredictorV2GRPC(KPHeatmapPredictorV2):
    def __init__(
        self,
        host: str,
        model_name: str,
        class_to_label_name,
        confidence_threshold: float = 0.5,
        timeout: float = 10.0,
    ):
        self.host = host
        self.model_name = model_name
        self.channel = grpc.insecure_channel(self.host)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.timeout = timeout
        super().__init__(class_to_label_name, confidence_threshold)

    def __enter__(self):
        return self

    def __exit__(self):
        self.channel.close()

    def _batch_predict(self, batch_image_np: np.ndarray) -> np.ndarray:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"
        request.inputs["image"].CopyFrom(tf.make_tensor_proto(batch_image_np))
        result = self.stub.Predict(request, self.timeout)
        return self._decode_proto_results(result)

    def _decode_proto_results(self, result: PredictResponse) -> np.ndarray:
        # TODO rename the model outputs name later
        output = result.outputs["model_3"]
        value = output.float_val
        shape = [dim.size for dim in output.tensor_shape.dim]
        return np.array(value).reshape(shape)


class KPRegressionPredictor(KPPredictor):
    def _decode_keypoints(self, image, predicted) -> Tuple[Point, Point]:
        center = Point(predicted[0], predicted[1], name="Center")
        top = Point(predicted[2], 0.25, name="Top")
        scale_x = image.width
        scale_y = image.height
        center = center.scale(scale_x, scale_y)
        top = top.scale(scale_x, scale_y)
        return center, top


class HandPredictor(ABC):
    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold

    @abc.abstractmethod
    def _batch_predict(self, batch_image_numpy: np.ndarray) -> np.ndarray:
        pass

    def predict(
        self,
        image: ImageType,
        center_point: Point,
        debug: bool = False,
    ) -> Tuple[Tuple[Line, Line], List[Line], Polygon]:
        """Runs predictions on a crop of a watch face.
        Returns keypoints in pixel coordinates of the image
        """
        image_np = np.expand_dims(np.array(image), 0).astype("float32")
        predicted = self._batch_predict(image_np)[0].squeeze()

        predicted = predicted > self.confidence_threshold
        polygon = Polygon.from_binary_mask(predicted)

        scale_x = image.width / predicted.shape[0]
        scale_y = image.height / predicted.shape[1]

        center_scaled_to_segmask = center_point.scale(1 / scale_x, 1 / scale_y)

        all_hands_lines = fit_lines_to_hands_mask(
            predicted, center=center_scaled_to_segmask, debug=debug
        )
        all_hands_lines = [line.scale(scale_x, scale_y) for line in all_hands_lines]
        polygon = polygon.scale(scale_x, scale_y)
        return (*line_selector(all_hands_lines, center=center_point), polygon)

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
                crop, center_point_inside_bbox, debug=debug
            )

            valid_lines = [line.translate(bbox.left, bbox.top) for line in valid_lines]
            other_lines = [line.translate(bbox.left, bbox.top) for line in other_lines]
            polygon = polygon.translate(bbox.left, bbox.top)
            return valid_lines, other_lines, polygon


class HandPredictorLocal(HandPredictor):
    def __init__(self, model_path, confidence_threshold: float = 0.5):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = tuple(self.model.inputs[0].shape[1:3])
        self.output_size = tuple(self.model.outputs[0].shape[1:3])
        super().__init__(confidence_threshold=confidence_threshold)

    def _batch_predict(self, batch_image_numpy: np.ndarray) -> np.ndarray:
        predicted = self.model.predict(batch_image_numpy)
        return predicted


class HandPredictorGRPC(HandPredictor):
    def __init__(
        self,
        host: str,
        model_name: str,
        timeout: float = 10.0,
        confidence_threshold: float = 0.5,
    ):
        self.host = host
        self.model_name = model_name
        self.channel = grpc.insecure_channel(self.host)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.timeout = timeout
        super().__init__(confidence_threshold=confidence_threshold)

    # TODO: this could be inherited from another class using Mixin pattern
    def _batch_predict(self, batch_image_np: np.ndarray) -> np.ndarray:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"
        request.inputs["image"].CopyFrom(tf.make_tensor_proto(batch_image_np))
        result = self.stub.Predict(request, self.timeout)
        return self._decode_proto_results(result)

    def _decode_proto_results(self, result: PredictResponse) -> np.ndarray:
        # TODO rename the model outputs name later to make it universal
        output = result.outputs["model_3"]
        value = output.float_val
        shape = [dim.size for dim in output.tensor_shape.dim]
        return np.array(value).reshape(shape)


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


class RetinanetDetector(ABC):
    def __init__(
        self,
        class_to_label_name: Dict[int, str],
    ):
        self.class_to_label_name = class_to_label_name

    @abc.abstractmethod
    def _batch_predict(
        self, batch_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def predict(self, image: ImageType) -> List[BBox]:
        """Run object detection on the input image and draw the detection results"""
        # TODO integrate the image preprocessing with the exported model
        input_image, ratio = retinanet_prepare_image(np.array(image))
        ratio = ratio.numpy()
        (
            nmsed_boxes,
            nmsed_classes,
            nmsed_scores,
            valid_detections,
        ) = self._batch_predict(input_image.numpy())
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
            # cast everything to float to prevent issues with JSON serialization
            # and stick to types specified by BBox class
            float_bbox = list(map(float, box))
            bbox = BBox(
                *float_bbox, name=self.class_to_label_name[cls], score=float(score)
            )
            bboxes.append(bbox)
        return bboxes


class RetinanetDetectorLocal(RetinanetDetector):
    def __init__(
        self,
        model_path: Path,
        class_to_label_name: Dict[int, str],
    ):
        self.model = tf.keras.models.load_model(model_path)
        super().__init__(class_to_label_name=class_to_label_name)

    def _batch_predict(self, input_image):
        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = self.model.predict(
            input_image,
            verbose=0,
        )
        return nmsed_boxes, nmsed_classes, nmsed_scores, valid_detections


class RetinaNetDetectorGRPC(RetinanetDetector):
    def __init__(
        self,
        host: str,
        model_name: str,
        class_to_label_name: Dict[int, str],
        timeout: float = 10.0,
    ):
        self.host = host
        self.model_name = model_name
        self.channel = grpc.insecure_channel(self.host)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.timeout = timeout
        super().__init__(class_to_label_name)

    def _batch_predict(
        self, input_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"
        request.inputs["image"].CopyFrom(tf.make_tensor_proto(input_images))
        result = self.stub.Predict(request, self.timeout)
        parsed_results = {}
        for key in {"RetinaNet", "RetinaNet_1", "RetinaNet_2", "RetinaNet_3"}:
            output = result.outputs[key]
            if output.float_val:
                value = output.float_val
            elif output.int_val:
                value = output.int_val
            else:
                raise ValueError(f"unrecognized dtype for {output}")
            shape = [dim.size for dim in output.tensor_shape.dim]
            parsed_results[key] = np.array(value).reshape(shape)

        nmsed_boxes = parsed_results["RetinaNet"]
        nmsed_classes = parsed_results["RetinaNet_2"]
        nmsed_scores = parsed_results["RetinaNet_1"]
        valid_detections = parsed_results["RetinaNet_3"]
        return nmsed_boxes, nmsed_classes, nmsed_scores, valid_detections


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
