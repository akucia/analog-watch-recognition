import abc
import dataclasses
import hashlib
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import grpc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from distinctipy import distinctipy
from PIL import Image
from PIL.Image import BICUBIC
from PIL.Image import Image as ImageType
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.apis.predict_pb2 import PredictResponse

from watch_recognition.models import points_to_time
from watch_recognition.targets_encoding import (
    _fit_lines_to_points,
    extract_points_from_map,
    line_selector,
    segment_hands_mask,
)
from watch_recognition.utilities import BBox, Point, Polygon
from watch_recognition.visualization import draw_masks


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
            if crop.width * crop.height < 1:
                return []
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
        image_width, image_height = _get_image_shape(image)

        heatmap = self.predict_heatmap(image)
        points = self._decode_keypoints(
            heatmap,
            crop_image_width=image_width,
            crop_image_height=image_height,
        )

        return points

    def predict_heatmap(self, image: Union[ImageType, np.ndarray]) -> np.ndarray:
        """Runs predictions on a crop of a watch face.
        Returns heatmap of keypoint regions
        """

        image_np = np.array(image)
        batch_image_np = np.expand_dims(image_np, 0)
        predicted = self._batch_predict(batch_image_np)[0]
        return predicted

    def predict_and_plot(self, image: Union[ImageType, np.ndarray], ax=None):
        image_width, image_height = _get_image_shape(image)
        ax = ax or plt.gca()
        colors = distinctipy.get_colors(len(self.class_to_label_name))
        points = self.predict(image)
        heatmap = self.predict_heatmap(image)
        predicted = heatmap > self.confidence_threshold
        masks = []
        for cls, name in self.class_to_label_name.items():
            mask = predicted[:, :, cls]
            mask = cv2.resize(
                mask.astype("uint8"),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype("bool")
            masks.append(mask)

        ax.imshow(draw_masks(np.array(image), masks, colors))
        for i, point in enumerate(points):
            point.plot(color=colors[i], ax=ax)
        ax.legend()
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
            if crop.width * crop.height < 1:
                return []
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
            # TODO if keypoint detector is improved this might be simplified to back to
            #   decode_single_point_from_heatmap
            map_points = extract_points_from_map(
                predicted[:, :, cls], detection_threshold=self.confidence_threshold
            )
            map_points = sorted(map_points, key=lambda p: p.score, reverse=True)

            if map_points:
                point = map_points[0]
                point = point.rename(name).scale(
                    crop_image_width / predicted.shape[1],
                    crop_image_height / predicted.shape[0],
                )
                points.append(point)
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


# TODO Rename to Polygon predictor
class HandPredictor(ABC):
    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold

    @abc.abstractmethod
    def _batch_predict(self, batch_image_numpy: np.ndarray) -> np.ndarray:
        pass

    def predict(
        self,
        image: Union[ImageType, np.ndarray],
    ) -> Polygon:
        """Runs predictions on a crop of a watch face.
        Returns keypoints in pixel coordinates of the image
        """
        image_width, image_height = _get_image_shape(image)
        image_np = np.expand_dims(np.array(image), 0)
        predicted = self._batch_predict(image_np)[0].squeeze()

        predicted = predicted > self.confidence_threshold
        polygon = Polygon.from_binary_mask(predicted)

        scale_x = image_width / predicted.shape[0]
        scale_y = image_height / predicted.shape[1]

        polygon = polygon.scale(scale_x, scale_y)
        return polygon

    def predict_mask_and_draw(self, image: Union[ImageType, np.ndarray], ax=None):
        ax = ax or plt.gca()
        image_np = np.array(image)
        image_batch_np = np.expand_dims(image_np, 0)
        predicted = self._batch_predict(image_batch_np)[0]

        predicted = predicted > self.confidence_threshold
        masks = []
        for cls, name in {0: "ClockHands"}.items():
            mask = predicted[:, :, cls]
            mask = cv2.resize(
                mask.astype("uint8"),
                image_np.shape[:2][::-1],
                interpolation=cv2.INTER_NEAREST,
            ).astype("bool")
            masks.append(mask)

        ax.imshow(draw_masks(image_np, masks))

    def predict_and_plot(self, image: Union[ImageType, np.ndarray], ax=None) -> Polygon:
        ax = ax or plt.gca()
        ax.imshow(np.array(image))
        polygon = self.predict(image)
        polygon.plot(ax=ax)
        return polygon

    def predict_from_image_and_bbox(self, image: ImageType, bbox: BBox) -> Polygon:
        """Runs predictions on full image using bbox to crop area of interest before
        running the model.
        Returns keypoints in pixel coordinates of the image
        """
        with image.crop(box=bbox.as_coordinates_tuple) as crop:
            polygon = self.predict(crop)
            return polygon.translate(bbox.left, bbox.top)


class HandPredictorLocal(HandPredictor):
    def __init__(self, model_path, confidence_threshold: float = 0.5):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.input_size = tuple(self.model.inputs[0].shape[1:3])
        self.output_size = tuple(self.model.outputs[0].shape[1:3])
        super().__init__(confidence_threshold=confidence_threshold)

    def _batch_predict(self, batch_image_numpy: np.ndarray) -> np.ndarray:
        predicted = self.model.predict(batch_image_numpy, verbose=0)
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

    def predict(self, image: Union[ImageType, np.ndarray]) -> List[BBox]:
        """Run object detection on the input image and draw the detection results"""
        image_width, image_height = _get_image_shape(image)
        image_np = np.expand_dims(np.array(image), axis=0)
        predictions = self._batch_predict(image_np)[0]
        bboxes = []
        for box, cls, score in zip(
            predictions[:, :4], predictions[:, 4], predictions[:, 5]
        ):

            # cast everything to float to prevent issues with JSON serialization
            # and stick to types specified by BBox class
            float_bbox = list(map(float, box))
            # TODO integrate bbox scaling with model export - models should return
            #  outputs in normalized coordinates in corners format
            clip_bbox = BBox(0, 0, 512, 512)

            bbox = (
                BBox.from_ltwh(
                    *float_bbox, name=self.class_to_label_name[cls], score=float(score)
                )
                .intersection(clip_bbox)
                .scale(x=image_width / 512, y=image_height / 512)
            )
            bboxes.append(bbox)
        return bboxes

    def predict_and_plot(
        self, image: Union[ImageType, np.ndarray], ax=None
    ) -> List[BBox]:
        bboxes = self.predict(image)
        ax = ax or plt.gca()
        ax.imshow(image)
        for bbox in bboxes:
            bbox.plot(ax=ax)
        return bboxes


class RetinanetDetectorLocal(RetinanetDetector):
    def __init__(
        self,
        model_path: Path,
        class_to_label_name: Dict[int, str],
    ):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        super().__init__(class_to_label_name=class_to_label_name)

    def _batch_predict(self, input_images):
        return self.model.predict(input_images, False, verbose=0)


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
        # TODO this will require updates
        request.inputs["image"].CopyFrom(tf.make_tensor_proto(input_images))
        result = self.stub.Predict(request, self.timeout)
        output = result.outputs["bboxes"]
        value = output.float_val
        shape = [dim.size for dim in output.tensor_shape.dim]
        return np.array(value).reshape(shape)


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


def read_time(
    polygon: Polygon,
    points: List[Point],
    image_shape: Tuple[int, int],
    debug: bool = False,
    debug_image=None,
    hands_mask=None,
) -> Optional[Tuple[float, float]]:
    detected_center_points = [p for p in points if p.name == "Center"]
    if detected_center_points:
        center = sorted(detected_center_points, key=lambda p: p.score, reverse=True)[0]
    else:
        return None

    detected_top_points = [p for p in points if p.name == "Top"]
    if detected_top_points:
        top = sorted(detected_top_points, key=lambda p: p.score, reverse=True)[0]
    else:
        return None
    if hands_mask is not None:
        mask = hands_mask
    else:
        mask = polygon.to_binary_mask(image_shape[0], image_shape[1])

    # TODO this might be faster if image shape is smaller
    segmented_hands = segment_hands_mask(mask, center=center, debug=debug)
    points_per_hand = []
    for mask in segmented_hands:
        points = mask_to_point_list(mask)
        points_per_hand.append(points)
    # TODO fitting lines as main blob axis with sklearn.measure might work better
    #  https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html#sphx-glr-auto-examples-segmentation-plot-regionprops-py
    hands = _fit_lines_to_points(points_per_hand, center)
    valid_lines, other_lines = line_selector(hands, center=center)
    if debug:
        fig, axarr = plt.subplots(1, 1)
        axarr.imshow(draw_masks(np.array(debug_image), segmented_hands))
        for hand in valid_lines:
            hand.plot(ax=axarr, color="green")
            hand.end.plot(ax=axarr)
        for hand in other_lines:
            hand.plot(ax=axarr, color="red")
        center.plot(ax=axarr, color="magenta")
        top.plot(ax=axarr, color="yellow")

    if valid_lines:
        pred_minute, pred_hour = valid_lines
        minute_kp = dataclasses.replace(pred_minute.end, name="Minute")
        hour_kp = dataclasses.replace(pred_hour.end, name="Hour")
        return points_to_time(center, hour_kp, minute_kp, top, debug=debug)
    return None


def mask_to_point_list(mask):
    points = []
    for i, row in enumerate(mask):
        for j, value in enumerate(row):
            if value > 0:
                points.append(Point(j, i))
    return points


class TimePredictor:
    def __init__(
        self,
        detector: RetinanetDetector,
        kp_predictor: KPHeatmapPredictorV2,
        hand_predictor: HandPredictor,
    ):
        self.detector = detector
        self.kp_predictor = kp_predictor
        self.hand_predictor = hand_predictor

    def predict(self, image: ImageType) -> List[BBox]:
        pil_img = image.convert("RGB")
        bboxes = self.detector.predict(pil_img)
        transcriptions = []

        bboxes = [dataclasses.replace(bbox, name="WatchFace") for bbox in bboxes]
        for box in bboxes:
            with image.crop(box=box.as_coordinates_tuple) as crop:
                crop.thumbnail((256, 256), BICUBIC)
                points = self.kp_predictor.predict(crop)
                hands_polygon = self.hand_predictor.predict(crop)

                time = read_time(
                    hands_polygon,
                    points,
                    crop.size,
                )
            if time:
                read_hour, read_minute = time
                predicted_time = f"{read_hour:02.0f}:{read_minute:02.0f}"
                transcriptions.append(predicted_time)
            else:
                transcriptions.append("??:??")
        bboxes = [
            bbox.rename(transcription)
            for bbox, transcription in zip(bboxes, transcriptions)
        ]
        return bboxes

    def predict_and_plot(self, img):
        fig, axarr = plt.subplots(1, 1)
        bboxes = self.detector.predict_and_plot(img, ax=axarr)
        for i, bbox in enumerate(bboxes):
            with img.crop(box=bbox.as_coordinates_tuple) as crop:
                crop.thumbnail((256, 256), BICUBIC)
                fig, axarr = plt.subplots(1, 3)
                axarr[0].axis("off")
                points = self.kp_predictor.predict_and_plot(
                    crop,
                    ax=axarr[0],
                )
                hands_polygon = self.hand_predictor.predict_and_plot(
                    crop,
                    ax=axarr[1],
                )
                self.hand_predictor.predict_mask_and_draw(crop, ax=axarr[2])

                time = read_time(
                    hands_polygon, points, crop.size, debug=True, debug_image=crop
                )
                if time is not None:
                    predicted_time = f"{time[0]:02.0f}:{time[1]:02.0f}"
                    fig.suptitle(predicted_time, fontsize=16)


def _get_image_shape(image: Union[ImageType, np.ndarray]):
    if isinstance(image, ImageType):
        image_width = image.width
        image_height = image.height
    elif isinstance(image, np.ndarray):
        image_width = image.shape[1]
        image_height = image.shape[0]
    else:
        raise ValueError(
            f"image type {type(image)} is not supported, please provide PIL.Image or np.array"
        )
    return image_width, image_height
