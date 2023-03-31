import abc
import dataclasses
import hashlib
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import grpc
import numpy as np
from distinctipy import distinctipy
from matplotlib import pyplot as plt
from more_itertools import chunked
from PIL.Image import BICUBIC
from PIL.Image import Image as ImageType
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tensorflow_serving.apis.predict_pb2 import PredictResponse

import tensorflow as tf
from watch_recognition.targets_encoding import (
    _fit_lines_to_points,
    extract_points_from_map,
    line_selector,
    segment_hands_mask,
)
from watch_recognition.utilities import BBox, Line, Point, Polygon
from watch_recognition.visualization import draw_masks


class KPHeatmapPredictorV2(ABC):
    """Base class for keypoint predictors based on heatmap outputs"""

    def __init__(self, class_to_label_name, confidence_threshold: float = 0.5):
        """

        Args:
            class_to_label_name: mapping from class index to label name
            confidence_threshold: threshold for heatmap values to be considered as a keypoint
        """
        self.confidence_threshold = confidence_threshold
        self.class_to_label_name = class_to_label_name

    @abc.abstractmethod
    def _batch_predict(self, batch_image_np: np.ndarray) -> np.ndarray:
        """Runs predictions on a batch of images."""
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

    def predict_and_plot(
        self, image: Union[ImageType, np.ndarray], ax=None
    ) -> List[Point]:
        """Runs predictions on a crop of a watch face and plot resulting keypoints"""
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
    """Predictor for keypoint heatmaps based on local model"""

    def __init__(
        self, model_path, class_to_label_name, confidence_threshold: float = 0.5
    ):
        """

        Args:
            model_path: exported saved model directory
            class_to_label_name: mapping from class index to label name
            confidence_threshold: threshold for heatmap values to be considered as a keypoint
        """
        self.model: tf.keras.models.Model = tf.keras.models.load_model(
            model_path, compile=False
        )
        super().__init__(class_to_label_name, confidence_threshold)

    def _batch_predict(self, batch_image_np: np.ndarray) -> np.ndarray:
        predicted = self.model.predict(batch_image_np, verbose=0)
        return predicted


class KPHeatmapPredictorV2GRPC(KPHeatmapPredictorV2):
    """Predictor for keypoint heatmaps based on remote model served via TensorflowServing and gRPC API"""

    def __init__(
        self,
        host: str,
        model_name: str,
        class_to_label_name,
        confidence_threshold: float = 0.5,
        timeout: float = 10.0,
    ):
        """

        Args:
            host: address of model host
            model_name: name of the model served by TensorflowServing
            class_to_label_name: mapping from class index to label name
            confidence_threshold: threshold for heatmap values to be considered as a keypoint
            timeout: gRPC API timeout
        """
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

        weights = predicted.flatten() > self.confidence_threshold
        if weights.sum() > 0:
            mean_score = np.average(
                predicted.flatten(),
                weights=weights,
            )
        else:
            mean_score = 0.0
        predicted = predicted > self.confidence_threshold
        polygon = Polygon.from_binary_mask(predicted)
        polygon = dataclasses.replace(polygon, score=float(mean_score))

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


class RetinaNetDetector(ABC):
    def __init__(
        self,
        class_to_label_name: Dict[int, str],
        confidence_threshold: float = 0.5,
    ):
        self.class_to_label_name = class_to_label_name
        self.confidence_threshold = confidence_threshold

    @abc.abstractmethod
    def _batch_predict_api(
        self, batch_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def predict(self, image: Union[ImageType, np.ndarray]) -> List[BBox]:
        """Run object detection on the input image and convert results to BBox objects"""
        return self.batch_predict([image])[0]

    def _batch_predict(
        self, images: List[Union[ImageType, np.ndarray]]
    ) -> List[List[BBox]]:
        image_width, image_height = _get_image_shape(images[0])
        batch_images = np.array([np.array(image) for image in images])
        results = self._batch_predict_api(batch_images)
        all_bboxes = []
        for boxes, classes, scores, num_detections in zip(*results):
            image_bboxes = []
            for box, cls, score in zip(boxes, classes, scores):
                if score < self.confidence_threshold:
                    continue
                bbox = self._bbox_from_outputs(
                    box, cls, image_height, image_width, score
                )
                image_bboxes.append(bbox)
            all_bboxes.append(image_bboxes)
        return all_bboxes

    def batch_predict(
        self,
        images: List[Union[ImageType, np.ndarray]],
        maximum_request_batch_size: int = 10,
    ) -> List[List[BBox]]:
        # TODO add assert - all images must have the same size
        all_bboxes = []
        for batch in chunked(images, maximum_request_batch_size):
            all_bboxes.extend(self._batch_predict(batch))
        return all_bboxes

    def _bbox_from_outputs(self, box, cls, image_height, image_width, score):
        # cast everything to float to prevent issues with JSON serialization
        # and stick to types specified by BBox class
        float_bbox = list(map(float, box))
        # TODO integrate bbox scaling with model export - models should return
        #  outputs in normalized coordinates in corners format
        # model outputs bboxes have ys as first and third coordinate
        ymin, xmin, ymax, xmax = float_bbox
        bbox = BBox(
            xmin,
            ymin,
            xmax,
            ymax,
            name=self.class_to_label_name[cls],
            score=float(score),
        ).scale(x=image_width, y=image_height)
        return bbox

    def predict_and_plot(
        self, image: Union[ImageType, np.ndarray], ax=None
    ) -> List[BBox]:
        bboxes = self.predict(image)
        ax = ax or plt.gca()
        ax.imshow(image)
        ax.grid(False)
        for bbox in bboxes:
            bbox.plot(ax=ax, draw_label=True, draw_score=True)
        return bboxes


class RetinaNetDetectorLocal(RetinaNetDetector):
    def __init__(
        self,
        model_path: Path,
        class_to_label_name: Dict[int, str],
        confidence_threshold: float = 0.5,
    ):
        imported = tf.saved_model.load(model_path)
        self.model_fn = imported.signatures["serving_default"]
        super().__init__(
            class_to_label_name=class_to_label_name,
            confidence_threshold=confidence_threshold,
        )

    def _batch_predict_api(
        self, input_images
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        input_images = tf.convert_to_tensor(input_images)
        results = self.model_fn(input_images)
        num_detections = results["num_detections"].numpy()
        boxes = results["detection_boxes"].numpy()
        classes = results["detection_classes"].numpy().astype(int)
        scores = results["detection_scores"].numpy()
        return (
            boxes,
            classes,
            scores,
            num_detections,
        )


class RetinaNetDetectorGRPC(RetinaNetDetector):
    def __init__(
        self,
        host: str,
        model_name: str,
        class_to_label_name: Dict[int, str],
        confidence_threshold: float = 0.5,
        timeout: float = 10.0,
    ):
        self.host = host
        self.model_name = model_name
        self.channel = grpc.insecure_channel(self.host)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.timeout = timeout
        super().__init__(
            class_to_label_name=class_to_label_name,
            confidence_threshold=confidence_threshold,
        )

    def _batch_predict_api(
        self, input_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"
        request.inputs["inputs"].CopyFrom(tf.make_tensor_proto(input_images))
        result = self.stub.Predict(request, self.timeout)
        boxes = tf_serving_response_proto_to_numpy(
            result.outputs["detection_boxes"].float_val,
            result.outputs["detection_boxes"].tensor_shape.dim,
        )
        classes = tf_serving_response_proto_to_numpy(
            result.outputs["detection_classes"].int_val,
            result.outputs["detection_classes"].tensor_shape.dim,
        )
        scores = tf_serving_response_proto_to_numpy(
            result.outputs["detection_scores"].float_val,
            result.outputs["detection_scores"].tensor_shape.dim,
        )
        num_detections = tf_serving_response_proto_to_numpy(
            result.outputs["num_detections"].int_val,
            result.outputs["num_detections"].tensor_shape.dim,
        )
        return (
            boxes,
            classes,
            scores,
            num_detections,
        )


class RetinaNetLiteDetector(RetinaNetDetector):
    @property
    @abc.abstractmethod
    def input_shape(self) -> Tuple[int, int, int]:
        pass

    def _batch_predict_api(
        self, batch_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
        pass
        # TODO there's no real batch prediction for lite models
        # if batch_images.shape[1] != self.input_shape[1]:
        #     raise NotImplementedError(
        #         "Batch prediction is not supported for RetinaNetLite models"
        #     )
        # # TODO implementation of inputs validation, float32 conversion, image resizing
        # #   and outputs scaling

        # TODO image has to be float32
        # boxes, classes, scores, num_detections = self._batch_predict_api(batch_images)
        # # lite models don't normalize boxes
        # boxes = boxes.reshape(-1, 4)
        # boxes[:, 0] = boxes[:, 0] / self.input_shape[0]  # ymin
        # boxes[:, 1] = boxes[:, 1] / self.input_shape[1]  # xmin
        # boxes[:, 2] = boxes[:, 2] / self.input_shape[0]  # ymax
        # boxes[:, 3] = boxes[:, 3] / self.input_shape[1]  # xmax
        #
        # boxes = boxes.reshape(batch_images.shape[0], -1, 4)
        # return boxes, classes, scores, num_detections


class RetinaNetLiteDetectorGRPC(RetinaNetLiteDetector):
    def __init__(
        self,
        host: str,
        model_name: str,
        input_shape: Tuple[int, int, int],
        class_to_label_name: Dict[int, str],
        confidence_threshold: float = 0.5,
        timeout: float = 10.0,
    ):
        self.host = host
        self.model_name = model_name
        self.channel = grpc.insecure_channel(self.host)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self._input_shape = input_shape
        self.timeout = timeout
        super().__init__(
            class_to_label_name=class_to_label_name,
            confidence_threshold=confidence_threshold,
        )

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    def _batch_predict_api(
        self, input_images: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"
        request.inputs["inputs"].CopyFrom(
            tf.make_tensor_proto(input_images, shape=input_images.shape)
        )
        result = self.stub.Predict(request)
        boxes = tf_serving_response_proto_to_numpy(
            result.outputs["Identity"].float_val,
            result.outputs["Identity"].tensor_shape.dim,
        )

        classes = tf_serving_response_proto_to_numpy(
            result.outputs["Identity_1"].int_val,
            result.outputs["Identity_1"].tensor_shape.dim,
        )
        scores = tf_serving_response_proto_to_numpy(
            result.outputs["Identity_2"].float_val,
            result.outputs["Identity_2"].tensor_shape.dim,
        )
        num_detections = tf_serving_response_proto_to_numpy(
            result.outputs["Identity_4"].int_val,
            result.outputs["Identity_4"].tensor_shape.dim,
        )

        # lite models don't normalize boxes
        boxes = boxes.reshape(-1, 4)
        boxes[:, 0] = boxes[:, 0] / self.input_shape[0]  # ymin
        boxes[:, 1] = boxes[:, 1] / self.input_shape[1]  # xmin
        boxes[:, 2] = boxes[:, 2] / self.input_shape[0]  # ymax
        boxes[:, 3] = boxes[:, 3] / self.input_shape[1]  # xmax

        boxes = boxes.reshape(input_images.shape[0], -1, 4)

        return boxes, classes, scores, num_detections


class RetinaNetLiteDetectorLocal(RetinaNetLiteDetector):
    def __init__(
        self,
        model_path: Union[Path, str],
        class_to_label_name: Dict[int, str],
        confidence_threshold: float = 0.5,
    ):
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.interpreter = interpreter
        self._input_shape = input_details[0]["shape"][1:]
        self.input_index = input_details[0]["index"]

        output_details = interpreter.get_output_details()
        self.bbox_index = output_details[0]["index"]
        self.class_index = output_details[1]["index"]
        self.score_index = output_details[2]["index"]
        self.image_info_index = output_details[3]["index"]
        self.num_detections_index = output_details[4]["index"]

        super().__init__(
            class_to_label_name=class_to_label_name,
            confidence_threshold=confidence_threshold,
        )

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    def _batch_predict_api(
        self, input_images
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # TODO info about images api, data types etc
        self.interpreter.set_tensor(self.input_index, input_images)

        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.bbox_index)
        scores = self.interpreter.get_tensor(self.score_index)
        classes = self.interpreter.get_tensor(self.class_index)
        num_detections = self.interpreter.get_tensor(self.num_detections_index)
        # lite models don't normalize boxes
        boxes = boxes.reshape(-1, 4)
        boxes[:, 0] = boxes[:, 0] / self.input_shape[0]  # ymin
        boxes[:, 1] = boxes[:, 1] / self.input_shape[1]  # xmin
        boxes[:, 2] = boxes[:, 2] / self.input_shape[0]  # ymax
        boxes[:, 3] = boxes[:, 3] / self.input_shape[1]  # xmax

        boxes = boxes.reshape(input_images.shape[0], -1, 4)
        return (
            boxes,
            classes,
            scores,
            num_detections,
        )


def tf_serving_response_proto_to_numpy(value, dim):
    shape = [d.size for d in dim]
    output = np.array(value).reshape(shape)
    if len(shape) != 1:
        output = output.reshape(shape)
    return output


def _hash_image(image: ImageType) -> str:
    md5hash = hashlib.md5(image.tobytes(), usedforsecurity=False)
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
    hands_mask=None,
) -> Tuple[Optional[Tuple[float, float]], Tuple[Line, Line]]:
    detected_center_points = [p for p in points if p.name == "Center"]
    if detected_center_points:
        center = sorted(detected_center_points, key=lambda p: p.score, reverse=True)[0]
    else:
        return None, tuple()

    detected_top_points = [p for p in points if p.name == "Top"]
    if detected_top_points:
        top = sorted(detected_top_points, key=lambda p: p.score, reverse=True)[0]
    else:
        return None, tuple()
    if hands_mask is not None:
        mask = hands_mask
    else:
        mask = polygon.to_mask(image_shape[0], image_shape[1])

    # TODO this might be faster if image shape is smaller
    segmented_hands = segment_hands_mask(mask, center=center)
    points_per_hand = []
    for mask in segmented_hands:
        points = mask_to_point_list(mask)
        points_per_hand.append(points)
    # TODO fitting lines as main blob axis with sklearn.measure might work better
    #  https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html#sphx-glr-auto-examples-segmentation-plot-regionprops-py
    hands = _fit_lines_to_points(points_per_hand, center)
    valid_lines, other_lines = line_selector(hands, center=center)

    if valid_lines:
        pred_minute, pred_hour = valid_lines
        minute_kp = pred_minute.end
        hour_kp = pred_hour.end
        return points_to_time(center, hour_kp, minute_kp, top), valid_lines
    return None, tuple()


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
        detector: RetinaNetDetector,
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

                time, _ = read_time(
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

    def predict_debug(
        self, image: ImageType
    ) -> Tuple[List[BBox], List[Point], List[Line]]:
        # TODO DRY refactor this function and predict
        pil_img = image.convert("RGB")
        bboxes = self.detector.predict(pil_img)
        transcriptions = []

        bboxes = [dataclasses.replace(bbox, name="WatchFace") for bbox in bboxes]
        all_points = []
        all_lines = []
        for box in bboxes:
            with image.crop(box=box.as_coordinates_tuple) as crop:
                crop.thumbnail((256, 256), BICUBIC)
                points = self.kp_predictor.predict(crop)
                hands_polygon = self.hand_predictor.predict(crop)

                time, lines = read_time(
                    hands_polygon,
                    points,
                    crop.size,
                )
                # scale and move lines and points to image size
                points = [
                    p.scale(box.width / crop.width, box.height / crop.height).translate(
                        box.left, box.top
                    )
                    for p in points
                ]
                lines = [
                    line.scale(
                        box.width / crop.width, box.height / crop.height
                    ).translate(box.left, box.top)
                    for line in lines
                ]
                all_points.extend(points)
                # TODO lines are not originating from the center point
                all_lines.extend(lines)
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
        return bboxes, all_points, all_lines

    def predict_and_plot(self, img):
        fig, axarr = plt.subplots(1, 1)
        bboxes = self.detector.predict_and_plot(img, ax=axarr)
        for i, bbox in enumerate(bboxes):
            with img.crop(box=bbox.as_coordinates_tuple) as crop:
                crop.thumbnail((256, 256), BICUBIC)
                fig, axarr = plt.subplots(1, 3)
                axarr[0].axis("off")
                points = self.kp_predictor.predict_and_plot(crop, ax=axarr[0])
                hands_polygon = self.hand_predictor.predict_and_plot(crop, ax=axarr[1])
                self.hand_predictor.predict_mask_and_draw(crop, ax=axarr[2])

                time, lines = read_time(hands_polygon, points, crop.size)
                for line in lines:
                    line.plot(ax=axarr[0], color="red")
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


def points_to_time(
    center: Point, hour: Point, minute: Point, top: Point, debug: bool = False
) -> Tuple[float, float]:
    if debug:
        fig, ax = plt.subplots(1, 1)
        top.plot(color="green")
        hour.plot(color="red")
        center.plot(color="k")
        minute.plot(color="orange")
        ax.invert_yaxis()
        ax.legend()

    hour = hour.translate(-center.x, -center.y)
    minute = minute.translate(-center.x, -center.y)
    top = top.translate(-center.x, -center.y)
    center = center.translate(-center.x, -center.y)
    up = center.translate(0, -10)
    line_up = Line(center, up)
    line_top = Line(center, top)
    if debug:
        fig, ax = plt.subplots(1, 1)
        top.plot(color="green")
        hour.plot(color="red")
        center.plot(color="k")
        minute.plot(color="orange")
        line_up.plot(draw_arrow=False, color="black")
        line_top.plot(draw_arrow=False, color="green")
        ax.invert_yaxis()
        ax.legend()
    rot_angle = -np.rad2deg(line_up.angle_between(line_top))
    if top.x < 0:
        rot_angle = -rot_angle
    top = top.rotate_around_point(center, rot_angle)
    minute = minute.rotate_around_point(center, rot_angle)
    hour = hour.rotate_around_point(center, rot_angle)

    if debug:
        fig, ax = plt.subplots(1, 1)
        top.plot(color="green")
        hour.plot(color="red")
        center.plot(color="k")
        minute.plot(color="orange")
        line_up = Line(center, up)
        line_top = Line(center, top)
        line_up.plot(draw_arrow=False, color="black")
        line_top.plot(draw_arrow=False, color="green")

        ax.invert_yaxis()
        ax.legend()
    hour = hour.as_array
    minute = minute.as_array
    top = top.as_array

    minute_deg = np.rad2deg(angle_between(top, minute))
    # TODO verify how to handle negative angles
    if minute[0] < top[0]:
        minute_deg = 360 - minute_deg
    read_minute = minute_deg / 360 * 60
    read_minute = np.floor(read_minute).astype(int)
    read_minute = read_minute % 60

    hour_deg = np.rad2deg(angle_between(top, hour))
    # TODO verify how to handle negative angles
    if hour[0] < top[0]:
        hour_deg = 360 - hour_deg
    # In case where the minute hand is close to 12
    # we can assume that the hour hand will be close to the next hour
    # to prevent incorrect hour reading we can move it back by 10 deg
    if read_minute > 45:
        hour_deg -= 10

    read_hour = hour_deg / 360 * 12
    read_hour = np.floor(read_hour).astype(int)
    read_hour = read_hour % 12
    if read_hour == 0:
        read_hour = 12
    return read_hour, read_minute


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    https://stackoverflow.com/a/13849249
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
