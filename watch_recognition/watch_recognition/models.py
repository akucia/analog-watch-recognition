from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import euclidean_distances

from watch_recognition.utilities import Point


def hour_diff(y_true, y_pred):
    diff = tf.round(tf.abs(y_true - y_pred) * 12)
    return tf.reduce_mean(diff, axis=-1)


def minutes_diff(y_true, y_pred):
    diff = tf.round(tf.abs(y_true - y_pred) * 60)
    return tf.reduce_mean(diff, axis=-1)


def export_tflite(model, export_path, quantize: bool = False, test_image=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("quantaizing model")

        # TODO allow to use more images
        def representative_dataset():
            for data in (
                tf.data.Dataset.from_tensor_slices(test_image).batch(1).take(100)
            ):
                yield [tf.dtypes.cast(data, tf.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    else:
        optimizations = [tf.lite.Optimize.DEFAULT]
        converter.optimizations = optimizations

    tflite_model = converter.convert()

    with tf.io.gfile.GFile(export_path, "wb") as f:
        f.write(tflite_model)
    print(f"model exported to {export_path}")


def build_backbone(image_size, backbone_layer="block5c_project_conv"):
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(*image_size, 3),
        include_top=False,
    )
    outputs = [
        base_model.get_layer(layer_name).output for layer_name in [backbone_layer]
    ]
    return tf.keras.Model(inputs=[base_model.inputs], outputs=outputs)


def decode_single_point(mask) -> Optional[Tuple[float, float]]:
    if mask.sum() == 0:
        mask = np.ones_like(mask)
    y_idx, x_idx = np.indices(mask.shape) + 0.5
    x_mask = np.average(x_idx.flatten(), weights=mask.flatten())
    y_mask = np.average(y_idx.flatten(), weights=mask.flatten())
    return x_mask, y_mask


def extract_points_from_map(
    predicted_map,
    detection_threshold=0.5,
    text_threshold=0.5,
    size_threshold=1,
) -> List[Point]:
    """
    Inspired by keras-ocr segmentation to bboxes code
    https://github.com/faustomorales/keras-ocr/blob/6473e146dc3fc2c386c595efccb55abe558b2529/keras_ocr/detection.py#L207
    Args:
        predicted_map:
        detection_threshold:
        text_threshold:
        size_threshold:

    Returns:

    """

    _, text_score = cv2.threshold(
        predicted_map, thresh=text_threshold, maxval=1, type=cv2.THRESH_BINARY
    )
    n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.clip(text_score, 0, 1).astype("uint8"), connectivity=4
    )
    points = []
    for component_id in range(1, n_components):
        # Filter by size
        size = stats[component_id, cv2.CC_STAT_AREA]
        if size < size_threshold:
            continue

        mean_score = np.mean(predicted_map[labels == component_id])
        if mean_score < detection_threshold:
            continue

        segmap = np.where(
            labels == component_id, predicted_map, np.zeros_like(predicted_map)
        )

        box_center = np.array(decode_single_point(segmap))
        points.append(Point(*box_center, score=float(mean_score)))
    return points


def convert_outputs_to_keypoints(
    predicted,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks = predicted.transpose((2, 0, 1))
    center_points = extract_points_from_map(masks[0])
    if not center_points:
        center_points = [Point(0, 0, "", 0)]
    center_point = sorted(center_points, key=lambda x: x.score)[-1]
    center = np.array(center_point.as_coordinates_tuple)
    # Top
    top_points = extract_points_from_map(masks[1])
    if not top_points:
        top_points = [Point(0, 0, "", 0)]
    top_point = sorted(top_points, key=lambda x: x.score)[-1]
    top = np.array(top_point.as_coordinates_tuple)
    # Hands

    hands_points = extract_points_from_map(masks[2])
    if not hands_points:
        hands_points = [Point(0, 0, "", 0), Point(0, 0, "", 0)]
    hands_points = sorted(hands_points, key=lambda x: x.score)[-2:]
    hands_points = np.array([p.as_coordinates_tuple for p in hands_points])
    hour, minute = get_minute_and_hour_points(center, hands_points)
    return center, hour, minute, top


def get_minute_and_hour_points(center, hand_points):
    distances = euclidean_distances(hand_points, [center])
    hour_index = np.argmin(distances)
    hour = hand_points[hour_index]
    minute = hand_points[np.argmax(distances)]
    return hour, minute


def points_to_time(center, hour, minute, top):
    hour = hour - center
    minute = minute - center
    top = top - center
    read_hour = (
        np.rad2deg(np.arctan2(top[0], top[1]) - np.arctan2(hour[0], hour[1])) / 360 * 12
    )
    read_hour = np.floor(read_hour).astype(int)
    read_hour = read_hour % 12
    if read_hour == 0:
        read_hour = 12
    read_minute = (
        np.rad2deg(np.arctan2(top[0], top[1]) - np.arctan2(minute[0], minute[1]))
        / 360
        * 60
    )
    read_minute = np.floor(read_minute).astype(int)
    read_minute = read_minute % 60
    return read_hour, read_minute


def custom_focal_loss(target, output, gamma=4, alpha=0.25):
    epsilon_ = tf.keras.backend.epsilon()
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    bce = tf.pow(target, gamma) * tf.math.log(output + epsilon_) * (1 - alpha)
    bce += tf.pow(1 - target, gamma) * tf.math.log(1 - output + epsilon_) * alpha
    return -bce
