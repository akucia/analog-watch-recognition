import dataclasses
from itertools import combinations
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from skimage.draw import circle_perimeter, disk
from skimage.filters import gaussian
from sklearn.metrics import euclidean_distances
from tensorflow.python.keras.utils.np_utils import to_categorical

from watch_recognition.data_preprocessing import binarize, keypoints_to_angle
from watch_recognition.utilities import Line, Point


def set_shapes(img, target, img_shape=(224, 224, 3), target_shape=(28, 28, 4)):
    img.set_shape(img_shape)
    target.set_shape(target_shape)
    return img, target


def encode_keypoints_to_mask_np(
    keypoints,
    image_size,
    mask_size,
    radius=1,
    include_background=True,
    separate_hour_and_minute_hands: bool = True,
    add_perimeter: bool = False,
    sparse: bool = True,
    with_perimeter_for_hands: bool = False,
    blur: bool = False,
):
    downsample_factor = image_size[0] / mask_size[0]
    all_masks = []
    points = keypoints[:, :2]
    fm_point = points / downsample_factor
    int_points = np.floor(fm_point).astype(int)
    # center and top
    for int_point in int_points[:2]:
        mask = _encode_point_to_mask(radius * 2, int_point, mask_size, add_perimeter)
        if blur:
            mask = _blur_mask(mask)
        all_masks.append(mask)
    # hour and minute hands
    if separate_hour_and_minute_hands:
        for int_point in int_points[2:]:
            mask = _encode_point_to_mask(
                radius, int_point, mask_size, with_perimeter_for_hands
            )
            if blur:
                mask = _blur_mask(mask)
            all_masks.append(mask)
    else:
        mask = _encode_multiple_points_to_mask(
            radius, int_points[2:], mask_size, with_perimeter_for_hands
        )
        if blur:
            mask = _blur_mask(mask)
        all_masks.append(mask)

    masks = np.array(all_masks).transpose((1, 2, 0))
    if include_background:
        background_mask = ((np.ones(mask_size) - masks.sum(axis=-1)) > 0).astype(
            "float32"
        )
        background_mask = np.expand_dims(background_mask, axis=-1)
        masks = np.concatenate((masks, background_mask), axis=-1)
    if sparse:
        masks = np.expand_dims(np.argmax(masks, axis=-1), axis=-1)
    return masks.astype("float32")


def _blur_mask(mask):
    mask = gaussian(
        mask,
        sigma=2,
    )
    mask = mask / (np.max(mask) + 1e-8)
    return mask


def _encode_multiple_points_to_mask(extent, int_points, mask_size, with_perimeter):
    mask = np.zeros(mask_size, dtype=np.float32)
    for int_point in int_points:
        mask += _encode_point_to_mask(extent, int_point, mask_size, with_perimeter)
    masks_clipped = np.clip(mask, 0, 1)
    return masks_clipped


def _encode_point_to_mask(radius, int_point, mask_size, with_perimeter: bool = False):
    mask = np.zeros(mask_size, dtype=np.float32)
    coords = tuple(int_point)
    rr, cc = disk(coords, radius)
    cc, rr = select_rows_and_columns_inside_mask(cc, mask_size, rr)
    mask[cc, rr] = 1
    if with_perimeter:
        rr, cc = circle_perimeter(*coords, radius)
        cc, rr = select_rows_and_columns_inside_mask(cc, mask_size, rr)
        mask[cc, rr] = 1
    return mask


def encode_keypoints_to_mask(
    image,
    keypoints,
    image_size,
    mask_size,
    radius,
    include_background=True,
    separate_hour_and_minute_hands=False,
    add_perimeter=False,
    sparse=False,
    with_perimeter_for_hands: bool = False,
    blur: bool = False,
):
    mask = tf.numpy_function(
        func=encode_keypoints_to_mask_np,
        inp=[
            keypoints,
            image_size,
            mask_size,
            radius,
            include_background,
            separate_hour_and_minute_hands,
            add_perimeter,
            sparse,
            with_perimeter_for_hands,
            blur,
        ],
        Tout=tf.float32,
    )
    return image, mask


def add_sample_weights(image, label):
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([1, 1, 1, 1e-3])
    class_weights = class_weights / tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


def encode_keypoints_to_angle(image, keypoints, bin_size=90):
    angle = tf.numpy_function(
        func=encode_keypoints_to_angle_np,
        inp=[
            keypoints,
            bin_size,
        ],
        Tout=tf.float32,
    )
    return image, angle


def encode_keypoints_to_angle_np(keypoints, bin_size=90):
    center = keypoints[0, :2]
    top = keypoints[1, :2]
    angle = keypoints_to_angle(center, top)
    angle = binarize(angle, bin_size)
    return to_categorical(angle, num_classes=360 // bin_size)


def decode_single_point(mask, threshold=0.1) -> Point:
    mask = np.where(mask < threshold, np.zeros_like(mask), mask)
    if mask.sum() == 0:
        mask = np.ones_like(mask)
    y_idx, x_idx = np.indices(mask.shape)
    x_mask = np.average(x_idx.flatten(), weights=mask.flatten())
    y_mask = np.average(y_idx.flatten(), weights=mask.flatten())

    return Point(x_mask, y_mask, score=float(mask.flatten().mean()))


def extract_points_from_map(
    predicted_map,
    detection_threshold=0.5,
    text_threshold=0.5,
    size_threshold=2,
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

        score = np.max(predicted_map[labels == component_id])
        if score < detection_threshold:
            continue

        segmap = np.where(
            labels == component_id, predicted_map, np.zeros_like(predicted_map)
        )

        box_center = np.array(decode_single_point(segmap).as_coordinates_tuple)
        points.append(Point(*box_center, score=float(score)))
    return points


def convert_mask_outputs_to_keypoints(
    predicted: np.ndarray,
    return_all_hand_points: bool = False,
    experimental_hands_decoding: bool = False,
) -> Tuple[Point, ...]:
    masks = predicted.transpose((2, 0, 1))

    center = decode_single_point(masks[0])
    center = dataclasses.replace(center, name="Center")

    # Top
    top_points = extract_points_from_map(
        masks[1],
    )
    if not top_points:
        top_points = [decode_single_point(masks[1])]

    top = sorted(top_points, key=lambda x: x.score)[-1]
    top = dataclasses.replace(top, name="Top")
    # Hands

    hands_map = masks[2]
    hands_points = extract_points_from_map(
        predicted_map=hands_map,
        size_threshold=4,
        detection_threshold=0.15,
        text_threshold=0.15,
    )
    if return_all_hand_points:
        points = (center, top, *hands_points)
        return points

    if experimental_hands_decoding:
        lines = []
        used_points = set()
        for a, b in combinations(hands_points, 2):
            line = Line(a, b)
            proj_point = line.projection_point(center)
            d = proj_point.distance(center)
            if d < 1:
                lines.append(line)
                used_points.add(a)
                used_points.add(b)
        unused_points = [p for p in hands_points if p not in used_points]
        for point in unused_points:
            lines.append(Line(point, center))

        best_lines = sorted(lines, key=lambda l: l.length)[:2]
        hands = []
        for line in best_lines:
            if line.start.distance(center) > line.end.distance(center):
                hands.append(line.start)
            else:
                hands.append(line.end)
        hour, minute = get_minute_and_hour_points(center, tuple(hands))
        points = (center, top, hour, minute)
        return points

    if not hands_points:
        hands_points = [Point.none(), Point.none()]
    if len(hands_points) == 1:
        hands_points = (hands_points[0], hands_points[0])
    hands_points = sorted(hands_points, key=lambda x: x.score)[-2:]
    hour, minute = get_minute_and_hour_points(center, tuple(hands_points))
    hour = dataclasses.replace(hour, name="Hour")
    minute = dataclasses.replace(minute, name="Minute")

    return center, top, hour, minute


def poly_area(x, y):
    """https://stackoverflow.com/a/30408825/8814045"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def select_minute_and_hour_points(
    center: Point, hand_points: List[Point]
) -> Tuple[Point, Point]:
    point_combinations = list(combinations(hand_points, 2))
    areas = [
        poly_area(np.array([center.x, a.x, b.x]), np.array([center.y, a.y, b.y]))
        for a, b in point_combinations
    ]
    sort = np.argsort(areas)
    idx = sort[-1]

    return point_combinations[idx]


def get_minute_and_hour_points(
    center: Point, hand_points: Tuple[Point, Point]
) -> Tuple[Point, Point]:
    assert len(hand_points) < 3, "expected max 2 points for hands"
    hand_points_np = np.array([p.as_coordinates_tuple for p in hand_points]).reshape(
        -1, 2
    )
    center = np.array(center.as_coordinates_tuple).reshape(1, -1)
    distances = euclidean_distances(hand_points_np, center)
    hour = hand_points[int(np.argmin(distances))]
    minute = hand_points[int(np.argmax(distances))]
    return hour, minute


def select_rows_and_columns_inside_mask(cc, mask_size, rr):
    row_filter = np.where(
        (0 <= rr) & (rr < mask_size[0]),
        np.ones_like(rr).astype(bool),
        np.zeros_like(rr).astype(bool),
    )
    col_filter = np.where(
        (0 <= cc) & (cc < mask_size[1]),
        np.ones_like(cc).astype(bool),
        np.zeros_like(cc).astype(bool),
    )
    filter = row_filter & col_filter
    cc = cc[filter]
    rr = rr[filter]
    return cc, rr
