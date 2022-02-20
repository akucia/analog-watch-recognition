import dataclasses
from collections import defaultdict
from itertools import chain, combinations
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from distinctipy import distinctipy
from matplotlib import pyplot as plt
from numpy import mean
from scipy.optimize import minimize
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths
from skimage.draw import circle_perimeter, disk, line
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KernelDensity
from tensorflow.python.keras.utils.np_utils import to_categorical

from watch_recognition.data_preprocessing import (
    binarize,
    keypoint_to_sin_cos_angle,
    keypoints_to_angle,
)
from watch_recognition.utilities import BBox, Line, Point


def set_shapes(img, target, img_shape=(224, 224, 3), target_shape=(28, 28, 4)):
    img.set_shape(img_shape)
    target.set_shape(target_shape)
    return img, target


def set_shapes_with_sample_weight(
    img, target, weights, img_shape=(224, 224, 3), target_shape=(28, 28, 4)
):
    img.set_shape(img_shape)
    target.set_shape(target_shape)
    weights.set_shape((*target_shape[:-1], 1))
    return img, target, weights


def encode_keypoints_to_mask_np(
    keypoints,
    image_size,
    mask_size,
    radius=1,
    include_background=False,
    separate_hour_and_minute_hands: bool = False,
    add_perimeter: bool = False,
    sparse: bool = False,
    with_perimeter_for_hands: bool = False,
    blur: bool = False,
    hands_as_lines: bool = False,
):
    downsample_factor = image_size[0] / mask_size[0]
    all_masks = []
    points = keypoints[:, :2]
    fm_point = points / downsample_factor
    int_points = np.floor(fm_point).astype(int)
    # center and top
    for int_point in int_points[:2]:
        mask = _encode_point_to_mask(radius, int_point, mask_size, add_perimeter)
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
        if hands_as_lines:
            mask = _encode_multiple_points_to_lines(
                int_points[2:], int_points[0], mask_size, blur
            )

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


def _blur_mask(mask, sigma=3):
    mask = gaussian(
        mask,
        sigma=sigma,
    )
    mask = mask / (np.max(mask) + 1e-8)
    mask = (mask > 0.3).astype(float)

    return mask


def _encode_multiple_points_to_lines(int_points, center, mask_size, blur):
    masks = []
    for int_point in int_points:
        mask = np.zeros(mask_size, dtype=np.float32)
        # TODO make lines thicker, maybe stronger blur? maybe line_aa?
        rr, cc = line(*int_point, *center)
        cc, rr = select_rows_and_columns_inside_mask(cc, mask_size, rr)
        mask[cc, rr] = 1
        if blur:
            mask = _blur_mask(mask)
        masks.append(mask)
    masks = np.stack(masks, axis=-1)
    mask = np.max(masks, axis=-1)
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
    hands_as_lines: bool = False,
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
            hands_as_lines,
        ],
        Tout=tf.float32,
    )
    return image, mask


def add_sample_weights(image, label, class_weights: List[float]):
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights_tf = tf.constant(class_weights)
    class_weights_tf = class_weights_tf / tf.reduce_sum(class_weights_tf)

    # Create an image of `sample_weights` by using the label at each pixel as an
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights_tf, indices=tf.cast(label, tf.int32))

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


def encode_keypoints_to_hands_angles(image, keypoints):
    angle = tf.numpy_function(
        func=encode_keypoints_to_hands_angle,
        inp=[
            keypoints,
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


def encode_keypoints_to_hands_angle(keypoints):
    center = keypoints[0, :2]
    hour = keypoints[2, :2]
    return keypoint_to_sin_cos_angle(center, hour).astype("float32")


def decode_single_point(mask, threshold=0.1) -> Point:
    # this might be faster implementation, and for batch of outputs
    # https://github.com/OlgaChernytska/2D-Hand-Pose-Estimation-RGB/blob/c9f201ca114129fa750f4bac2adf0f87c08533eb/utils/prep_utils.py#L114

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
    decode_hands_from_lines: bool = False,
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
        hands = select_hand_points_with_line_fits(center, hands_points)
        hour, minute = get_minute_and_hour_points(center, tuple(hands))
        points = (center, top, hour, minute)
        return points
    if decode_hands_from_lines:
        hands_points = decode_keypoints_via_line_fits(hands_map, center)

    if not hands_points:
        hands_points = [Point.none(), Point.none()]
    if len(hands_points) == 1:
        hands_points = (hands_points[0], hands_points[0])
    hands_points = sorted(hands_points, key=lambda x: x.score)[-2:]
    hour, minute = get_minute_and_hour_points(center, tuple(hands_points))
    hour = dataclasses.replace(hour, name="Hour")
    minute = dataclasses.replace(minute, name="Minute")

    return center, top, hour, minute


def select_hand_points_with_line_fits(center, hands_points, max_distance=1):
    """
    Finds points that are collinear with the center point to get hand lines lengths.
    Then selects 2 shortest hand lines (to get rid of seconds hand)
    Args:
        center:
        hands_points:
        max_distance:

    Returns:

    """

    lines = []
    used_points = set()
    for a, b in combinations(hands_points, 2):
        if a.distance(b) < a.distance(center):
            continue
        line = Line(a, b)
        proj_point = line.projection_point(center)
        d = proj_point.distance(center)
        if d < max_distance:
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
    return hands


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


# TODO replace with poly1d from numpy/scipy
def linear(x, a=1, b=0):
    return a * x + b


def inverse_mse_line_angle(params, x, y):
    angle_1 = params
    y_1_hat = linear(x, angle_1)
    mse = (y - y_1_hat) ** 2

    return 1 / mse.sum()


def mse_line_angle(params, x, y):
    angle_1 = params
    y_1_hat = linear(x, angle_1)
    mse = (y - y_1_hat) ** 2
    return mse.sum()


def decode_keypoints_via_line_fits(
    mask, center: Point, threshold=0.5, debug: bool = False
) -> Tuple[Point, Point]:
    image = np.where(mask > threshold, np.ones_like(mask), np.zeros_like(mask))

    idx = np.nonzero(image)
    if len(idx) == 0:
        return Point.none(), Point.none()

    x = idx[1] - center.x
    y = idx[0] - center.y
    mean_y = np.mean(y)
    mean_x = np.mean(x)
    mean_tan = mean_y / mean_x
    res = minimize(
        inverse_mse_line_angle,
        x0=np.array([mean_tan]),
        args=(x, y),
    )
    y_0_fit = linear(x, res.x[0])

    if debug:
        print(mean_tan)
        print(res)

        plt.scatter(x, y, marker="x")
        plt.plot(x, y_0_fit)
        plt.show()

    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []

    for x_i, y_i in zip(x, y):
        if y_i > linear(x_i, res.x[0]):
            x_1.append(x_i)
            y_1.append(y_i)
        else:
            x_2.append(x_i)
            y_2.append(y_i)
    x_1 = np.array(x_1)
    y_1 = np.array(y_1)
    x_2 = np.array(x_2)
    y_2 = np.array(y_2)

    p1 = _fit_line_and_get_extreme_point(center, x_1, y_1)
    p2 = _fit_line_and_get_extreme_point(center, x_2, y_2)

    return p1, p2


def _fit_line_and_get_extreme_point(center, x, y):
    if len(x) > 0:
        res = minimize(
            mse_line_angle,
            x0=np.array([0]),
            args=(x, y),
        )
        x_1_max = x[np.argmax(np.abs(x))]
        y_1_fit = linear(x_1_max, res.x[0])
        p1 = Point(x_1_max + center.x, y_1_fit + center.y, score=1)
    else:
        p1 = Point.none()
    return p1


def vonmises_kde(data, kappa, n_bins=100):
    """https://stackoverflow.com/a/44783738/8814045"""
    from scipy.special import i0

    bins = np.linspace(-np.pi, np.pi, n_bins)
    x = np.linspace(-np.pi, np.pi, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa * np.cos(x[:, None] - data[None, :])).sum(1) / (
        2 * np.pi * i0(kappa)
    )
    kde /= np.trapz(kde, x=bins)
    return bins, kde


def fit_lines_to_hands_mask(
    padded_mask: np.ndarray,
    center: Point,
    use_largest_region: bool = True,
    debug: bool = False,
    min_peak_height: float = 0.01,
    min_prominence: float = 5e-4,
    min_peak_width: float = 3,
) -> List[Line]:
    if use_largest_region:
        label_image = label(padded_mask)
        # select the largest object to filter out small false positives
        regions = sorted(regionprops(label_image), key=lambda r: r.area, reverse=True)
        if not regions:
            return []
        region = regions[0]

        (
            ymin,
            xmin,
            ymax,
            xmax,
        ) = region.bbox
        region_bbox = BBox(xmin, ymin, xmax, ymax, name="")

        padded_mask = label_image == region.label
        if debug:
            region_bbox.plot()
            center.plot()
            plt.imshow(padded_mask)
            plt.show()
        if not region_bbox.contains(center):
            return []

    # TODO proper debug handling
    # TODO optionally restrict to the largest shape found in the mask
    if debug:
        plt.imshow(padded_mask)
        center.plot()
        plt.show()
    vectors = []
    points = []
    for i, row in enumerate(padded_mask):
        for j, value in enumerate(row):
            if value > 0:
                line1 = Line(center, Point(j, i))
                points.append(Point(j, i))
                if line1.length:
                    vectors.append(line1.unit_vector)
    if len(points) < 50:
        return []
    vectors = np.array(vectors)
    angles = np.rad2deg(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angles = np.where(angles < 0, angles + 180, angles)
    X = angles.copy()[:, np.newaxis]
    # TODO there's a problem with linked boundaries here
    step = 0.5
    X_plot = np.arange(0, 180, step)[:, np.newaxis]  # KDE requires 2D inputs
    # print(X_plot)
    ax = None
    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        ax = ax.ravel()
        ax[1].imshow(padded_mask)
        center.plot(ax[1])

    kernel = "gaussian"
    lw = 2
    # TODO the KDE stuff should be extracted and tested separately
    kde = KernelDensity(kernel=kernel, bandwidth=1.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    exp_dens = np.exp(log_dens)

    peaks, properties = find_peaks(
        exp_dens,
        height=min_peak_height,
        width=min_peak_width,
        distance=5,
        prominence=min_prominence,
    )
    if debug:
        print(properties)
    results_half = peak_widths(exp_dens, peaks, rel_height=0.5)[0]
    colors = distinctipy.get_colors(len(peaks), n_attempts=1)
    peak_to_points = defaultdict(list)
    for i, (peak_idx, peak_w) in enumerate(zip(peaks, results_half)):
        peak_position = X_plot[peak_idx, 0]
        left_peak_border = max(peak_position - peak_w / 2, X_plot[0, 0])
        right_peak_border = min(peak_position + peak_w / 2, X_plot[-1, 0])
        left_peak_border_idx = np.argwhere(X_plot[:, 0] <= left_peak_border)[-1][0]
        right_peak_border_idx = np.argwhere(X_plot[:, 0] >= right_peak_border)[0][0]
        points_ids_left = angles >= left_peak_border
        points_ids_right = angles <= right_peak_border
        points_ids = points_ids_left & points_ids_right
        points_ids = points_ids.squeeze()

        peak_to_points[peak_idx].extend((np.array(points)[points_ids]))

        if debug:
            # print(peak_position, peak_height, peak_w)
            ax[0].fill_between(
                X_plot[left_peak_border_idx:right_peak_border_idx, 0],
                exp_dens[left_peak_border_idx:right_peak_border_idx],
                facecolor=colors[i],
                alpha=0.5,
            )
    fitted_hands = [
        Line.from_multiple_points(points) for points in peak_to_points.values()
    ]
    hands = []
    for hand in fitted_hands:
        if hand.end.distance(center) < hand.start.distance(center):
            new_hand = Line(hand.end, hand.start)
        else:
            new_hand = hand
        hands.append(new_hand)
    hands = [
        dataclasses.replace(hand, score=width)
        for hand, width in zip(hands, results_half)
    ]

    if debug:
        ax[0].plot(
            X_plot[:, 0],
            exp_dens,
            lw=lw,
            linestyle="-",
        )
        ax[0].axhline(min_peak_height, color="red")
        Y = -0.005 - 0.01 * np.random.random(X.shape[0])
        ax[0].plot(X[:, 0], Y, "+k")

        empty_mask = np.zeros((*padded_mask.shape, 3)).astype("uint8")
        ax[1].invert_yaxis()

        for i, peak_points in enumerate(peak_to_points.values()):
            for p in peak_points:
                color = (np.array(colors[i]) * 255).astype("uint8")
                empty_mask = p.draw(empty_mask, color=color)
            hands[i].plot(ax=ax[1], color="white", lw=2)
        ax[1].imshow(empty_mask)
        plt.tight_layout()
        plt.savefig("debug_plots.jpg")
        plt.show()
    hands = remove_complementary_hands(hands, center)
    return hands


def remove_complementary_hands(hands: List[Line], center) -> List[Line]:
    angles_between_hands = np.zeros((len(hands), len(hands)))
    complementary_hands = []
    new_hands = []
    hands_and_index = list(enumerate(hands))
    for (i, hand_a), (j, hand_b) in combinations(hands_and_index, 2):
        angle_between = hand_a.angle_between(hand_b)
        angles_between_hands[i][j] = angle_between

    angles_between_hands = np.rad2deg(angles_between_hands)
    complementary_hands_on_the_same_side = angles_between_hands < 5
    complementary_hands_on_the_opposite_sides = angles_between_hands > 175
    elements_to_remove = np.nonzero(complementary_hands_on_the_opposite_sides)
    i_s, j_s = elements_to_remove
    new_hands = [
        hand for k, hand in hands_and_index if k not in np.concatenate((i_s, j_s))
    ]
    for i, j in zip(i_s, j_s):
        start = hands[i].end
        end = hands[j].end
        if start.distance(center) > end.distance(center):
            end, start = start, end
        new_hands.append(
            Line(
                start,
                end,
                score=float(mean([hands[i].score, hands[j].score])),
            )
        )

    return new_hands


def line_selector(
    all_hands_lines: List[Line], center: Point
) -> Tuple[Tuple[Line, Line], List[Line]]:
    # select which line is hour and which is a minute hand

    # nothing to do
    if not all_hands_lines:
        return tuple(), []

    sorted_lines = sorted(all_hands_lines, key=lambda l: l.length, reverse=False)
    # if there's just one line: count it as both hour and minute hand
    if len(all_hands_lines) == 1:
        selected_lines = [sorted_lines[0], sorted_lines[0]]
        rejected_lines = []

    # 2 lines are fine
    elif len(all_hands_lines) == 2:
        selected_lines = sorted_lines
        rejected_lines = []

    # if there are three lines - reject the longest
    elif len(all_hands_lines) == 3:
        selected_lines = sorted_lines[:2]
        rejected_lines = sorted_lines[2:]
    # if there are 4 lines - reject the shorted and the longest
    elif len(all_hands_lines) == 4:
        selected_lines = sorted_lines[1:3]
        rejected_lines = sorted_lines[:1] + sorted_lines[3:]
    # give up, something must have gone terribly wrong
    else:
        return tuple(), []
    if len(selected_lines) != 2:
        raise NotImplementedError(
            f"I don't know what to do with {len(sorted_lines)} recognized hand lines"
        )
    # selected_lines = [
    #     dataclasses.replace(selected_line, start=center)
    #     for selected_line in selected_lines
    # ]
    # if there are two lines: shorter is hour, longer is minutes
    minute, hour = sorted(
        selected_lines, key=lambda l: l.end.distance(center), reverse=True
    )

    return (minute, hour), rejected_lines
