import io
from datetime import timedelta
from itertools import combinations
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from watch_recognition.models import IouLoss2, points_to_time
from watch_recognition.targets_encoding import convert_mask_outputs_to_keypoints
from watch_recognition.utilities import Line


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def time_diff(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """

    Args:
        a
        b

    Returns:

    """

    if a[0] > 12 or b[0] > 12:
        raise ValueError(
            "only 12-hour clock times are supported, "
            "hour value cannot be larger than 12"
        )

    if a[0] < 1 or b[0] < 1:
        raise ValueError(
            "only 12-hour clock times are supported, "
            "hour value cannot be smaller than 1"
        )
    if a[0] == 12:
        a = (0, a[1])

    if b[0] == 12:
        b = (0, b[1])

    a = timedelta(hours=a[0], minutes=a[1])
    b = timedelta(hours=b[0], minutes=b[1])
    delta = abs(a - b)
    hours_delta = delta.seconds // 3600
    if hours_delta > 6:
        delta = timedelta(hours=12, minutes=0) - delta
    return int(delta.seconds / 60)


def time_diff_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """

    Args:
        a
        b

    Returns:

    """
    if a.shape != b.shape:
        raise ValueError("shape of a != shape of b")
    if a.shape[1] != 2:
        raise ValueError("expected argument a shape[1] to be 2")

    if b.shape[1] != 2:
        raise ValueError("expected argument b shape[1] to be 2")

    if (a[:, 0] > 12).any() or (b[:, 0] > 12).any():
        raise ValueError(
            "only 12-hour clock times are supported, "
            "hour value cannot be larger than 12"
        )

    if (a[:, 0] < 1).any() or (b[:, 0] < 1).any():
        raise ValueError(
            "only 12-hour clock times are supported, "
            "hour value cannot be smaller than 1"
        )

    a[:, 0] = np.where(a[:, 0] == 12, np.zeros_like(a[:, 0]), a[:, 0])
    b[:, 0] = np.where(b[:, 0] == 12, np.zeros_like(b[:, 0]), b[:, 0])
    deltas = []
    for a_i, b_i in zip(a, b):
        a = timedelta(hours=int(a_i[0]), minutes=int(a_i[1]))
        b = timedelta(hours=int(b_i[0]), minutes=int(b_i[1]))
        delta = abs(a - b)
        hours_delta = delta.seconds // 3600
        if hours_delta > 6:
            delta = timedelta(hours=12, minutes=0) - delta
        deltas.append(delta.seconds / 60)
    deltas = np.array(deltas).reshape(-1, 1)
    return deltas.astype(int)


def run_on_image_debug(model, image):
    predicted = model.predict(np.expand_dims(image, 0))[0]

    draw_predictions_debug(image, predicted)


def draw_predictions_debug(image, predicted):
    # TODO Cleanup and refactor this

    keypoints = convert_mask_outputs_to_keypoints(
        predicted, decode_hands_from_lines=True
    )
    for k in keypoints:
        print(k)

    keypoints = [np.array(p.as_coordinates_tuple).astype(float) for p in keypoints]
    center = keypoints[0]
    top = keypoints[1]
    hands = keypoints[2:]
    downsample_factor = image.shape[1] / predicted.shape[1]
    masks = predicted.transpose((2, 1, 0))
    extent_mask = [0, predicted.shape[1], predicted.shape[1], 0]
    extent_image = [0, image.shape[0], image.shape[1], 0]
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    # Center
    ax[0].imshow(masks.T, extent=extent_mask)
    ax[1].imshow(image.astype("uint8"), extent=extent_image)
    ax[0].scatter(*center, marker="x", color="white", vmin=0, vmax=1, s=20)
    ax[1].scatter(*(center * downsample_factor), marker="x", color="red")
    # Top
    ax[0].scatter(*top, marker="x", color="white", s=20)
    ax[1].scatter(*(top * downsample_factor), marker="x", color="red")

    # Hands
    for hand in hands:
        ax[0].scatter(*hand, marker="x", color="white", s=20)
        ax[1].scatter(*(hand * downsample_factor), marker="x", color="red")
        ax[0].scatter(*hand, marker="x", color="white", s=20)
        ax[1].scatter(*(hand * downsample_factor), marker="x", color="red")

    colored_mask = (masks.T * 255).astype("uint8")
    # print(image.dtype)
    # print(colored_mask.dtype)
    overlay = cv2.addWeighted(image.astype("uint8"), 0.35, colored_mask, 0.65, 0)
    ax[2].imshow(overlay, extent=extent_mask)
    plt.show()
    if len(hands) == 2:
        read_hour, read_minute = points_to_time(center, hands[0], hands[1], top)
        print(f"{read_hour:.0f}:{read_minute:.0f}")


def visualize_high_loss_examples(
    epoch, logs, dataset, file_writer, model, loss, every_n_epoch=1
):
    if epoch % every_n_epoch == 0:
        img = _visualize_high_loss_examples(dataset, loss, model)
        with file_writer.as_default():
            tf.summary.image(
                "top_5_high_loss_examples",
                img,
                step=epoch,
            )


def _visualize_high_loss_examples(dataset, loss, model):
    images, losses, predictions, targets, worst_examples = get_n_worst_loss_examples(
        dataset, loss, model
    )
    fig, axarr = plt.subplots(len(worst_examples), 3, figsize=(10, 10))
    for i, idx in enumerate(worst_examples):
        for j in range(3):
            axarr[i][j].set_xticks([])
            axarr[i][j].set_yticks([])
            axarr[i][j].grid(False)
        axarr[i][0].imshow(images[idx])
        axarr[i][1].imshow(targets[idx])
        axarr[i][2].imshow(predictions[idx])
        axarr[i][2].set_title(f"{losses[idx]:.3f}")
    plt.tight_layout()
    img = plot_to_image(fig)
    return img


def get_n_worst_loss_examples(dataset, loss, model, n=5):
    iterator = dataset.as_numpy_iterator()
    images = []
    losses = []
    predictions = []
    targets = []
    for batch in iterator:
        # there might be 3rd element in batch, which is a sample weight
        X_batch, y_batch = batch[0], batch[1]
        y_pred = model.predict(X_batch)
        for target, pred in zip(y_batch, y_pred):
            value = loss(target, pred).numpy()
            losses.append(value)
        images.append(X_batch)
        predictions.append(y_pred)
        targets.append(y_batch)
    images = np.concatenate(images, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    losses = np.array(losses)
    worst_examples = np.argsort(losses)[::-1][:n]
    return images, losses, predictions, targets, worst_examples


def euclidean_distance(x_1, y_1, x_2, y_2) -> float:
    return np.sqrt(((x_1 - x_2) ** 2) + ((y_1 - y_2) ** 2))


def log_scalar_metrics(epoch, logs, X, y, file_writer, model, every_n_epoch: int = 1):
    if epoch % every_n_epoch == 0:
        predicted = model.predict(X)
        (
            center_distances,
            top_distances,
            hour_distances,
            minute_distances,
        ) = calculate_distances_between_points(X, predicted, y)
        distances = [center_distances, top_distances, hour_distances, minute_distances]
        with file_writer.as_default():
            means = []
            for tag, array in zip(["Center", "Top", "Hour", "Minute"], distances):

                mean = np.mean(array)
                means.append(mean)
                tf.summary.scalar(f"point_distance_{tag}", mean, step=epoch)
        with file_writer.as_default():
            tf.summary.scalar(f"point_distance_mean", np.mean(means), step=epoch)


def calculate_time_lost(X, predicted, y):
    predicted_hours = np.zeros(predicted.shape[0])
    predicted_minutes = np.zeros(predicted.shape[0])
    target_hours = np.zeros(predicted.shape[0])
    target_minutes = np.zeros(predicted.shape[0])

    for row in range(predicted.shape[0]):
        center_hat, top_hat, hour_hat, minute_hat = convert_mask_outputs_to_keypoints(
            predicted[row]
        )
        pred_hour, pred_minute = points_to_time(
            np.array(center_hat.as_coordinates_tuple),
            np.array(hour_hat.as_coordinates_tuple),
            np.array(minute_hat.as_coordinates_tuple),
            np.array(top_hat.as_coordinates_tuple),
        )
        predicted_hours[row] = pred_hour
        predicted_minutes[row] = pred_minute

        center = y[row, 0, :2]
        top = y[row, 1, :2]
        hour = y[row, 2, :2]
        minute = y[row, 3, :2]

        target_hour, target_minute = points_to_time(center, hour, minute, top)
        target_hours[row] = target_hour
        target_minutes[row] = target_minute
    pred = np.vstack((predicted_hours.reshape(-1, 1), predicted_minutes.reshape(-1, 1)))
    target = np.vstack((target_hours.reshape(-1, 1), target_minutes.reshape(-1, 1)))
    time_diffs = time_diff_np(pred, target)
    return time_diffs


def calculate_distances_between_points(X, predicted, y):
    center_distances = np.zeros(predicted.shape[0])
    top_distances = np.zeros(predicted.shape[0])
    minute_distances = np.zeros(predicted.shape[0])
    hour_distances = np.zeros(predicted.shape[0])
    scale_factor = X.shape[1] / predicted.shape[1]
    for row in range(predicted.shape[0]):

        center_hat, top_hat, hour_hat, minute_hat = convert_mask_outputs_to_keypoints(
            predicted[row]
        )
        center_hat = center_hat.scale(scale_factor, scale_factor).as_coordinates_tuple
        top_hat = top_hat.scale(scale_factor, scale_factor).as_coordinates_tuple
        hour_hat = hour_hat.scale(scale_factor, scale_factor).as_coordinates_tuple
        minute_hat = minute_hat.scale(scale_factor, scale_factor).as_coordinates_tuple

        center = y[row, 0, :2]
        top = y[row, 1, :2]
        hour = y[row, 2, :2]
        minute = y[row, 3, :2]

        center = np.where(center < 0, np.zeros_like(center), center)
        top = np.where(top < 0, np.zeros_like(top), top)
        hour = np.where(hour < 0, np.zeros_like(hour), hour)
        minute = np.where(minute < 0, np.zeros_like(minute), minute)

        center_distance = euclidean_distance(*center_hat, *center)
        top_distance = euclidean_distance(*top_hat, *top)
        hour_distance = euclidean_distance(*hour_hat, *hour)
        minute_distance = euclidean_distance(*minute_hat, *minute)
        center_distances[row] = center_distance
        top_distances[row] = top_distance
        hour_distances[row] = hour_distance
        minute_distances[row] = minute_distance
    # TODO return as dict to make sure user of this function doesn't make and error
    # on the metrics names
    center_distances, top_distances, hour_distances, minute_distances = [
        distance / X.shape[1]
        for distance in [
            center_distances,
            top_distances,
            hour_distances,
            minute_distances,
        ]
    ]
    return center_distances, top_distances, hour_distances, minute_distances
