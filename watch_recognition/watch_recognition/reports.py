import io
from datetime import timedelta
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.core.display import display

from watch_recognition.models import convert_outputs_to_keypoints, points_to_time


def predict_on_image(path, model, image_size=(100, 100)):
    test_image = tf.keras.preprocessing.image.load_img(
        path, "rgb", target_size=image_size, interpolation="bicubic"
    )
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.array([test_image])
    pred = model.predict(test_image)
    pred["minute"] = np.zeros_like(pred["hour"])
    for i, (h_pred, m_pred) in enumerate(zip(pred["hour"], pred["minute"])):
        pred_hour = np.round((h_pred * 20) + 6)[0]
        pred_minutes = 0  # np.round((m_pred + 0.5) * 60)[0]
        print(pred_hour, ":", pred_minutes)
    display(tf.keras.preprocessing.image.array_to_img(test_image[0]))


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
    # TODO Cleanup and refactor this
    predicted = model(np.expand_dims(image, 0)).numpy()[0]

    center, hour, minute, top = convert_outputs_to_keypoints(predicted)
    center, hour, minute, top = [
        np.array(p.as_coordinates_tuple).astype(float)
        for p in (center, hour, minute, top)
    ]
    downsample_factor = image.shape[1] / predicted.shape[1]
    masks = predicted.transpose((2, 1, 0))
    extent = [0, predicted.shape[1], predicted.shape[1], 0]
    fig, ax = plt.subplots(3, 2, figsize=(5, 10))
    # Center
    ax[0][0].imshow(masks[0].T, extent=extent, cmap="gray", vmin=0, vmax=1)
    ax[0][0].title.set_text(f"Output|Center")
    ax[0][1].imshow(
        image.astype("uint8"), extent=[0, image.shape[0], image.shape[1], 0]
    )
    ax[0][1].title.set_text(f"Image|Center")
    ax[0][1].axis("off")
    ax[0][0].scatter(*center, marker="x", color="red", vmin=0, vmax=1)
    ax[0][1].scatter(*(center * downsample_factor), marker="x", color="red")

    # Top

    ax[1][0].imshow(masks[1].T, extent=extent, cmap="gray", vmin=0, vmax=1)
    ax[1][0].title.set_text(f"Output|Top")
    ax[1][1].imshow(
        image.astype("uint8"), extent=[0, image.shape[0], image.shape[1], 0]
    )
    ax[1][1].title.set_text(f"Image|Top")
    ax[1][1].axis("off")
    ax[1][0].scatter(*top, marker="x", color="red")
    ax[1][1].scatter(*(top * downsample_factor), marker="x", color="red")

    # Hands

    ax[2][0].imshow(masks[2].T, extent=extent, cmap="gray", vmin=0, vmax=1)
    ax[2][0].title.set_text(f"Output|Hands")
    ax[2][1].imshow(
        image.astype("uint8"), extent=[0, image.shape[0], image.shape[1], 0]
    )
    ax[2][1].title.set_text(f"Image|Hands")
    ax[2][1].axis("off")
    ax[2][0].scatter(*minute, marker="x", color="red")
    ax[2][1].scatter(*(minute * downsample_factor), marker="x", color="red")
    ax[2][0].scatter(*minute, marker="x", color="red")
    ax[2][1].scatter(*(minute * downsample_factor), marker="x", color="red")

    ax[2][0].scatter(*hour, marker="x", color="red")
    ax[2][1].scatter(*(hour * downsample_factor), marker="x", color="red")
    ax[2][0].scatter(*hour, marker="x", color="red")
    ax[2][1].scatter(*(hour * downsample_factor), marker="x", color="red")
    plt.show()
    read_hour, read_minute = points_to_time(center, hour, minute, top)
    print(f"{read_hour:.0f}:{read_minute:.0f}")


def visualize_high_loss_examples(epoch, logs, dataset, file_writer, model):
    iterator = dataset.as_numpy_iterator()
    batch = next(iterator)
    X_batch, y_batch = batch
    predictions = model.predict(X_batch)
    loss = tf.keras.metrics.categorical_crossentropy(y_batch, predictions).numpy()
    loss_for_example = loss.sum(axis=-1).sum(axis=-1)
    worst_examples = np.argsort(loss_for_example)[::-1][:5]

    fig, axarr = plt.subplots(1, 5)
    for i, idx in enumerate(worst_examples):
        axarr[i].set_xticks([])
        axarr[i].set_yticks([])
        axarr[i].grid(False)
        axarr[i].imshow(X_batch[idx])
        axarr[i].set_title(str(loss_for_example[idx]))
    plt.tight_layout()
    img = plot_to_image(fig)

    with file_writer.as_default():
        tf.summary.image(
            "top_5_high_loss_examples",
            img,
            step=epoch,
        )


def euclidean_distance(x_1, y_1, x_2, y_2) -> float:
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


def log_scalar_metrics(epoch, logs, X, y, file_writer, model):
    predicted = model.predict(X)
    (
        center_distances,
        hour_distances,
        minute_distances,
        top_distances,
    ) = calculate_distances_between_points(X, predicted, y)
    distances = [
        center_distances,
        top_distances,
        hour_distances,
        minute_distances,
    ]
    with file_writer.as_default():
        means = []
        for tag, array in zip(["Center", "Top", "Hour", "Minute"], distances):

            mean = np.mean(array)
            means.append(mean)
            tf.summary.scalar(f"point_distance_{tag}", mean, step=epoch)
    with file_writer.as_default():
        tf.summary.scalar(f"point_distance_mean", np.mean(means), step=epoch)


def calculate_distances_between_points(X, predicted, y):
    center_distances = np.zeros(predicted.shape[0])
    top_distances = np.zeros(predicted.shape[0])
    minute_distances = np.zeros(predicted.shape[0])
    hour_distances = np.zeros(predicted.shape[0])
    scale_factor = X.shape[1] / predicted.shape[1]
    for row in range(predicted.shape[0]):
        center_hat, hour_hat, minute_hat, top_hat = convert_outputs_to_keypoints(
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
    return center_distances, hour_distances, minute_distances, top_distances
