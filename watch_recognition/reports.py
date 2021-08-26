import io
import itertools
from datetime import datetime, timedelta
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.core.display import display
from matplotlib import patches

from watch_recognition.models import decode_batch_predictions


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


def generate_report(X, y, model, kind="regression"):
    train_image = X
    pred = model.predict(train_image)
    pred["minute"] = np.zeros((len(pred["hour"]), 1))
    y["minute"] = np.zeros((len(y["hour"]), 1))
    records = []
    for i, (h_pred, m_pred, h_true, m_true) in enumerate(
        zip(pred["hour"], pred["minute"], y["hour"], y["minute"])
    ):
        # print(h_pred, h_true)
        if kind == "regression":
            pred_hour = np.round((h_pred * 20) + 6)[0]
            pred_minutes = 0  # np.round((m_pred + 0.5) * 60)[0]
            expected_hour = np.round((h_true * 20) + 6)
            expected_minutes = 0
        elif kind == "classification":
            pred_hour = np.argmax(h_pred)
            pred_minutes = 0  # np.round((m_pred + 0.5) * 60)[0]
            expected_hour = np.argmax(h_true)
            expected_minutes = 0
        else:
            raise ValueError(f"unknown kind: {kind}")
        record = {
            "pred_hour": pred_hour,
            "pred_minutes": pred_minutes,
            "expected_hour": expected_hour,
            "expected_minutes": expected_minutes,
        }
        records.append(record)
    df = pd.DataFrame(records)  # .astype(int)
    pred_time = df["pred_hour"] * 60 + df["pred_minutes"]
    expected_time = df["expected_hour"] * 60 + df["expected_minutes"]
    df["total_minutes_diff"] = (pred_time - expected_time).abs().values
    sorted_df = df.sort_values("total_minutes_diff", ascending=False)
    sorted_df = sorted_df[sorted_df["total_minutes_diff"] > 0]
    print(sorted_df.head(5)[["pred_hour", "expected_hour"]])
    print("-" * 100)
    print(f"total_minutes_diff: {df['total_minutes_diff'].sum()}")
    print(f"avg_minutes_diff: {df['total_minutes_diff'].mean()}")
    print("-" * 100)
    print("10 worst cases: ")

    for i, row in sorted_df.head(10).iterrows():

        print(
            f"{i}. pred: {row.pred_hour}:{row.pred_minutes}| target: {row.expected_hour}:{row.expected_minutes}"
        )
        display(tf.keras.preprocessing.image.array_to_img((X[i])))
        print("-" * 50)

    return sorted_df


def plot_confusion_matrix(cm, class_names):
    """
    https://www.tensorflow.org/tensorboard/image_summaries
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


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


# def image_grid():
#     """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
#     # Create a figure to contain the plot.
#     figure = plt.figure(figsize=(10, 10))
#     for i in range(25):
#         # Start next subplot.
#         plt.subplot(5, 5, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(train_images[i], cmap=plt.cm.binary)
#
#     return figure


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


def run_on_image_debug(model, image, targets=None, show_grid=True):
    predicted = model(np.expand_dims(image, 0)).numpy()[0]
    downsample_factor = image.shape[1] / predicted.shape[1]
    tags = ["Center", "Top", "Hour", "Minute"]
    for i, (point, tag) in enumerate(zip(predicted.transpose((2, 1, 0)), tags)):
        print(tag)
        extent = [0, predicted.shape[1], predicted.shape[1], 0]
        if targets is not None:
            fig, ax = plt.subplots(1, 3)
            ax[2].title.set_text(f"Target|{tag}")
            ax[2].imshow(targets[:, :, i], extent=extent)
        else:
            fig, ax = plt.subplots(1, 2)
        point = point.T
        ax[0].imshow(point, extent=extent)
        ax[0].title.set_text(f"Output|{tag}")

        ax[1].imshow(
            image.astype("uint8"), extent=[0, image.shape[0], image.shape[1], 0]
        )
        ax[1].title.set_text(f"Image|{tag}")
        ax[1].axis("off")
        if show_grid:
            for j in range(predicted.shape[0]):
                ax[1].axvline(j * downsample_factor)
            for j in range(predicted.shape[1]):
                ax[1].axhline(j * downsample_factor)
        grid_predicted = np.unravel_index(
            np.argmax(predicted[:, :, i]), predicted[:, :, i].shape
        )

        rectangle_predicted = (
            grid_predicted[1] * downsample_factor,
            grid_predicted[0] * downsample_factor,
        )

        rect_pred = patches.Rectangle(
            rectangle_predicted,
            downsample_factor,
            downsample_factor,
            linewidth=1,
            edgecolor="r",
            facecolor="red",
        )

        ax[1].add_patch(rect_pred)
        plt.show()
    read_time = decode_batch_predictions(np.expand_dims(predicted, 0))[0]
    print(f"{read_time[0]}:{read_time[1]}")


def generate_report_for_keypoints(model, X, y, show_top_n_errors=0):
    y_pred = model.predict(X)
    y_pred_decoded = decode_batch_predictions(y_pred)
    y_target_decoded = decode_batch_predictions(y)

    h_diff = np.abs(y_pred_decoded[:, 0] - y_target_decoded[:, 0])
    m_diff = np.abs(y_pred_decoded[:, 1] - y_target_decoded[:, 1])
    df = pd.DataFrame(
        {
            "h_pred": y_pred_decoded[:, 0],
            "m_pred": y_pred_decoded[:, 1],
            "h_target": y_target_decoded[:, 0],
            "m_target": y_target_decoded[:, 1],
            "h_diff": h_diff,
            "m_diff": m_diff,
            "total_diff": h_diff * 60 + m_diff,
        }
    )
    df["total_diff"] = time_diff_np(
        df[["h_pred", "m_pred"]].values,
        df[["h_target", "m_target"]].values,
    )
    df = df.sort_values("total_diff", ascending=False)
    if show_top_n_errors > 0:
        df_head = df.head(show_top_n_errors)
    else:
        df_head = df.tail((-1) * show_top_n_errors)
    print(f"average of lost minutes: {df.total_diff.mean()}")
    for i, row in df_head.iterrows():
        print(i)
        print("-" * 50)
        print(
            f"target {row.h_target:02}:{row.m_target:02} | predicted {row.h_pred:02}:{row.m_pred:02} | diff: {row.total_diff:02.2f}"
        )
        plt.figure(figsize=(1, 1))
        plt.imshow(X[i].astype("uint8"))
        plt.axis("off")
        plt.show()
        run_on_image_debug(model, X[i], show_grid=False)
    return df


def log_distances(epoch, logs, X, y, file_writer, model):
    # Use the model to predict the values from the validation dataset.
    predicted = model.predict(X)
    output_2d_shape = predicted.shape[1:3]
    mean_distances = []
    for i, tag in enumerate(["Center", "Top", "Hour", "Minute"]):
        distances = []
        for row in range(predicted.shape[0]):
            # TODO take argmax for all outputs at once instead of for loop
            pred = np.array(
                np.unravel_index(np.argmax(predicted[row, :, :, i]), output_2d_shape)
            )[::-1]
            gt = np.array(
                np.unravel_index(np.argmax(y[row, :, :, i]), output_2d_shape)
            )[::-1]
            distances.append(
                np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
                / predicted.shape[1]
            )
        mean_distance = np.mean(distances)
        mean_distances.append(mean_distance)
        with file_writer.as_default():
            tf.summary.scalar(
                f"maximum_point_distance_{tag}", mean_distance, step=epoch
            )
    with file_writer.as_default():
        tf.summary.scalar(
            f"maximum_point_distance_mean", np.mean(mean_distances), step=epoch
        )
