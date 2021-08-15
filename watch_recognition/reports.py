import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.core.display import display


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
