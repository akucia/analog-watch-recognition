from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def unison_shuffled_copies(a, b, seed=42):
    np.random.seed(seed)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b.iloc[p]


def load_data(path: Path, IMAGE_SIZE):
    data = pd.read_csv(path)
    # n_samples = 32*5
    n_samples = len(data)
    print(n_samples)
    y = data.head(n_samples)

    images = [
        tf.keras.preprocessing.image.load_img(
            path.parent / filename,
            "rgb",
            target_size=IMAGE_SIZE,
            interpolation="bicubic",
        )
        for filename in y["filename"].values
    ]
    X = [tf.keras.preprocessing.image.img_to_array(img) for img in images]
    X = np.array(X)
    y = y[["hour", "minute"]]

    return X, y


def preprocess_targets(y, kind="regression"):
    y.loc[y.hour == 12, "hour"] = 0

    if kind == "regression":
        y_hours = (y["hour"].values.astype(np.float32) - 6) / 20
        y_minutes = y["minute"].values.astype(np.float32)

        # y = {"hour": y_hours, "minute": y_minutes}
        y = {"hour": y_hours}
    elif kind == "classification":
        y_hours = tf.keras.utils.to_categorical(
            y["hour"].values.astype("int32"), num_classes=12
        )
        y_minutes = tf.keras.utils.to_categorical(
            y["minute"].values.astype("int32"), num_classes=60
        )

        # y = {"hour": y_hours, "minute": y_minutes}
        y = {"hour": y_hours}
    else:
        raise ValueError(f"unknown kind: {kind}")
    return y


def load_synthethic_data(path, image_size, n_samples=200):
    data = pd.read_csv(path)
    images = [
        tf.keras.preprocessing.image.load_img(
            path.parent / f"images/{i}.jpg",
            "rgb",
            target_size=image_size,
            interpolation="bicubic",
        )
        for i in range(n_samples)
    ]
    X = [tf.keras.preprocessing.image.img_to_array(img) for img in images]
    X_synth = np.array(X)
    y = data[["hour", "minute"]][:n_samples]
    return X_synth, y
