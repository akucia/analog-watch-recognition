from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.draw import rectangle
from skimage.filters import gaussian


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


def load_keypoints_data(
    source: Path,
    model_output_shape: Tuple[int, int],
    image_size=(224, 224),
    split="train",
    gaussian_target: bool = False,
):
    downsample_factor = image_size[0] / model_output_shape[0]
    df = pd.read_csv(source)
    labels_df = pd.read_csv(source.parent / f"labels_{split}.csv")
    data = df[df["split"] == split]
    all_masks = []
    all_images = []
    all_labels = []
    for image_name, data in data.groupby("image_name"):
        image_path = source.parent / split / image_name
        img = tf.keras.preprocessing.image.load_img(
            image_path, "rgb", target_size=image_size, interpolation="bicubic"
        )
        label = labels_df[labels_df["filename"] == image_name].iloc[0]
        time = (label["hour"], label["minute"])
        all_labels.append(time)

        image_np = tf.keras.preprocessing.image.img_to_array(img)
        all_images.append(image_np)
        points = []
        for tag in ["Center", "Top", "Hour", "Minute"]:
            tag_data = data[data["tag_name"] == tag]
            # print(tag_data)
            matched = np.zeros((28, 28))
            # matched += 1e-6
            if not tag_data.empty:
                point = np.array((tag_data["x"].values[0], tag_data["y"].values[0]))
                # print(point)
                point[0] *= image_size[0]
                point[1] *= image_size[1]
                fm_point = point / downsample_factor
                int_point = np.floor(fm_point).astype(int)
                if gaussian_target:
                    extent = (1, 1)
                    rr, cc = rectangle(
                        tuple(int_point), extent=extent, shape=matched.shape
                    )
                    matched[rr, cc] = 1
                    matched = gaussian(matched, sigma=0.75)
                    matched /= matched.max()
                else:

                    matched[int_point[1], int_point[0]] = 1
            points.append(matched)
        masks = np.array(points).transpose((1, 2, 0))

        all_masks.append(masks)
    return np.array(all_images), np.array(all_masks), np.array(all_labels)
