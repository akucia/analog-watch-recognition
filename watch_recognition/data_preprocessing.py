from pathlib import Path

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
    image_size=(224, 224),
    mask_size=(14, 14),
    extent=(2, 2),
    split="train",
    gaussian_target: bool = False,
):
    print("deprecated, use load_keypoints_data_2")
    downsample_factor = image_size[0] / mask_size[0]
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

        image_np = tf.keras.preprocessing.image.img_to_array(img).astype("uint8")
        mean = np.zeros_like(image_np)
        mean[:, :, :] = 128
        image_np = np.where(image_np < 1, mean, image_np)
        all_images.append(image_np)
        points = []
        for tag in ["Center", "Top", "Hour", "Minute"]:
            tag_data = data[data["tag_name"] == tag]

            matched = np.zeros(mask_size)
            if not tag_data.empty:
                point = np.array((tag_data["x"].values[0], tag_data["y"].values[0]))
                point[0] *= image_size[0]
                point[1] *= image_size[1]
                fm_point = point / downsample_factor
                int_point = np.floor(fm_point).astype(int)
                if gaussian_target:
                    rr, cc = rectangle(
                        tuple(int_point - 1), extent=extent, shape=matched.shape
                    )
                    matched[cc, rr] = 1
                    matched = gaussian(matched, sigma=1)
                    matched /= matched.max()
                else:

                    rr, cc = rectangle(
                        tuple(int_point - 1), extent=extent, shape=matched.shape
                    )
                    matched[cc, rr] = 1
            points.append(matched)
        masks = np.array(points).transpose((1, 2, 0))

        all_masks.append(masks)
    return np.array(all_images), np.array(all_masks), np.array(all_labels)


def load_keypoints_data_2(
    source: Path,
    image_size=(224, 224),
    mask_size=(14, 14),
    extent=(2, 2),
    gaussian_target: bool = False,
):
    downsample_factor = image_size[0] / mask_size[0]
    labels_df = pd.read_csv(source / f"tags.csv")
    all_masks = []
    all_images = []
    for image_name, data in labels_df.groupby("crop_file"):
        image_path = source / image_name
        img = tf.keras.preprocessing.image.load_img(
            image_path, "rgb", target_size=image_size, interpolation="bicubic"
        )

        image_np = tf.keras.preprocessing.image.img_to_array(img)
        all_images.append(image_np)
        points = []
        for tag in ["Center", "Top", "Hour", "Minute"]:
            tag_data = data[data["label"] == tag]
            matched = np.zeros(mask_size)

            if not tag_data.empty:
                point = np.array((tag_data["x"].values[0], tag_data["y"].values[0]))
                point[0] *= image_size[0]
                point[1] *= image_size[1]
                fm_point = point / downsample_factor
                int_point = np.floor(fm_point).astype(int)
                if gaussian_target:
                    rr, cc = rectangle(
                        tuple(int_point - 1), extent=extent, shape=matched.shape
                    )
                    matched[cc, rr] = 1
                    matched = gaussian(matched, sigma=1)
                    matched /= matched.max()
                else:

                    rr, cc = rectangle(
                        tuple(int_point - 1), extent=extent, shape=matched.shape
                    )
                    matched[cc, rr] = 1
            points.append(matched)
        masks = np.array(points).transpose((1, 2, 0))

        all_masks.append(masks)
    return np.array(all_images), np.array(all_masks)


def load_keypoints_data_as_kp(
    source: Path,
    image_size=(224, 224),
):
    labels_df = pd.read_csv(source / f"tags.csv")
    all_keypoints = []
    all_images = []
    for image_name, data in labels_df.groupby("crop_file"):
        image_path = source / image_name
        img = tf.keras.preprocessing.image.load_img(
            image_path, "rgb", target_size=image_size, interpolation="bicubic"
        )

        image_np = tf.keras.preprocessing.image.img_to_array(img).astype("uint8")
        all_images.append(image_np)
        points = []
        for tag in ["Center", "Top", "Hour", "Minute"]:
            tag_data = data[data["label"] == tag]
            kp = (0, 0, 0, 0)  # two last values are ignored

            if not tag_data.empty:
                point = np.array(
                    (tag_data["x"].values[0], tag_data["y"].values[0], 0, 0)
                )
                point[0] *= image_size[0]
                point[1] *= image_size[1]
                int_point = np.floor(point).astype(int)
                kp = tuple(int_point)

            points.append(kp)
        points = np.array(points)

        all_keypoints.append(points)
    return np.array(all_images), np.array(all_keypoints)
