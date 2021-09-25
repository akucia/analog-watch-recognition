from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def load_single_kp_example(
    source: Path,
    image_name: str,
    image_size: Tuple[int, int] = (224, 224),
):
    labels_df = pd.read_csv(source / f"tags.csv")
    image_path = source / image_name

    img = tf.keras.preprocessing.image.load_img(
        image_path, "rgb", target_size=image_size, interpolation="bicubic"
    )
    data = labels_df[labels_df["crop_file"] == image_name]
    return img, data


def load_keypoints_data_as_kp(
    source: Path,
    image_size: Tuple[int, int] = (224, 224),
    skip_examples_without_all_keypoints: bool = True,
):
    labels_df = pd.read_csv(source / f"tags.csv")
    all_keypoints = []
    all_images = []
    all_filenames = []
    for image_name, data in labels_df.groupby("crop_file"):
        if skip_examples_without_all_keypoints and len(data) != 4:
            continue
        if len(data["label"].unique()) != 4:
            # print(data)
            print(f"{image_name} keypoints are not unique")
        image_path = source / image_name
        img = tf.keras.preprocessing.image.load_img(
            image_path, "rgb", target_size=image_size, interpolation="bicubic"
        )

        image_np = tf.keras.preprocessing.image.img_to_array(img).astype("uint8")
        all_images.append(image_np)

        points = []
        for tag in ["Center", "Top", "Hour", "Minute"]:
            tag_data = data[data["label"] == tag]
            # if tag is missing, negative values will throw it outside the image
            # TODO this should probably be handled better by using the keypoint
            #  labels explicitly
            kp = (-1, -1, -1, -1)
            # two last coordinate values are ignored, 4 values are required to
            # correctly use albumentations for keypoints with tf.Data
            if not tag_data.empty:
                point = np.array(
                    (tag_data["x"].values[0], tag_data["y"].values[0], 0, 0)
                )
                point[0] *= image_size[0]
                point[1] *= image_size[1]
                int_point = np.floor(point).astype(int)
                kp = tuple(int_point)

            points.append(kp)
        all_filenames.append(data["crop_file"].values[0])
        points = np.array(points)
        all_keypoints.append(points)
    return np.array(all_images), np.array(all_keypoints), np.array(all_filenames)
