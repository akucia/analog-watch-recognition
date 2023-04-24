from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from PIL.Image import BICUBIC
from skimage.transform import rotate

from watch_recognition.utilities import Line, Point


def load_single_kp_example(
    source: Path,
    image_name: str,
    image_size: Optional[Tuple[int, int]] = (224, 224),
):
    labels_df = pd.read_csv(source / "tags.csv")
    image_path = source / image_name

    img = tf.keras.preprocessing.image.load_img(
        image_path, "rgb", target_size=image_size, interpolation="bicubic"
    )
    data = labels_df[labels_df["crop_file"] == image_name]
    return img, data


def load_keypoints_data_as_kp(
    source: str,
    image_size: Tuple[int, int] = (224, 224),
    skip_examples_without_all_keypoints: bool = True,
    min_image_size: Optional[Tuple[int, int]] = (100, 100),
    autorotate: bool = False,
):
    if not source.endswith("/"):
        source += "/"
    with tf.io.gfile.GFile(source + "tags.csv", "r") as f:
        labels_df = pd.read_csv(f)
    all_keypoints = []
    all_images = []
    all_filenames = []
    for image_name, data in labels_df.groupby("crop_file"):
        if skip_examples_without_all_keypoints and len(data) < 4:
            continue
        if len(data["label"].unique()) != 4:
            print(f"{image_name} keypoints are not unique")
        image_path = source + f"{image_name}"

        with tf.io.gfile.GFile(image_path, "rb") as f:
            with Image.open(f) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                image_too_small = (
                    img.size[0] < min_image_size[0] or img.size[1] < min_image_size[1]
                )
                if min_image_size is not None and image_too_small:
                    continue

                if image_size is not None:
                    img = img.resize(image_size, BICUBIC)
                image_np = tf.keras.preprocessing.image.img_to_array(img).astype(
                    "uint8"
                )
        points = []
        for tag in ["Center", "Top", "Hour", "Minute"]:
            tag_data = data[data["label"] == tag]
            # if tag is missing, negative values will throw it outside the image
            # TODO this should probably be handled better by using the keypoint
            #  labels explicitly
            if not tag_data.empty:
                # two last coordinate values are ignored, 4 values are required to
                # correctly use albumentations for keypoints with tf.Data
                point = np.array(
                    (tag_data["x"].values[0], tag_data["y"].values[0], 0, 0)
                )
                # if tag == "Center":
                #     img_center = np.array([0.5, 0.5, 0, 0])
                #     distance_to_center = np.sqrt(((point - img_center) ** 2).sum())
                #     if distance_to_center > 0.1:
                #         print(f"{image_name} has weird center point - skipping...")
                #         break

                point[0] *= image_np.shape[0]
                point[1] *= image_np.shape[1]
                # int_point = np.round(point).astype(int)
                kp = tuple(point)
                # else:
                #     kp = (-100, -100, 0, 0)
                # raise ValueError(f"no keypoint data for {tag} on {image_name}")

                points.append(kp)
        if len(points) < 4:
            continue
        points = np.array(points)

        if autorotate:
            angle = keypoints_to_angle(points[0, :2], points[1, :2])
            image_np = rotate(image_np.astype("float32"), -angle).astype("uint8")
            # center_point = Point(*points[0, :2])
            # todo to rotate around center point I would need to translate the
            #  center of the image to exactly match the center point
            image_center_point = Point(image_np.shape[0] / 2, image_np.shape[1] / 2)
            for i in range(4):
                points[i, :2] = np.array(
                    Point(*points[i, :2])
                    .rotate_around_point(image_center_point, angle)
                    .as_coordinates_tuple
                )

        all_images.append(image_np)
        all_filenames.append(data["crop_file"].values[0])

        all_keypoints.append(points)
    all_images = np.array(all_images)
    all_keypoints = np.array(all_keypoints)
    all_filename = np.array(all_filenames)

    return all_images, all_keypoints, all_filename


def load_image(
    image_path: str,
    image_size: Optional[Tuple[int, int]] = None,
    preserve_aspect_ratio: bool = False,
):
    if image_path.startswith("gs://"):
        file = tf.io.gfile.GFile(image_path, "rb")
    else:
        file = open(image_path, "rb")
    with Image.open(file) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if image_size is not None:
            if preserve_aspect_ratio:
                img.thumbnail(size=image_size)
            else:
                img = img.resize(image_size, BICUBIC)

        image_np = tf.keras.preprocessing.image.img_to_array(img).astype("uint8")
    file.close()
    return image_np


def keypoints_to_angle(center, top):
    top = top - center
    top = top / np.linalg.norm(top)

    reference_vector = np.array([0, -0.5])
    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    angle = np.rad2deg(np.arctan2(*top) - np.arctan2(*reference_vector))
    return int(angle)


def keypoint_to_sin_cos_angle(center, kp):
    c = Point(*center)
    k = Point(*kp)
    angle = Line(c, k).angle
    sin_hour = float(np.sin(angle))
    cos_hour = float(np.cos(angle))

    return np.array([sin_hour, cos_hour])


def binarize(value, bin_size):
    if value < 0:
        value += 360
    n_bins = 360 // bin_size
    b = (value + bin_size / 2) / bin_size
    b = int(b)
    return b % n_bins
