from functools import partial
from pathlib import Path
from typing import Tuple

import click
import keras_cv
import numpy as np
import tensorflow as tf
import yaml
from dvclive.keras import DvcLiveCallback
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from watch_recognition.label_studio_adapters import (
    load_label_studio_kp_detection_dataset,
)
from watch_recognition.models import get_regression_model
from watch_recognition.serving import save_tf_serving_warmup_request
from watch_recognition.targets_encoding import _encode_point_to_mask
from watch_recognition.utilities import Point
from watch_recognition.visualization import visualize_keypoints, visualize_kp_dataset


def _prepare_inputs_and_targets(_encode_keypoints, dataset_train):
    X = []
    y = []
    for img, kps in dataset_train:
        X.append(img)
        y.append(_encode_keypoints(kps))
    X = np.array(X)
    y = np.array(y)
    return X, y


def encode_kps_to_mask(
    kps: np.ndarray,
    n_labels: int,
    mask_size: Tuple[int, int],
    disk_radius: int = 5,
) -> np.ndarray:
    mask = np.zeros((*mask_size, n_labels))
    for kp in kps:
        x_y = np.floor(kp[:2])
        cls = int(kp[2])
        mask[:, :, cls] = _encode_point_to_mask(
            radius=disk_radius,
            int_point=x_y,
            mask_size=mask_size,
        )
    return mask


def encode_kps(
    kps: np.ndarray,
    n_labels: int,
    crop_size: Tuple[int, int],
) -> np.ndarray:
    targets = np.zeros((n_labels, 2), dtype=np.float32) - 1
    for kp in kps:
        targets[int(kp[2])] = kp[:2]
    targets[:, 0] /= crop_size[0]
    targets[:, 1] /= crop_size[1]
    return targets.reshape(1, n_labels * 2)


def unpackage_dict(inputs):
    return inputs["images"], inputs["keypoints"]


@click.command()
@click.option("--epochs", default=1, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--image-size", default=96, type=int)
@click.option("--seed", default=None, type=int)
@click.option("--confidence-threshold", default=0.5, type=float)
@click.option("--verbosity", default=1, type=int)
@click.option("--augment", is_flag=True)
@click.option(
    "--fine-tune-from-checkpoint",
    is_flag=True,
    help="Use previous model's weight to initialize the model. "
    "If not set ImageNet weights are used instead.",
)
def main(
    epochs: int,
    batch_size: int,
    image_size: int,
    seed: int,
    confidence_threshold: float,
    verbosity: int,
    fine_tune_from_checkpoint: bool,
    augment: bool,
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    max_images = params["max_images"]
    label_to_cls = params["keypoint"]["label_to_cls"]

    dataset_path = Path("datasets/watch-faces-local.json")

    debug_data_path = Path("debug/keypoint")
    debug_data_path.mkdir(exist_ok=True, parents=True)

    cls_to_label = {v: k for k, v in label_to_cls.items()}
    num_classes = len(label_to_cls)
    image_size = (image_size, image_size)
    # TODO augment crops
    # TODO idea for 2nd improvement augment bboxes before cropping to introduce jitter
    bbox_labels = params["keypoint"]["bbox_labels"]
    checkpoint_path = Path("checkpoints/keypoint/checkpoint")
    model_save_path = Path("models/keypoint/")
    crop_size = image_size
    augmentation_layers = [
        keras_cv.layers.ChannelShuffle(groups=3),
        keras_cv.layers.RandomHue(0.5, [0.0, 255.0]),
        keras_cv.layers.AutoContrast(value_range=[0.0, 255.0]),
        keras_cv.layers.GridMask(),
        keras_cv.layers.RandomColorJitter([0.0, 255.0], 0.5, 0.5, 0.5, 0.5),
        keras_cv.layers.preprocessing.RandomFlip(),
    ]
    pipeline = keras_cv.layers.RandomAugmentationPipeline(
        layers=augmentation_layers, augmentations_per_image=2
    )

    def augment_fn(sample):
        return pipeline(sample)

    _encode_keypoints = partial(
        encode_kps,
        n_labels=len(label_to_cls),
        crop_size=image_size,
    )
    dataset_train = list(
        load_label_studio_kp_detection_dataset(
            dataset_path,
            crop_size=crop_size,
            bbox_labels=bbox_labels,
            label_mapping=label_to_cls,
            max_num_images=max_images,
            split="train",
        )
    )
    X, y = _prepare_inputs_and_targets(_encode_keypoints, dataset_train)
    print(X.shape, y.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices({"images": X, "keypoints": y})
    if augment:
        train_dataset = train_dataset.map(
            augment_fn, num_parallel_calls=tf.data.AUTOTUNE
        )
    train_dataset = train_dataset.shuffle(8 * batch_size).batch(batch_size)

    visualize_kp_dataset(train_dataset)
    plt.savefig(debug_data_path / "train_dataset_sample.jpg", bbox_inches="tight")

    train_dataset = train_dataset.map(
        unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    dataset_val = list(
        load_label_studio_kp_detection_dataset(
            dataset_path,
            crop_size=crop_size,
            bbox_labels=bbox_labels,
            label_mapping=label_to_cls,
            max_num_images=max_images,
            split="val",
        )
    )
    X_val, y_val = _prepare_inputs_and_targets(_encode_keypoints, dataset_val)
    print(X_val.shape, y_val.shape)

    val_dataset = (
        tf.data.Dataset.from_tensor_slices({"images": X_val, "keypoints": y_val})
        .shuffle(8 * batch_size)
        .batch(batch_size)
    )

    visualize_kp_dataset(val_dataset)
    plt.savefig(debug_data_path / "val_dataset_sample.jpg", bbox_inches="tight")
    val_dataset = val_dataset.map(
        unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    train_model = get_regression_model(
        image_size=image_size,
        num_keypoints=num_classes,
    )
    train_model.summary()

    optimizer = tf.keras.optimizers.Adam(1e-3)

    train_model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mae"],
    )
    if fine_tune_from_checkpoint and checkpoint_path.exists():
        train_model.load_weights(checkpoint_path)

    callbacks_list = [
        DvcLiveCallback(path="metrics/keypoint"),
        ReduceLROnPlateau(patience=15, verbose=1, factor=0.5),
    ]

    # -- train model
    train_model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks_list,
        verbose=verbosity,
        batch_size=batch_size,
    )

    #  -- export inference-only model
    image = tf.keras.Input(shape=[None, None, 3], name="image", dtype=tf.uint8)
    resized_image = tf.keras.layers.Resizing(
        crop_size[0], crop_size[1], interpolation="bilinear", crop_to_aspect_ratio=False
    )(image)
    inference_model = get_regression_model(
        image_size=image_size,
        num_keypoints=num_classes,
    )
    predictions = inference_model(resized_image)
    # TODO name outputs
    inference_model = tf.keras.Model(inputs=image, outputs=predictions)
    inference_model.set_weights(train_model.get_weights())
    inference_model.save(model_save_path, save_format="tf")

    # run on a single example image for sanity check if exported detector is working
    example_image_path = Path("example_data/test-image-2.jpg")
    save_file = Path(f"example_predictions/keypoint/{example_image_path.name}")
    save_file.parent.mkdir(exist_ok=True)

    with Image.open(example_image_path) as img:
        input_image = np.array(img).astype(np.float32)
        input_image = np.expand_dims(input_image, axis=0)

        results = inference_model.predict(input_image)[0]
        preds = results.reshape(num_classes, 2)
        points = [
            Point(*preds[cls], name=name).scale(
                input_image.shape[2], input_image.shape[1]
            )
            for cls, name in cls_to_label.items()
        ]
        visualize_keypoints(img, points, savefile=save_file)

    #  -- export warmup data for tf serving
    with Image.open(example_image_path) as img:
        example_image_np = np.array(img)
    save_tf_serving_warmup_request(
        np.expand_dims(example_image_np, axis=0), model_save_path, dtype="uint8"
    )


if __name__ == "__main__":
    main()
