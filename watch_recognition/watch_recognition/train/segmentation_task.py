import argparse
import os
from datetime import datetime
from functools import partial
from typing import Dict
from uuid import uuid1

from watch_recognition.datasets import get_watch_hands_mask_dataset

os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import segmentation_models as sm
import tensorflow as tf

from watch_recognition.data_preprocessing import load_binary_masks_from_coco_dataset
from watch_recognition.models import get_segmentation_model
from watch_recognition.reports import visualize_high_loss_examples


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir",
        type=str,
        required=False,
        default=".",
        help="local or GCS location for writing checkpoints and exporting models",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        default=None,
        help="local or GCS location for reading datasets",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of times to go through the data, default=20",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="number of records to read during each training step, default=128",
    ),
    parser.add_argument(
        "--image-size",
        default=96,
        type=int,
        help="number of records to read during each training step, default=128",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="learning rate for gradient descent, default=.001",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="INFO",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--use-cloud-tb",
        type=bool,
        required=False,
        default=False,
    )
    args, _ = parser.parse_known_args()
    return args.__dict__


def unison_shuffled_copies(a, b, seed=42):
    """https://stackoverflow.com/a/4602224/8814045"""
    np.random.seed(seed)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def train_and_export(
    image_size,
    data_dir: str,
    batch_size,
    learning_rate,
    epochs,
    verbosity,
    export_dir: str = ".",
    model_id: str = None,
    use_cloud_tb: bool = False,
):
    tf.compat.v1.logging.set_verbosity(verbosity)
    if model_id is None:
        model_id = str(uuid1()).split("-")[0]

    image_size = (image_size, image_size)

    X, y, _, float_hashes = load_binary_masks_from_coco_dataset(
        os.path.join(data_dir, "segmentation/train/"),
        image_size=image_size,
    )

    print(X.shape, y.shape)
    X_val, y_val, _, float_hashes_val = load_binary_masks_from_coco_dataset(
        os.path.join(data_dir, "segmentation/validation/"),
        image_size=image_size,
    )

    X = np.concatenate((X, X_val))
    y = np.concatenate((y, y_val))
    hashes = np.concatenate((float_hashes, float_hashes_val))
    train_indices = hashes > 0.2

    X_train = X[train_indices]
    X_val = X[~train_indices]

    y_train = y[train_indices]
    y_val = y[~train_indices]

    N = len(X_train)

    MODEL_NAME = f"effnet-b3-FPN-{image_size}-{N}-weighted-jl/{model_id}"

    dataset_train = get_watch_hands_mask_dataset(
        X_train,
        y_train,
        image_size=image_size,
        batch_size=batch_size,
        augment=True,
        class_weights=[1, 45],
    )
    print(dataset_train)
    dataset_val = get_watch_hands_mask_dataset(
        X_val,
        y_val,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        class_weights=[1, 45],
    )
    dataset_val = dataset_val.cache()

    print(dataset_val)

    model = get_segmentation_model(
        unet_output_layer=None,
        image_size=image_size,
        n_outputs=1,
        output_activation="sigmoid",
    )
    model.summary()

    loss = sm.losses.JaccardLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            sm.metrics.FScore(beta=1, threshold=0.1),
            sm.metrics.IOUScore(threshold=0.1),
        ],
    )

    start = datetime.now()
    if use_cloud_tb:
        logdir = os.path.join(os.environ["AIP_TENSORBOARD_LOG_DIR"], MODEL_NAME)

    else:
        logdir = os.path.join(export_dir, f"tensorboard_logs/{MODEL_NAME}/")

    print(f"tensorboard logs will be saved in {logdir}")

    file_writer_distance_metrics_train = tf.summary.create_file_writer(
        os.path.join(logdir, "/train")
    )
    file_writer_distance_metrics_validation = tf.summary.create_file_writer(
        os.path.join(logdir, "/validation")
    )
    if use_cloud_tb:
        model_path = os.path.join(os.environ["AIP_MODEL_DIR"], MODEL_NAME)
    else:
        model_path = os.path.join(export_dir, f"models/{MODEL_NAME}/")
    print(f"model will will be saved in {model_path}")
    model.fit(
        dataset_train,
        epochs=epochs,
        verbose=2,
        validation_data=dataset_val,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=logdir,
                update_freq="epoch",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.8,
                patience=10,
                min_lr=1e-6,
                cooldown=3,
                verbose=1,
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=partial(
                    visualize_high_loss_examples,
                    dataset=dataset_train,
                    loss=loss,
                    file_writer=file_writer_distance_metrics_train,
                    model=model,
                    every_n_epoch=5,
                )
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=partial(
                    visualize_high_loss_examples,
                    dataset=dataset_val,
                    loss=loss,
                    file_writer=file_writer_distance_metrics_validation,
                    model=model,
                )
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                save_weights_only=False,
                mode="max",
                monitor="val_iou_score",
                save_best_only=True,
            ),
        ],
    )
    elapsed = (datetime.now() - start).seconds
    print(
        f"total training time: {elapsed / 60} minutes, average: {elapsed / 60 / epochs} minutes/epoch"
    )
    # model.save(model_path)


if __name__ == "__main__":
    args = get_args()
    train_and_export(**args)
