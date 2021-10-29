import os
from typing import Dict

os.environ["SM_FRAMEWORK"] = "tf.keras"

import argparse
from datetime import datetime
from functools import partial

import tensorflow as tf

from watch_recognition.data_preprocessing import load_keypoints_data_as_kp
from watch_recognition.datasets import get_watch_keypoints_dataset
from watch_recognition.models import IouLoss2, get_unet_model
from watch_recognition.reports import log_scalar_metrics


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir",
        type=str,
        required=True,
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
    args, _ = parser.parse_known_args()
    return args.__dict__


def train_and_export(
    image_size,
    data_dir: str,
    batch_size,
    learning_rate,
    export_dir: str,
    epochs,
    verbosity,
):
    tf.compat.v1.logging.set_verbosity(verbosity)

    TYPE = "keypoint"
    MODEL_NAME = f"efficientnetb0-unet-sigmoid-{image_size}"

    image_size = (image_size, image_size)
    mask_size = image_size

    X, y, _ = load_keypoints_data_as_kp(
        data_dir + "keypoints/train/",
        autorotate=True,
        image_size=image_size,
    )
    print(X.shape, y.shape)

    X_val, y_val, _ = load_keypoints_data_as_kp(
        data_dir + "keypoints/validation/",
        autorotate=True,
        image_size=image_size,
    )

    print(X_val.shape, y_val.shape)

    dataset_train = get_watch_keypoints_dataset(
        X,
        y,
        augment=True,
        image_size=image_size,
        mask_size=mask_size,
        batch_size=batch_size,
    )
    print(dataset_train)

    dataset_val = get_watch_keypoints_dataset(
        X_val,
        y_val,
        augment=False,
        image_size=image_size,
        mask_size=mask_size,
    )
    print(dataset_val)

    model = get_unet_model(
        unet_output_layer=None,
        image_size=image_size,
        n_outputs=3,
        output_activation="sigmoid",
    )

    model.summary()

    loss = IouLoss2()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model.compile(
        loss=loss,
        optimizer=optimizer,
    )

    start = datetime.now()

    logdir = (
        export_dir + f"/tensorboard_logs/{TYPE}/{MODEL_NAME}/run_{start.timestamp()}"
    )
    print(logdir)
    file_writer_distance_metrics_train = tf.summary.create_file_writer(
        str(logdir) + "/train"
    )
    file_writer_distance_metrics_validation = tf.summary.create_file_writer(
        str(logdir) + "/validation"
    )

    model_path = export_dir + f"/models/{TYPE}/{MODEL_NAME}/run_{start.timestamp()}"
    model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=logdir,
                update_freq="epoch",
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                save_weights_only=False,
                monitor="val_loss",
                save_best_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.8,
                patience=5,
                min_lr=1e-6,
                cooldown=3,
                verbose=1,
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=partial(
                    log_scalar_metrics,
                    X=X,
                    y=y,
                    file_writer=file_writer_distance_metrics_train,
                    model=model,
                )
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=partial(
                    log_scalar_metrics,
                    X=X_val,
                    y=y_val,
                    file_writer=file_writer_distance_metrics_validation,
                    model=model,
                )
            ),
        ],
    )
    elapsed = (datetime.now() - start).seconds
    print(
        f"total training time: {elapsed / 60} minutes, average: {elapsed / 60 / epochs} minutes/epoch"
    )


if __name__ == "__main__":
    args = get_args()
    train_and_export(**args)
