import os
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from dvclive.keras import DvcLiveCallback
from PIL import Image
from segmentation_models.metrics import IOUScore
from tensorflow_serving.apis import model_pb2, predict_pb2, prediction_log_pb2

from watch_recognition.label_studio_adapters import (
    load_label_studio_kp_detection_dataset,
)
from watch_recognition.models import get_segmentation_model
from watch_recognition.targets_encoding import (
    _encode_point_to_mask,
    decode_single_point_from_heatmap,
)
from watch_recognition.visualization import visualize_keypoints

os.environ["SM_FRAMEWORK"] = "tf.keras"


def encode_kps_to_mask(
    kps: np.ndarray, n_labels: int, mask_size: Tuple[int, int]
) -> np.ndarray:
    mask = np.zeros((*mask_size, n_labels))
    for kp in kps:
        x_y = np.floor(kp[:2])
        cls = int(kp[2])
        mask[:, :, cls] = _encode_point_to_mask(
            radius=5,
            int_point=x_y,
            mask_size=mask_size,
        )
    return mask


@click.command()
@click.option("--epochs", default=1, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--image-size", default=96, type=int)
@click.option("--max-images", default=None, type=int)
@click.option("--seed", default=None, type=int)
@click.option("--confidence-threshold", default=0.5, type=float)
@click.option("--verbosity", default=1, type=int)
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
    max_images: int,
    seed: int,
    confidence_threshold: float,
    verbosity: int,
    fine_tune_from_checkpoint: bool,
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    label_to_cls = {
        "Top": 0,
        "Center": 1,
        "Crown": 2,
    }  # TODO this should be in params.yaml
    dataset_path = Path("datasets/watch-faces-local.json")
    cls_to_label = {v: k for k, v in label_to_cls.items()}
    num_classes = len(label_to_cls)
    image_size = (image_size, image_size)
    # TODO new data loader - augment before cropping
    bbox_labels = ["WatchFace"]
    checkpoint_path = Path("checkpoints/keypoint/checkpoint")

    crop_size = image_size
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
    X = []
    y = []
    for img, kps in dataset_train:
        X.append(img)
        y.append(encode_kps_to_mask(kps, len(label_to_cls), crop_size))

    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)

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
    X_val = []
    y_val = []
    for img, kps in dataset_val:
        X_val.append(img)
        y_val.append(encode_kps_to_mask(kps, len(label_to_cls), crop_size))

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print(X_val.shape, y_val.shape)

    train_model = get_segmentation_model(
        image_size=image_size,
        n_outputs=num_classes,
        backbone="efficientnetb0",
    )
    train_model.summary()

    loss = sm.losses.JaccardLoss()
    optimizer = tf.keras.optimizers.Adam(1e-3)

    train_model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            IOUScore(),
        ],
    )
    if fine_tune_from_checkpoint and checkpoint_path.exists():
        train_model.load_weights(checkpoint_path)

    callbacks_list = [DvcLiveCallback(path="metrics/keypoint")]
    if not fine_tune_from_checkpoint:
        callbacks_list.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="auto",
                save_freq="epoch",
            ),
        )
    # -- train model
    train_model.fit(
        X,
        y,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=verbosity,
        batch_size=batch_size,
    )

    #  -- export inference-only model
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    resized_image = tf.keras.layers.Resizing(
        crop_size[0], crop_size[1], interpolation="bilinear", crop_to_aspect_ratio=False
    )(image)
    inference_model = get_segmentation_model(
        image_size=image_size,
        n_outputs=num_classes,
        backbone="efficientnetb0",
    )
    predictions = inference_model(resized_image)
    # TODO name outputs
    inference_model = tf.keras.Model(inputs=image, outputs=predictions)
    inference_model.set_weights(train_model.get_weights())
    model_save_path = Path("models/keypoint/")
    inference_model.save(model_save_path)

    # run on a single example image for sanity check if exported detector is working
    example_image_path = Path("example_data/test-image-2.jpg")
    save_file = Path(f"example_predictions/keypoint/{example_image_path.name}")
    save_file.parent.mkdir(exist_ok=True)
    # TODO use predictor?
    with Image.open(example_image_path) as img:

        input_image = np.array(img).astype(np.float32)
        input_image = np.expand_dims(input_image, axis=0)

    results = inference_model.predict(input_image)[0]
    points = []
    for cls, name in cls_to_label.items():
        point = decode_single_point_from_heatmap(
            results[:, :, cls],
            threshold=confidence_threshold,
        )
        if point is not None:
            point = point.rename(name)
            point = point.scale(
                img.width / results.shape[1], img.height / results.shape[0]
            )
            points.append(point)
    visualize_keypoints(img, points, savefile=save_file)

    warmup_tf_record_file = (
        model_save_path / "assets.extra" / "tf_serving_warmup_requests"
    )
    warmup_tf_record_file.parent.mkdir(exist_ok=True, parents=True)
    with tf.io.TFRecordWriter(str(warmup_tf_record_file)) as writer:
        tensor_proto = tf.make_tensor_proto(input_image)
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(signature_name="serving_default"),
            inputs={"image": tensor_proto},
        )
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request)
        )
        writer.write(log.SerializeToString())


if __name__ == "__main__":
    main()
