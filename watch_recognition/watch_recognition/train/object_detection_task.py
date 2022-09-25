"""
Adapted from keras-cv examples
https://github.com/keras-team/keras-cv/blob/master/examples/models/object_detection/retina_net/basic/pascal_voc/train.py
"""
from pathlib import Path
from typing import Optional

import click
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dvclive.keras import DvcLiveCallback
from keras_cv import bounding_box
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from watch_recognition.label_studio_adapters import (
    load_label_studio_bbox_detection_dataset,
)
from watch_recognition.serving import save_tf_serving_warmup_request
from watch_recognition.visualization import visualize_detections


def visualize_dataset(dataset: tf.data.Dataset, bounding_box_format: str, ax=None):
    color = tf.constant(((255.0, 0, 255.0),))
    example = next(iter(dataset))
    images, boxes = example["images"], example["bounding_boxes"]
    boxes = keras_cv.bounding_box.convert_format(
        boxes, source=bounding_box_format, target="rel_yxyx", images=images
    )
    boxes = boxes.to_tensor(default_value=-1)
    # TODO remove padded tensors before drawing
    plotted_images = tf.image.draw_bounding_boxes(images, boxes[..., :4], color)
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    if ax is None:
        ax = plt.gca()
    max_imgs = min(9, len(example["images"]))
    for i in range(max_imgs):
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[i].numpy().astype("uint8"))
    ax.axis("off")
    return ax


def convert_format_and_resize_image(bounding_box_format, img_size):
    """Mapping function to create batched image and bbox coordinates"""

    resizing = keras.layers.Resizing(
        height=img_size[0], width=img_size[1], crop_to_aspect_ratio=False
    )

    def apply(image, boxes, labels):
        image = resizing(image)
        boxes = bounding_box.convert_format(
            boxes,
            images=image,
            source="rel_xyxy",
            target=bounding_box_format,
        )

        labels = tf.cast(labels, tf.float32)
        bounding_boxes = tf.concat([boxes, labels], axis=-1)
        return {"images": image, "bounding_boxes": bounding_boxes}

    return apply


def load(
    dataset_path,
    label_to_cls,
    split,
    bounding_box_format,
    batch_size=None,
    shuffle=True,
    shuffle_buffer=None,
    img_size=(512, 512),
    max_images=None,
):

    dataset = tf.data.Dataset.from_generator(
        lambda: load_label_studio_bbox_detection_dataset(
            dataset_path,
            label_mapping=label_to_cls,
            split=split,
            max_num_images=max_images,
        ),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name="image"),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name="objects/bbox"),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32, name="objects/label"),
        ),
    )
    _map_fn = convert_format_and_resize_image(
        bounding_box_format=bounding_box_format, img_size=img_size
    )
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if shuffle:
        if not batch_size and not shuffle_buffer:
            raise ValueError(
                "If `shuffle=True`, either a `batch_size` or `shuffle_buffer` must be "
                "provided to `keras_cv.datasets.pascal_voc.load().`"
            )
        shuffle_buffer = shuffle_buffer or 8 * batch_size
    dataset = dataset.shuffle(shuffle_buffer)
    if batch_size is not None:
        if max_images is not None:
            batch_size = min(batch_size, max_images)
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
        )
    return dataset


def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]


@click.command()
@click.option("--epochs", default=1)
@click.option("--batch-size", default=32)
@click.option("--max-images", default=None, type=int)
@click.option("--seed", default=None, type=int)
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
    max_images: Optional[int],
    seed: Optional[int],
    verbosity: int,
    fine_tune_from_checkpoint: bool,
):
    # -- configuration
    if fine_tune_from_checkpoint:
        print("fine tuning is currently disabled")
        fine_tune_from_checkpoint = False

    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    label_to_cls = {"WatchFace": 0}  # TODO this should be in params.yaml
    cls_to_label = {v: k for k, v in label_to_cls.items()}
    image_size = (512, 512)
    num_classes = len(label_to_cls)
    checkpoint_path = Path("checkpoints/detector/checkpoint")
    model_save_path = Path("models/detector/")
    model_save_path.mkdir(exist_ok=True, parents=True)
    metrics_path = "metrics/detector"
    debug_data_path = Path("debug/detector")
    debug_data_path.mkdir(exist_ok=True, parents=True)

    # -- setup dataset pipeline
    dataset_path = Path("datasets/watch-faces-local.json")
    train_dataset = load(
        dataset_path,
        label_to_cls=label_to_cls,
        bounding_box_format="xywh",
        split="train",
        batch_size=batch_size,
        img_size=image_size,
        max_images=max_images,
    )
    val_dataset = load(
        dataset_path,
        label_to_cls=label_to_cls,
        bounding_box_format="xywh",
        split="val",
        batch_size=batch_size,
        img_size=image_size,
        max_images=max_images,
    )
    AUGMENTATION_LAYERS = [
        keras_cv.layers.ChannelShuffle(groups=3),
        keras_cv.layers.RandomHue(0.5, [0.0, 255.0]),
        keras_cv.layers.AutoContrast(value_range=[0.0, 255.0]),
        keras_cv.layers.GridMask(),
        keras_cv.layers.RandomColorJitter([0.0, 255.0], 0.5, 0.5, 0.5, 0.5),
        keras_cv.layers.preprocessing.RandomFlip(bounding_box_format="xywh"),
        keras_cv.layers.preprocessing.RandomShear(
            x_factor=0.15,
            y_factor=0.15,
            bounding_box_format="xywh",
            fill_mode="nearest",
        ),
    ]
    pipeline = keras_cv.layers.RandomAugmentationPipeline(
        layers=AUGMENTATION_LAYERS, augmentations_per_image=2
    )

    def augment_fn(sample):
        return pipeline(sample)

    train_dataset = train_dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    visualize_dataset(train_dataset, bounding_box_format="xywh")
    plt.savefig(debug_data_path / "train_dataset_sample.jpg", bbox_inches="tight")

    visualize_dataset(val_dataset, bounding_box_format="xywh")
    plt.savefig(debug_data_path / "val_dataset_sample.jpg", bbox_inches="tight")

    train_dataset = train_dataset.map(
        unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(
        unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # -- setup train model

    train_model = keras_cv.models.RetinaNet(
        classes=num_classes,
        bounding_box_format="xywh",
        backbone="resnet50",
        backbone_weights="imagenet",
        include_rescaling=True,
        evaluate_train_time_metrics=False,
    )
    if fine_tune_from_checkpoint and checkpoint_path.exists():
        train_model.load_weights(checkpoint_path)
    metrics = [
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=[0],
            bounding_box_format="xywh",
            name="MeanAveragePrecision",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=[0],
            bounding_box_format="xywh",
            max_detections=100,
            name="Recall",
        ),
    ]

    optimizer = tf.optimizers.Adam(learning_rate=3e-4)

    train_model.compile(
        classification_loss=keras_cv.losses.FocalLoss(
            from_logits=True, reduction="none"
        ),
        box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
        optimizer=optimizer,
        metrics=metrics,
    )

    callbacks_list = [
        DvcLiveCallback(path=metrics_path),
        ReduceLROnPlateau(patience=20, verbose=1),
        # callbacks_lib.EarlyStopping(patience=50),
    ]

    # if not fine_tune_from_checkpoint:
    #     callbacks_list.append(
    #         tf.keras.callbacks.ModelCheckpoint(
    #             checkpoint_path,
    #             monitor="val_loss",
    #             verbose=1,
    #             save_best_only=True,
    #             save_weights_only=True,
    #             mode="auto",
    #             save_freq="epoch",
    #         ),
    #     )

    # -- train model
    train_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=verbosity,
    )
    #  -- export inference-only model
    image = tf.keras.Input(shape=[None, None, 3], name="image", dtype="uint8")
    resized_image = tf.keras.layers.Resizing(
        *image_size,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
    )(image)
    outputs = train_model(resized_image)
    outputs = train_model.decode_training_predictions(resized_image, outputs)
    # todo normalize outputs' coordinates
    outputs = tf.keras.layers.Lambda(
        lambda x: x.to_tensor(default_value=-1), name="bboxes"
    )(outputs)
    inference_model = keras.Model(inputs=image, outputs=outputs)

    inference_model.save(model_save_path)

    # run on a single example image for sanity check if exported detector is working
    example_image_path = Path("example_data/test-image.jpg")
    with Image.open(example_image_path) as img:
        example_image_np = np.array(img)
        input_image = np.expand_dims(example_image_np, axis=0)
        boxes = inference_model.predict(input_image, False, verbose=0)[0]
        boxes[:, 0] *= img.width / 512
        boxes[:, 1] *= img.height / 512
        boxes[:, 2] *= img.width / 512
        boxes[:, 3] *= img.height / 512

    class_names = [cls_to_label[x] for x in boxes[:, 4]]
    save_file = Path(f"example_predictions/detector/{example_image_path.name}")
    save_file.parent.mkdir(exist_ok=True)

    visualize_detections(
        example_image_np,
        boxes[:, :4],
        class_names,
        boxes[:, 5],
        savefile=save_file,
    )

    #  -- export warmup data for tf serving
    example_image_path = Path("example_data/test-image.jpg")
    with Image.open(example_image_path) as img:
        example_image_np = np.array(img)

    save_tf_serving_warmup_request(
        np.expand_dims(example_image_np, axis=0), model_save_path, dtype="uint8"
    )


if __name__ == "__main__":
    main()
