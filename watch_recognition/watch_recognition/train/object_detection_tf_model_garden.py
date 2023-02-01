"""
Based on https://www.tensorflow.org/tfmodels/vision/object_detection
"""
import pprint
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_models as tfm
from official.core import exp_factory
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.serving import export_saved_model_lib
from official.vision.utils.object_detection import visualization_utils
from PIL import Image
from six import BytesIO

import tensorflow as tf

pp = pprint.PrettyPrinter(indent=4)  # Set Pretty Print Indentation
print(tf.__version__)  # Check the version of tensorflow used


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    if path.startswith("http"):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, "rb").read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return (
        np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)
    )


def build_inputs_for_object_detection(image, input_image_size):
    """Builds Object Detection model inputs for serving."""
    image, _ = resize_and_crop_image(
        image,
        input_image_size,
        padded_size=input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
    )
    return image


def show_batch(raw_records, tf_ex_decoder, category_index, filepath):
    plt.figure(figsize=(20, 20))
    use_normalized_coordinates = True
    min_score_thresh = 0.30
    for i, serialized_example in enumerate(raw_records):
        plt.subplot(1, 3, i + 1)
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = decoded_tensors["image"].numpy().astype("uint8")
        scores = np.ones(shape=(len(decoded_tensors["groundtruth_boxes"])))
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            decoded_tensors["groundtruth_boxes"].numpy(),
            decoded_tensors["groundtruth_classes"].numpy().astype("int"),
            scores,
            category_index=category_index,
            use_normalized_coordinates=use_normalized_coordinates,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            instance_masks=None,
            line_thickness=4,
        )

        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Image-{i + 1}")
    plt.savefig(filepath)


@click.command()
@click.option("--epochs", default=1)
@click.option("--batch-size", default=8)
@click.option("--seed", default=None, type=int)
@click.option("--verbosity", default=1, type=int)
def main(
    epochs: int,
    batch_size: int,
    seed: Optional[int],
    verbosity: int,
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    train_data_input_paths = list(
        map(
            str,
            Path("datasets/tf-records/object-detection/watch-faces/").glob(
                "watch-faces-train-*-of-00001.tfrecord"
            ),
        )
    )

    train_ds = tf.data.TFRecordDataset(train_data_input_paths)
    num_train_examples = int(train_ds.reduce(np.int64(0), lambda x, _: x + 1).numpy())
    valid_data_input_paths = list(
        map(
            str,
            Path("datasets/tf-records/object-detection/watch-faces/").glob(
                "watch-faces-val-*-of-00001.tfrecord"
            ),
        )
    )

    val_ds = tf.data.TFRecordDataset(valid_data_input_paths)

    model_dir = "models/detector"
    export_dir = "models/detector/exported_model/"

    exp_config = exp_factory.get_exp_config("retinanet_resnetfpn_coco")

    num_classes = 1

    HEIGHT, WIDTH = 256, 256
    IMG_SIZE = [HEIGHT, WIDTH, 3]

    # Backbone Config
    exp_config.task.freeze_backbone = False
    exp_config.task.annotation_file = ""

    # Model Config
    exp_config.task.model.input_size = IMG_SIZE
    exp_config.task.model.num_classes = num_classes + 1
    exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = (
        exp_config.task.model.num_classes
    )

    # Training Data Config
    exp_config.task.train_data.input_path = train_data_input_paths
    exp_config.task.train_data.dtype = "float32"
    exp_config.task.train_data.global_batch_size = batch_size
    exp_config.task.train_data.parser.aug_scale_max = 1.0
    exp_config.task.train_data.parser.aug_scale_min = 1.0

    # Validation Data Config
    exp_config.task.validation_data.input_path = valid_data_input_paths
    exp_config.task.validation_data.dtype = "float32"
    exp_config.task.validation_data.global_batch_size = batch_size

    logical_device_names = [
        logical_device.name for logical_device in tf.config.list_logical_devices()
    ]

    if "GPU" in "".join(logical_device_names):
        print("This may be broken in Colab.")
        device = "GPU"
    elif "TPU" in "".join(logical_device_names):
        print("This may be broken in Colab.")
        device = "TPU"
    else:
        print("Running on CPU is slow, so only train for a few steps.")
        device = "CPU"

    print(f"device: {device}")

    steps_per_loop = num_train_examples // batch_size
    train_steps = steps_per_loop * epochs

    exp_config.trainer.steps_per_loop = steps_per_loop
    exp_config.trainer.summary_interval = steps_per_loop * 10
    exp_config.trainer.checkpoint_interval = steps_per_loop * 10
    exp_config.trainer.validation_interval = steps_per_loop * 10
    exp_config.trainer.validation_steps = -1
    exp_config.trainer.train_steps = train_steps
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100
    exp_config.trainer.optimizer_config.learning_rate.type = "cosine"
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05

    pp.pprint(exp_config.as_dict())

    if exp_config.runtime.mixed_precision_dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if "GPU" in "".join(logical_device_names):
        distribution_strategy = tf.distribute.MirroredStrategy()
    elif "TPU" in "".join(logical_device_names):
        tf.tpu.experimental.initialize_tpu_system()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu="/device:TPU_SYSTEM:0"
        )
        distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        print("Warning: this will be really slow.")
        distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

    print("Done")

    with distribution_strategy.scope():
        task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)

    for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
        print()
        print(f"images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}")
        print(f"labels.keys: {labels.keys()}")

    category_index = {
        1: {"id": 1, "name": "WatchFace`"},
    }
    tf_ex_decoder = TfExampleDecoder()

    buffer_size = 20

    raw_train_records = (
        tf.data.TFRecordDataset(exp_config.task.train_data.input_path)
        .shuffle(buffer_size=buffer_size)
        .take(3)
    )
    show_batch(
        raw_train_records,
        tf_ex_decoder,
        category_index,
        filepath="debug/detector/train_dataset_sample.jpg",
    )

    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode="train_and_eval",
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True,
    )

    export_saved_model_lib.export_inference_graph(
        input_type="image_tensor",
        batch_size=1,
        input_image_size=[HEIGHT, WIDTH],
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(model_dir),
        export_dir=export_dir,
    )

    val_ds = tf.data.TFRecordDataset(valid_data_input_paths).take(3)
    show_batch(
        val_ds,
        tf_ex_decoder,
        category_index,
        filepath="debug/detector/val_dataset_sample.jpg",
    )

    imported = tf.saved_model.load(export_dir)
    model_fn = imported.signatures["serving_default"]

    input_image_size = (HEIGHT, WIDTH)
    plt.figure(figsize=(20, 20))
    min_score_thresh = (
        0.30  # Change minimum score for threshold to see all bounding boxes confidences
    )

    for i, serialized_example in enumerate(val_ds):
        plt.subplot(1, 3, i + 1)
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = build_inputs_for_object_detection(
            decoded_tensors["image"], input_image_size
        )
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        image_np = image[0].numpy()
        result = model_fn(image)
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            result["detection_boxes"][0].numpy(),
            result["detection_classes"][0].numpy().astype(int),
            result["detection_scores"][0].numpy(),
            category_index=category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            instance_masks=None,
            line_thickness=4,
        )
        plt.imshow(image_np)
        plt.axis("off")

    plt.savefig("example_predictions/detector/val_ds.jpg")


if __name__ == "__main__":
    main()
