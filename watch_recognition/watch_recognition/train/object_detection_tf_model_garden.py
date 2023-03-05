"""
Based on https://www.tensorflow.org/tfmodels/vision/object_detection
"""
import pprint
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_models as tfm
from official.core.base_task import RuntimeConfig
from official.core.base_trainer import ExperimentConfig, TrainerConfig
from official.vision.configs.common import Augmentation
from official.vision.configs.retinanet import Parser, RetinaNetTask
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils

import tensorflow as tf

pp = pprint.PrettyPrinter(indent=4)
print(tf.__version__)


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
@click.argument(
    "experiment-config-dir",
    default="train-configs/tf-model-garden/watch-face-detector.yaml",
    type=click.Path(exists=True),
)
@click.option("--seed", default=None, type=int)
def main(
    experiment_config_dir: str,
    seed: Optional[int],
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    experiment_config_dir = Path(experiment_config_dir)

    model_dir = "models/detector"

    runtime = RuntimeConfig.from_yaml(str(experiment_config_dir / "runtime.yaml"))
    trainer = TrainerConfig.from_yaml(str(experiment_config_dir / "trainer.yaml"))
    retinanet_task = RetinaNetTask.from_yaml(
        str(experiment_config_dir / "retinanet_task.yaml")
    )
    retinanet_task.train_data.parser = Parser(
        aug_rand_hflip=True,
        aug_scale_min=0.8,
        aug_scale_max=1.2,
        aug_type=Augmentation(
            type="randaug",
        ),
    )

    exp_config = ExperimentConfig(task=retinanet_task, trainer=trainer, runtime=runtime)
    train_data_input_paths = exp_config.task.train_data.input_path
    train_ds = tf.data.TFRecordDataset(train_data_input_paths)
    num_train_examples = int(train_ds.reduce(np.int64(0), lambda x, _: x + 1).numpy())
    print(f"Number of train examples: {num_train_examples}")

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
        1: {"id": 1, "name": "WatchFace"},
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


if __name__ == "__main__":
    main()
