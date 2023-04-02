import pprint
from pathlib import Path
from typing import Optional

import click
import numpy as np
import tensorflow_models as tfm
from matplotlib import pyplot as plt

import tensorflow as tf

pp = pprint.PrettyPrinter(indent=4)  # Set Pretty Print Indentation
print(tf.__version__)  # Check the version of tensorflow used

plt.rcParams["font.family"] = "Roboto"


@click.command()
@click.option("--seed", default=42, type=int)
def main(
    seed: Optional[int] = 42,
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    model_dir = "models/segmentation"
    train_data_tfrecords = "datasets/tf-records/segmentation/watch-hands/watch-hands-train-00001-of-00001.tfrecord"
    val_data_tfrecords = "datasets/tf-records/segmentation/watch-hands/watch-hands-val-00001-of-00001.tfrecord"
    # export_dir = "exported_models/segmentation"

    exp_config = tfm.core.exp_factory.get_exp_config("seg_deeplabv3_pascal")

    num_classes = 2
    WIDTH, HEIGHT = 128, 128
    input_size = [HEIGHT, WIDTH, 3]
    BATCH_SIZE = 16

    # Backbone Config
    exp_config.task.init_checkpoint = None
    exp_config.task.freeze_backbone = True

    # Model Config
    exp_config.task.model.num_classes = num_classes
    exp_config.task.model.input_size = input_size

    # Training Data Config
    exp_config.task.train_data.aug_scale_min = 1.0
    exp_config.task.train_data.aug_scale_max = 1.0
    exp_config.task.train_data.input_path = train_data_tfrecords
    exp_config.task.train_data.global_batch_size = BATCH_SIZE
    exp_config.task.train_data.dtype = "float32"
    exp_config.task.train_data.output_size = [HEIGHT, WIDTH]
    exp_config.task.train_data.preserve_aspect_ratio = False
    exp_config.task.train_data.seed = seed  # Reproducable Training Data

    # Validation Data Config
    exp_config.task.validation_data.input_path = val_data_tfrecords
    exp_config.task.validation_data.global_batch_size = BATCH_SIZE
    exp_config.task.validation_data.dtype = "float32"
    exp_config.task.validation_data.output_size = [HEIGHT, WIDTH]
    exp_config.task.validation_data.preserve_aspect_ratio = False
    exp_config.task.validation_data.groundtruth_padded_size = [HEIGHT, WIDTH]
    exp_config.task.validation_data.seed = seed  # Reproducable Validation Data

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

    train_ds = tf.data.TFRecordDataset(train_data_tfrecords)
    num_train_examples = int(train_ds.reduce(np.int64(0), lambda x, _: x + 1).numpy())
    # train_steps = 2000
    train_steps = 100
    exp_config.trainer.steps_per_loop = num_train_examples // BATCH_SIZE
    exp_config.trainer.steps_per_loop = 1

    exp_config.trainer.summary_interval = (
        exp_config.trainer.steps_per_loop
    )  # steps_per_loop = num_of_validation_examples // eval_batch_size
    exp_config.trainer.checkpoint_interval = exp_config.trainer.steps_per_loop
    exp_config.trainer.validation_interval = exp_config.trainer.steps_per_loop
    exp_config.trainer.validation_steps = -1
    exp_config.trainer.train_steps = train_steps
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = (
        exp_config.trainer.steps_per_loop
    )
    exp_config.trainer.optimizer_config.learning_rate.type = "cosine"
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05

    pp.pprint(exp_config.as_dict())

    # Setting up the Strategy
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

    for images, masks in task.build_inputs(exp_config.task.train_data).take(1):
        print()
        print(f"images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}")
        print(
            f'masks.shape: {str(masks["masks"].shape):16} images.dtype: {masks["masks"].dtype!r}'
        )

    debug_dir = Path("debug/segmentation/")
    debug_dir.mkdir(parents=True, exist_ok=True)

    num_examples = 3

    for i, batch in enumerate(
        task.build_inputs(exp_config.task.train_data).take(num_examples)
    ):
        images, targets = batch
        masks = targets["masks"]

        # limit to just 4 examples per batch
        images = images[:4]
        masks = masks[:4]
        fig, axs = plt.subplots((len(images)), 2, figsize=(9, 16))
        titles = ["Image", "Mask"]

        for ax, col in zip(axs[0], titles):
            ax.set_title(col)

        for j, (image, mask) in enumerate(zip(images, masks)):
            # draw the input image
            axs[j][0].imshow(tf.keras.utils.array_to_img(image))
            axs[j][0].axis("off")
            # draw the mask
            axs[j][1].imshow(tf.keras.utils.array_to_img(mask))
            axs[j][1].axis("off")

        fig.tight_layout()
        plt.savefig(debug_dir / f"train_data_{i}.png", bbox_inches="tight")

    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode="train_and_eval",
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True,
    )
    _ = model
    _ = eval_logs


if __name__ == "__main__":
    main()
