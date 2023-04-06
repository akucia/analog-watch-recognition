import pprint
from pathlib import Path
from typing import Optional

import click
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
from matplotlib import pyplot as plt
from official.core.base_task import RuntimeConfig
from official.core.base_trainer import ExperimentConfig, TrainerConfig
from official.vision.configs.semantic_segmentation import SemanticSegmentationTask

pp = pprint.PrettyPrinter(indent=4)  # Set Pretty Print Indentation
print(tf.__version__)  # Check the version of tensorflow used

plt.rcParams["font.family"] = "Roboto"


@click.command()
@click.argument(
    "experiment-config-dir",
    default="train-configs/tf-model-garden/watch-hands-segmentation/",
    type=click.Path(exists=True),
)
@click.option("--seed", default=42, type=int)
def main(
    experiment_config_dir: str,
    seed: Optional[int] = 42,
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    model_dir = "models/segmentation"
    train_data_tfrecords = "datasets/tf-records/segmentation/watch-hands/watch-hands-train-00001-of-00001.tfrecord"
    # val_data_tfrecords = "datasets/tf-records/segmentation/watch-hands/watch-hands-val-00001-of-00001.tfrecord"
    # export_dir = "exported_models/segmentation"
    experiment_config_dir = Path(experiment_config_dir)

    runtime = RuntimeConfig.from_yaml(str(experiment_config_dir / "runtime.yaml"))
    trainer = TrainerConfig.from_yaml(str(experiment_config_dir / "trainer.yaml"))
    task = SemanticSegmentationTask.from_yaml(str(experiment_config_dir / "task.yaml"))
    exp_config = ExperimentConfig(task=task, trainer=trainer, runtime=runtime)

    # exp_config = tfm.core.exp_factory.get_exp_config("seg_deeplabv3_pascal")

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
    print(f"dataset has {num_train_examples} examples")

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
        print(
            f"images.shape: {str(images.shape):16} "
            f"images.dtype: {images.dtype!r} "
            f"images.max_val: {tf.reduce_max(images).numpy():5}"
        )
        print(
            f"masks.shape: {str(masks['masks'].shape):16} "
            f"masks.dtype: {masks['masks'].dtype!r} "
            f"masks.max_val: {tf.reduce_max(masks['masks']).numpy():5}"
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
        nrows = 4
        images = images[:nrows]
        masks = masks[:nrows]
        fig, axs = plt.subplots(nrows, 2, figsize=(9, 16))
        titles = ["Image", "Mask"]

        for ax, col in zip(axs[0], titles):
            ax.set_title(col)

        for j, (image, mask) in enumerate(zip(images, masks)):
            # draw the input image
            axs[j][0].imshow(tf.keras.utils.array_to_img(image))
            axs[j][0].axis("off")
            # draw the mask
            mask = np.where(mask == 255, np.zeros_like(mask) - 1, mask)
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

    for i, batch in enumerate(
        task.build_inputs(exp_config.task.train_data).take(num_examples)
    ):
        images, targets = batch
        masks = targets["masks"]

        # limit to just 4 examples per batch
        nrows = 4
        images = images[:nrows] * 255
        masks = masks[:nrows]
        logits = model(images)["logits"].numpy()
        preds = np.expand_dims(np.argmax(logits, axis=-1), axis=-1)
        print(preds.shape)
        fig, axs = plt.subplots(nrows, 3, figsize=(9, 16))
        titles = ["Image", "Mask", "Predictions"]

        for ax, col in zip(axs[0], titles):
            ax.set_title(col)

        for j, (image, mask, pred) in enumerate(zip(images, masks, preds)):
            # draw the input image
            pil_img = tf.keras.utils.array_to_img(image)
            axs[j][0].imshow(pil_img)
            axs[j][0].axis("off")

            mask = np.where(mask == 255, np.zeros_like(mask) - 1, mask)
            axs[j][1].imshow(tf.keras.utils.array_to_img(mask))
            axs[j][1].axis("off")

            axs[j][2].imshow(tf.keras.utils.array_to_img(pred))
            axs[j][2].axis("off")

        fig.tight_layout()
        plt.savefig(debug_dir / f"train_data_predict_{i}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
