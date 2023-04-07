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
from official.vision.ops import preprocess_ops

pp = pprint.PrettyPrinter(indent=4)  # Set Pretty Print Indentation
print(tf.__version__)  # Check the version of tensorflow used

plt.rcParams["font.family"] = "Roboto"


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


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
    experiment_config_dir = Path(experiment_config_dir)

    runtime = RuntimeConfig.from_yaml(str(experiment_config_dir / "runtime.yaml"))
    trainer = TrainerConfig.from_yaml(str(experiment_config_dir / "trainer.yaml"))
    task = SemanticSegmentationTask.from_yaml(str(experiment_config_dir / "task.yaml"))
    exp_config = ExperimentConfig(task=task, trainer=trainer, runtime=runtime)

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

    train_ds = tf.data.TFRecordDataset(exp_config.task.train_data.input_path)
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

    input_image_size = exp_config.task.train_data.output_size

    for i, raw_record in enumerate(train_ds.take(3)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        original_image = tf.io.decode_jpeg(
            example.features.feature["image/encoded"].bytes_list.value[0]
        )
        mask = tf.io.decode_png(
            example.features.feature[
                "image/segmentation/class/encoded"
            ].bytes_list.value[0]
        )
        # TODO image might need to be normalized with image net mean and std
        inference_image = original_image
        inference_image = tf.cast(inference_image, dtype=tf.float32)
        inference_image = preprocess_ops.normalize_image(
            inference_image,
            offset=preprocess_ops.MEAN_RGB,
            scale=preprocess_ops.STDDEV_RGB,
        )
        inference_image, image_info = preprocess_ops.resize_image(
            inference_image, input_image_size
        )
        # drw image, mask, image with mask overlayed and histogram of mask values in a single row
        predicted_mask = model(tf.expand_dims(inference_image, axis=0))["logits"]
        predicted_mask = tf.image.resize(
            predicted_mask, input_image_size, method="bilinear"
        )
        predicted_mask = create_mask(predicted_mask)
        print(predicted_mask.shape)
        original_image = tf.image.resize(original_image, size=input_image_size)
        original_image = tf.cast(original_image, dtype=tf.uint8)
        mask = tf.image.resize(mask, size=input_image_size)

        fig, axs = plt.subplots(1, 6, figsize=(20, 5))
        axs[0].imshow(original_image)

        m = axs[1].imshow(mask)
        m.set_clim(-1, 1)

        axs[2].imshow(original_image)
        axs[2].imshow(mask, alpha=0.5)

        m = axs[3].imshow(predicted_mask)
        m.set_clim(-1, 1)
        plt.colorbar(m, ax=axs[3])

        axs[4].imshow(original_image)
        axs[4].imshow(predicted_mask, alpha=0.5)

        axs[5].hist(predicted_mask.numpy().flatten())

        # I'm not sure if the metrics here are applied correctly, but I want to have any measurement of mask quality
        # calculate ssim between target and predicted masks
        ssim = tf.reduce_mean(
            tf.image.ssim(
                tf.cast(tf.expand_dims(mask, axis=0), dtype=tf.float32),
                tf.cast(tf.expand_dims(predicted_mask, axis=0), dtype=tf.float32),
                max_val=1.0,
            )
        )
        title_iou = f"SSIM: {ssim.numpy():.3f}"

        # calculate the mean IOU between target and predicted masks
        meanIOU = tf.reduce_mean(
            tf.keras.metrics.MeanIoU(num_classes=2, name="mean_iou", dtype=tf.float32)(
                mask, predicted_mask
            )
        )
        title_meaniou = f"iou: {meanIOU.numpy():.3f}"
        # calculate the F1 score between target and predicted masks
        precision = tf.reduce_mean(
            tf.keras.metrics.Precision(dtype=tf.float32)(mask, predicted_mask)
        )
        recall = tf.reduce_mean(
            tf.keras.metrics.Recall(dtype=tf.float32)(mask, predicted_mask)
        )
        f1_score = 2 * (precision * recall) / (precision + recall)
        title_f1 = f"F1-score: {f1_score.numpy():.3f}"

        print(title_iou, title_meaniou, title_f1)
        fig.suptitle(title_iou + " " + title_meaniou + " " + title_f1)
        plt.savefig(debug_dir / f"train_data_predict_{i}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
