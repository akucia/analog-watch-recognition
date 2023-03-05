import shutil
from pathlib import Path

import click
import numpy as np
from matplotlib import pyplot as plt
from official.core.base_task import RuntimeConfig
from official.core.base_trainer import ExperimentConfig, TrainerConfig
from official.vision.configs.retinanet import RetinaNetTask
from official.vision.serving import export_saved_model_lib
from PIL import Image, ImageOps

import tensorflow as tf
from watch_recognition.predictors import RetinaNetDetectorLocal
from watch_recognition.serving import save_tf_serving_warmup_request


@click.command()
def main():
    """Export RetinaNet model for serving and tflite conversion with fp16 and int8 quantization
    Every exported model includes a warmup request for tf serving.

    Notes:
        Models exported for serving accept batches of images in uint8 format with shape [None, None, None, 3]
    and return predictions in normalized coordinates.
        Models exported for tflite conversion accept batches of images in float32 format with shape [1, 256, 256, 3]
        and return predictions in image coordinates.
        Images have to be resized to 256x256 and normalized to [0, 1] before passing them to the model.
    """
    serving_export_dir = Path("exported_models/detector/serving/")
    configs_dir = Path("train-configs/tf-model-garden/watch-face-detector/")
    retinanet_task = RetinaNetTask.from_yaml(str(configs_dir / "retinanet_task.yaml"))
    runtime = RuntimeConfig.from_yaml(str(configs_dir / "runtime.yaml"))
    trainer = TrainerConfig.from_yaml(str(configs_dir / "trainer.yaml"))

    exp_config = ExperimentConfig(task=retinanet_task, trainer=trainer, runtime=runtime)
    model_input_image_size = [
        int(retinanet_task.model.input_size[0]),
        int(retinanet_task.model.input_size[1]),
    ]
    export_saved_model_lib.export_inference_graph(
        input_type="image_tensor",
        batch_size=None,
        input_image_size=model_input_image_size,
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint("models/detector/"),
        export_dir="exported_models/detector/default/",
        log_model_flops_and_params=True,
    )

    export_saved_model_lib.export_inference_graph(
        input_type="tflite",
        batch_size=None,
        input_image_size=model_input_image_size,
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint("models/detector/"),
        export_dir="exported_models/detector/lite/",
        log_model_flops_and_params=True,
    )

    loaded_model = tf.saved_model.load("exported_models/detector/default/")
    model_fn = loaded_model.signatures["serving_default"]

    @tf.function(
        jit_compile=False,
        input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.uint8)],
    )
    def predict(inputs):
        method = tf.image.ResizeMethod.BILINEAR
        image_size = tf.cast(tf.shape(inputs)[1:3], tf.float32)
        # this function pads the image equally on both sides
        scaled_inputs = tf.image.resize_with_pad(
            inputs,
            target_height=model_input_image_size[1],
            target_width=model_input_image_size[0],
            method=method,
        )
        scaled_inputs = tf.cast(scaled_inputs, dtype=tf.uint8)
        result = model_fn(scaled_inputs)

        # calculate the amount of padding that was added
        max_size = tf.reduce_max(image_size)

        dx = (max_size - image_size[1]) / max_size * model_input_image_size[0]
        dy = (max_size - image_size[0]) / max_size * model_input_image_size[1]

        # shift the boxes by half of the padding
        boxes = tf.stack(
            [
                result["detection_boxes"][:, :, 0] - (dy / 2),  # y_min
                result["detection_boxes"][:, :, 1] - (dx / 2),  # x_min
                result["detection_boxes"][:, :, 2] - (dy / 2),  # y_max
                result["detection_boxes"][:, :, 3] - (dx / 2),  # x_max
            ],
            axis=-1,
        )
        # scale the boxes to normalized coordinates to make them independent of the input size
        boxes = tf.stack(
            [
                boxes[:, :, 0] / (model_input_image_size[1] - dy),  # y_min
                boxes[:, :, 1] / (model_input_image_size[0] - dx),  # x_min
                boxes[:, :, 2] / (model_input_image_size[1] - dy),  # y_max
                boxes[:, :, 3] / (model_input_image_size[0] - dx),  # x_max
            ],
            axis=-1,
        )
        return {
            "detection_boxes": boxes,
            "detection_classes": result["detection_classes"],
            "detection_scores": result["detection_scores"],
            "num_detections": result["num_detections"],
        }

    example_images = np.stack(
        [
            np.array(Image.open(Path("example_data/IMG_0040.jpg"))),
            np.array(Image.open(Path("example_data/IMG_0039.jpg"))),
        ],
        axis=0,
    )

    # test predict function
    predict(tf.convert_to_tensor(example_images))
    print("exporting model with predict function for serving...")
    # Export the model
    tf.saved_model.save(
        loaded_model,
        serving_export_dir,
        signatures={
            "default": loaded_model.signatures["serving_default"],
            "serving_default": predict.get_concrete_function(),
            "predict": predict.get_concrete_function(),
        },
    )
    print("done")
    print("testing inference with RetinanetDetectorLocal...")
    # test inference with RetinanetDetectorLocal
    detector = RetinaNetDetectorLocal(
        Path(serving_export_dir), class_to_label_name={1: "WatchFace"}
    )
    with Image.open(Path("example_data/IMG_0040.jpg")) as img:
        plt.figure(figsize=(10, 10))
        plt.tight_layout()
        detector.predict_and_plot(img)
        plt.axis("off")
        plt.savefig("example_predictions/detector/IMG_0040.jpg", bbox_inches="tight")
    print("done")
    print("creating warmup request...")
    save_tf_serving_warmup_request(
        example_images,
        serving_export_dir,
        dtype="uint8",
        inputs_key="inputs",
    )
    print("done")

    example_small_images = (
        np.stack(
            [
                np.array(
                    ImageOps.pad(
                        Image.open(
                            Path("example_data/IMG_0040.jpg"),
                        ),
                        size=(model_input_image_size[0], model_input_image_size[1]),
                    ),
                ),
            ],
            axis=0,
        ).astype("float32")
        / tf.uint8.max
    )

    print("exporting tflite model...")
    # tflite export of the tensorflow model
    lite = Path("exported_models/detector/lite")
    lite.mkdir(parents=True, exist_ok=True)
    imported = tf.saved_model.load(lite)
    predict_lite = imported.signatures["serving_default"]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([predict_lite])

    print("exporting quantized float16 tflite model...")
    # float16 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    # save the quantized model
    lite_16_dir = Path("exported_models/detector/lite16/")
    lite_16_dir.mkdir(parents=True, exist_ok=True)
    with open(lite_16_dir / "model.tflite", "wb") as f:
        f.write(tflite_model)

    print("creating warmup request...")
    save_tf_serving_warmup_request(
        example_small_images,
        lite_16_dir,
        dtype="float32",
        inputs_key="inputs",
    )

    print("done")
    print("exporting quantized int8 tflite model...")
    # int8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    print("creating representative dataset...")

    def representative_dataset():
        for data in tf.data.TFRecordDataset(
            "datasets/tf-records/object-detection/watch-faces/watch-faces-train-00001-of-00001.tfrecord"
        ):
            parsed = tf.train.Example.FromString(data.numpy())
            image_bytes = parsed.features.feature["image/encoded"].bytes_list.value[0]
            image_tensor = tf.io.decode_jpeg(image_bytes)
            # note: order of operations is important here
            # resize_with_pad returns image in float32, convert_image_dtype will not do anything unless it's converted
            # back to uint8. It's simpler to just convert to float32 first.
            image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
            image_tensor = tf.image.resize_with_pad(
                image_tensor, model_input_image_size[0], model_input_image_size[1]
            )
            yield [image_tensor]

    converter.representative_dataset = tf.lite.RepresentativeDataset(
        lambda: representative_dataset()
    )
    tflite_model = converter.convert()
    # save the quantized model
    lite_8_dir = Path("exported_models/detector/lite8/")
    lite_8_dir.mkdir(parents=True, exist_ok=True)
    with open(lite_8_dir / "model.tflite", "wb") as f:
        f.write(tflite_model)

    print("creating warmup request...")
    save_tf_serving_warmup_request(
        example_small_images,
        lite_8_dir,
        dtype="float32",
        inputs_key="inputs",
    )

    print("done")

    # cleanup unused data
    shutil.rmtree("exported_models/detector/default/")
    shutil.rmtree("exported_models/detector/lite/")


if __name__ == "__main__":
    main()
