from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from official.core.base_task import RuntimeConfig
from official.core.base_trainer import ExperimentConfig, TrainerConfig
from official.vision.configs.semantic_segmentation import SemanticSegmentationTask
from official.vision.serving import export_saved_model_lib
from PIL import Image


# TODO unify this with export_retinanet.py
@click.command()
@click.argument("checkpoint_dir", type=click.Path(exists=True))
def main(checkpoint_dir: str):
    """Export Segmentation model for serving and tflite conversion

    Notes:
        Models exported for serving accept batches of images in uint8 format with shape [None, 256, 256, 3]
    and return predictions in normalized coordinates.
        Models exported for tflite conversion accept batches of images in float32 format with shape [1, 256, 256, 3]
        and return predictions in image coordinates. Images have to be resized to 256x256 and normalized to [0, 1]
        before passing them to the model.
    """
    # serving_export_dir = Path("exported_models/segmentation/hands/serving/")
    configs_dir = Path("train-configs/tf-model-garden/watch-hands-segmentation/")
    task = SemanticSegmentationTask.from_yaml(str(configs_dir / "task.yaml"))
    runtime = RuntimeConfig.from_yaml(str(configs_dir / "runtime.yaml"))
    trainer = TrainerConfig.from_yaml(str(configs_dir / "trainer.yaml"))

    exp_config = ExperimentConfig(task=task, trainer=trainer, runtime=runtime)
    model_input_image_size = [
        int(task.train_data.output_size[0]),
        int(task.train_data.output_size[1]),
    ]
    export_saved_model_lib.export_inference_graph(
        input_type="image_tensor",
        batch_size=None,
        input_image_size=model_input_image_size,
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(checkpoint_dir),
        export_dir="exported_models/segmentation/hands/default/",
        log_model_flops_and_params=True,
    )

    export_saved_model_lib.export_inference_graph(
        input_type="tflite",
        batch_size=None,
        input_image_size=model_input_image_size,
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(checkpoint_dir),
        export_dir="exported_models/segmentation/hands/lite/",
        log_model_flops_and_params=True,
    )

    loaded_model = tf.saved_model.load("exported_models/segmentation/hands/default")
    model_fn = loaded_model.signatures["serving_default"]

    example_images = np.stack(
        [
            np.array(Image.open(Path("example_data/IMG_0040.jpg"))),
            np.array(Image.open(Path("example_data/IMG_0039.jpg"))),
        ],
        axis=0,
    )

    # test predict function
    model_fn(tf.convert_to_tensor(example_images))
    print("done")
    # print("testing inference with RetinanetDetectorLocal...")
    # # test inference with RetinanetDetectorLocal
    # detector = RetinaNetDetectorLocal(
    #     Path(serving_export_dir), class_to_label_name={1: "WatchFace"}
    # )
    # examples_dir = Path("example_predictions/detector/")
    # examples_dir.mkdir(parents=True, exist_ok=True)
    # with Image.open(Path("example_data/IMG_0040.jpg")) as img:
    #     plt.figure(figsize=(10, 10))
    #     plt.tight_layout()
    #     detector.predict_and_plot(img)
    #     plt.axis("off")
    #     plt.savefig(examples_dir / "IMG_0040.jpg", bbox_inches="tight")
    # print("done")
    # print("creating warmup request...")
    # save_tf_serving_warmup_request(
    #     example_images,
    #     serving_export_dir,
    #     dtype="uint8",
    #     inputs_key="inputs",
    # )
    # print("done")
    #
    # example_small_images = (
    #     np.stack(
    #         [
    #             np.array(
    #                 ImageOps.pad(
    #                     Image.open(
    #                         Path("example_data/IMG_0040.jpg"),
    #                     ),
    #                     size=(model_input_image_size[0], model_input_image_size[1]),
    #                 ),
    #             ),
    #         ],
    #         axis=0,
    #     ).astype("float32")
    #     / tf.uint8.max
    # )
    #
    # print("exporting tflite model...")
    # # tflite export of the tensorflow model
    # lite = Path("exported_models/detector/lite")
    # lite.mkdir(parents=True, exist_ok=True)
    # imported = tf.saved_model.load(lite)
    # predict_lite = imported.signatures["serving_default"]
    #
    # converter = tf.lite.TFLiteConverter.from_concrete_functions([predict_lite])
    #
    # print("exporting quantized float16 tflite model...")
    # # float16 quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    # tflite_model = converter.convert()
    # # save the quantized model
    # lite_16_dir = Path("exported_models/detector/lite16/")
    # lite_16_dir.mkdir(parents=True, exist_ok=True)
    # with open(lite_16_dir / "model.tflite", "wb") as f:
    #     f.write(tflite_model)
    #
    # print("creating warmup request...")
    # save_tf_serving_warmup_request(
    #     example_small_images,
    #     lite_16_dir,
    #     dtype="float32",
    #     inputs_key="inputs",
    # )
    #
    # print("done")
    # print("exporting quantized int8 tflite model...")
    # # int8 quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    #     tf.lite.OpsSet.SELECT_TF_OPS,
    # ]
    # print("creating representative dataset...")
    #
    # def representative_dataset():
    #     for data in tf.data.TFRecordDataset(
    #         "datasets/tf-records/object-detection/watch-faces/watch-faces-train-00001-of-00001.tfrecord"
    #     ):
    #         parsed = tf.train.Example.FromString(data.numpy())
    #         image_bytes = parsed.features.feature["image/encoded"].bytes_list.value[0]
    #         image_tensor = tf.io.decode_jpeg(image_bytes)
    #         # note: order of operations is important here
    #         # resize_with_pad returns image in float32, convert_image_dtype will not do anything unless it's converted
    #         # back to uint8. It's simpler to just convert to float32 first.
    #         image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    #         image_tensor = tf.image.resize_with_pad(
    #             image_tensor, model_input_image_size[0], model_input_image_size[1]
    #         )
    #         yield [image_tensor]
    #
    # converter.representative_dataset = tf.lite.RepresentativeDataset(
    #     lambda: representative_dataset()
    # )
    # tflite_model = converter.convert()
    # # save the quantized model
    # lite_8_dir = Path("exported_models/detector/lite8/")
    # lite_8_dir.mkdir(parents=True, exist_ok=True)
    # with open(lite_8_dir / "model.tflite", "wb") as f:
    #     f.write(tflite_model)
    #
    # print("creating warmup request...")
    # save_tf_serving_warmup_request(
    #     example_small_images,
    #     lite_8_dir,
    #     dtype="float32",
    #     inputs_key="inputs",
    # )
    #
    # print("done")
    #
    # # cleanup unused data
    # shutil.rmtree("exported_models/detector/default/")
    # shutil.rmtree("exported_models/detector/lite/")


if __name__ == "__main__":
    main()
