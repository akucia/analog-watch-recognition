from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from official.core.base_task import RuntimeConfig
from official.core.base_trainer import ExperimentConfig, TrainerConfig
from official.vision.configs.semantic_segmentation import SemanticSegmentationTask
from official.vision.serving import export_saved_model_lib
from PIL import Image, ImageOps

from watch_recognition.predictors import HandPredictorLocal
from watch_recognition.serving import save_tf_serving_warmup_request


# TODO unify this with export_retinanet.py
@click.command()
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.argument("configs_dir", type=click.Path(exists=True))
@click.argument(
    "export_dir", type=click.Path(), default="exported_models/segmentation/hands"
)
@click.argument(
    "example_predictions_dir",
    type=click.Path(),
    default="example_predictions/segmentation/hands",
)
def main(
    checkpoint_dir: str, configs_dir: str, export_dir: str, example_predictions_dir: str
):
    """Export Segmentation model for serving and tflite conversion

    Notes:
        Models exported for serving accept batches of images in uint8 format with shape [None, 256, 256, 3]
    and return predictions in normalized coordinates.
        Models exported for tflite conversion accept batches of images in float32 format with shape [1, 256, 256, 3]
        and return predictions in image coordinates. Images have to be resized to 256x256 and normalized to [0, 1]
        before passing them to the model.
    """
    configs_dir = Path(configs_dir)
    task = SemanticSegmentationTask.from_yaml(str(configs_dir / "task.yaml"))
    runtime = RuntimeConfig.from_yaml(str(configs_dir / "runtime.yaml"))
    trainer = TrainerConfig.from_yaml(str(configs_dir / "trainer.yaml"))

    export_dir = Path(export_dir)

    exp_config = ExperimentConfig(task=task, trainer=trainer, runtime=runtime)
    model_input_image_size = [
        int(task.train_data.output_size[0]),
        int(task.train_data.output_size[1]),
    ]
    serving_default_export = export_dir / "default"
    lite_export_dir = export_dir / "lite"
    export_saved_model_lib.export_inference_graph(
        input_type="image_tensor",
        batch_size=None,
        input_image_size=model_input_image_size,
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(checkpoint_dir),
        export_dir=str(serving_default_export),
        log_model_flops_and_params=True,
    )

    export_saved_model_lib.export_inference_graph(
        input_type="tflite",
        batch_size=None,
        input_image_size=model_input_image_size,
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(checkpoint_dir),
        export_dir=str(lite_export_dir),
        log_model_flops_and_params=True,
    )

    loaded_model = tf.saved_model.load(str(serving_default_export))
    model_fn = loaded_model.signatures["serving_default"]

    example_images = np.stack(
        [
            np.array(Image.open(Path("example_data/example-1.jpg")).resize((512, 512))),
            np.array(Image.open(Path("example_data/example-2.jpg")).resize((512, 512))),
        ],
        axis=0,
    )

    # test predict function
    model_fn(tf.convert_to_tensor(example_images))
    print("done")
    print("testing inference with HandPredictorLocal...")
    # test inference with HandPredictorLocal
    hands_model = HandPredictorLocal(Path(serving_default_export))
    examples_dir = Path(example_predictions_dir)
    examples_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(Path("example_data/example-1.jpg")) as img:
        plt.figure(figsize=(10, 10))
        plt.tight_layout()
        hands_model.predict_and_plot(img)
        plt.axis("off")
        plt.savefig(examples_dir / "example-1.jpg", bbox_inches="tight")
    print("done")
    print("creating warmup request...")
    save_tf_serving_warmup_request(
        example_images,
        serving_default_export,
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
                            Path("example_data/example-1.jpg"),
                        ),
                        size=(model_input_image_size[0], model_input_image_size[1]),
                    ),
                ),
            ],
            axis=0,
        ).astype("float32")
        / tf.uint8.max
    )
    _ = example_small_images

    print("done")
    print("exporting quantized int8 tflite model...")
    lite8_export_dir = lite_export_dir.parent / "lite8"
    lite8_export_dir.mkdir(parents=True, exist_ok=True)
    imported = tf.saved_model.load(lite_export_dir)
    predict_lite = imported.signatures["serving_default"]
    converter = tf.lite.TFLiteConverter.from_concrete_functions([predict_lite])
    # int8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    print("creating representative dataset...")

    # TODO provide path to the dataset as argument
    def representative_dataset():
        for data in tf.data.TFRecordDataset(
            "datasets/tf-records/segmentation/watch-hands/watch-hands-train-00001-of-00001.tfrecord"
        ):
            parsed = tf.train.Example.FromString(data.numpy())
            image_bytes = parsed.features.feature["image/encoded"].bytes_list.value[0]
            image_tensor = tf.io.decode_jpeg(image_bytes)
            # note (originally left for resize_with_pad): order of operations is important here
            # resize returns image in float32, convert_image_dtype will not do anything unless it's converted
            # back to uint8. It's simpler to just convert to float32 first.
            image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
            image_tensor = tf.image.resize(
                image_tensor, (model_input_image_size[0], model_input_image_size[1])
            )
            # this gives error
            # ValueError: You need to provide either a dictionary with input names and values,
            # a tuple with signature key and a dictionary with input names and values, or an array
            # with input values in the order of input tensors of the graph in the representative_dataset function.
            # Unsupported value from dataset (here was an array)
            yield tf.expand_dims(image_tensor, axis=0)

    # converter.representative_dataset = tf.lite.RepresentativeDataset(
    #     lambda: representative_dataset()
    # )
    # tflite_model = converter.convert()
    # #
    # # TODO generate tests example render with the model
    # # interpreter = tf.lite.Interpreter(model_content=tflite_model)
    # # interpreter.allocate_tensors()
    # # input_details = interpreter.get_input_details()
    # # output_details = interpreter.get_output_details()
    # # interpreter.set_tensor(input_details[0]["index"], example_small_images)
    # # interpreter.invoke()
    # # output_data = interpreter.get_tensor(output_details[0]["index"])
    #
    # with open(lite8_export_dir / "model.tflite", "wb") as f:
    #     f.write(tflite_model)
    #
    # print("creating warmup request...")
    # save_tf_serving_warmup_request(
    #     example_small_images,
    #     lite8_export_dir,
    #     dtype="float32",
    #     inputs_key="inputs",
    # )
    #
    # print("done")


if __name__ == "__main__":
    main()
