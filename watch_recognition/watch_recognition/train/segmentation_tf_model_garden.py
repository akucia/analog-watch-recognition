import pprint
from typing import Optional

import click
from official.vision.data import tfrecord_lib

import tensorflow as tf

pp = pprint.PrettyPrinter(indent=4)  # Set Pretty Print Indentation
print(tf.__version__)  # Check the version of tensorflow used


def process_record(record):
    keys_to_features = {
        "image/encoded": tfrecord_lib.convert_to_feature(
            tf.io.encode_jpeg(record["image"]).numpy()
        ),
        "image/height": tfrecord_lib.convert_to_feature(record["image"].shape[0]),
        "image/width": tfrecord_lib.convert_to_feature(record["image"].shape[1]),
        "image/segmentation/class/encoded": tfrecord_lib.convert_to_feature(
            tf.io.encode_png(record["segmentation_mask"] - 1).numpy()
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=keys_to_features))
    return example


@click.command()
@click.option("--epochs", default=1)
@click.option("--batch-size", default=8)
@click.option("--seed", default=None, type=int)
def main(
    epochs: int,
    batch_size: int,
    seed: Optional[int],
):
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    print(epochs, batch_size)


if __name__ == "__main__":
    main()
