from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from watch_recognition.label_studio_adapters import (
    _load_label_studio_bbox_detection_dataset,
)

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_list_feature(values):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def encode_example(image, bboxes, labels):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # change xyxy to yxyx format
    feature = {
        "image": _bytes_feature(image),
        "objects/bbox/xmin": _float_list_feature(bboxes[:, 0].tolist()),
        "objects/bbox/ymin": _float_list_feature(bboxes[:, 1].tolist()),
        "objects/bbox/xmax": _float_list_feature(bboxes[:, 2].tolist()),
        "objects/bbox/ymax": _float_list_feature(bboxes[:, 3].tolist()),
        "objects/bbox/label": _int64_list_feature(labels.tolist()),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


@click.command()
def main():
    label_to_cls = {"WatchFace": 0}  # TODO this should be in params.yaml
    # TODO split into workers, one worker per split?
    filenames = []
    for split in {"train", "val", "test"}:
        filename = Path(f"datasets/tfrecords/detection/watch-faces_{split}.tfrecord")
        filename.parent.mkdir(exist_ok=True, parents=True)
        filenames.append(filename)
        with tf.io.TFRecordWriter(str(filename)) as writer:
            dataset_gen = _load_label_studio_bbox_detection_dataset(
                Path("datasets/watch-faces-local.json"),
                label_mapping=label_to_cls,
                split=split,
            )
            for image_path, bboxes, labels in tqdm(dataset_gen):
                with image_path.open("rb") as f:
                    serialized_example = encode_example(f.read(), bboxes, labels)
                    writer.write(serialized_example)
    # test decoding, remove this later
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        result = {}
        # example.features.feature is the dictionary
        for key, feature in example.features.feature.items():
            kind = feature.WhichOneof("kind")
            result[key] = np.array(getattr(feature, kind).value)

        print(result.keys())
        break


if __name__ == "__main__":
    main()
